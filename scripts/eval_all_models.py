#!/usr/bin/env python3 -u
"""Optimized batch evaluation script that keeps model loaded and swaps LoRA adapters.

This script is much faster than evaluating models separately because:
1. Base model is loaded only once
2. LoRA adapters are hot-swapped without reloading
3. Checkpoints allow resuming if interrupted
4. Real-time progress logging

Usage:
    python -u scripts/eval_all_models.py --output-dir ./experiment_outputs/evals

For cluster with nohup:
    nohup python -u scripts/eval_all_models.py --output-dir ./experiment_outputs/evals > eval.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force unbuffered output for nohup
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reasoning_lab.env import load_project_env
from reasoning_lab.utils.text import extract_final_answer, extract_ground_truth_answer, answers_match

console = Console()


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both file and console with real-time output."""
    log_dir = output_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_batch_{timestamp}.log"

    logger = logging.getLogger("eval_batch")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler - unbuffered
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler with immediate flush
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Custom emit to force flush after each log
    original_emit = ch.emit
    def flushing_emit(record):
        original_emit(record)
        ch.flush()
        sys.stdout.flush()
    ch.emit = flushing_emit

    # Also flush file handler
    original_fh_emit = fh.emit
    def flushing_fh_emit(record):
        original_fh_emit(record)
        fh.flush()
    fh.emit = flushing_fh_emit

    logger.info(f"Logging to: {log_file}")
    print(f"Logging to: {log_file}", flush=True)
    return logger


class AdapterSwappableReasoner:
    """Reasoner that can hot-swap LoRA adapters without reloading base model."""

    def __init__(
        self,
        base_model_path: str,
        *,
        device_map: str = "auto",
        torch_dtype: str = "bf16",
        hf_token: Optional[str] = None,
        load_in_8bit: bool = True,
        max_forward_tokens: int = 512,
        max_backward_tokens: int = 256,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        self.base_model_path = base_model_path
        self.max_forward_tokens = max_forward_tokens
        self.max_backward_tokens = max_backward_tokens
        self.current_adapter: Optional[str] = None

        console.log(f"[bold blue]Loading base model: {base_model_path}[/bold blue]")

        # Resolve dtype
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        resolved_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            token=hf_token,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        quant_config = None
        if load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device_map,
            torch_dtype=resolved_dtype,
            token=hf_token,
            quantization_config=quant_config,
        )
        self.base_model.eval()

        # Track current model (base or with adapter)
        self.model = self.base_model
        self._peft_model = None

        # Default prompts
        self.forward_prompt = """You are an expert problem solver. Solve the following problem step by step.
Show your reasoning clearly, then provide the final answer.

Problem: {question}

Solution:"""

        self.backward_prompt = """You are a careful verifier. Given a problem and a candidate answer, verify whether the answer is correct by reasoning backwards from the answer.

Problem: {question}
Candidate Answer: {answer}

Verify the solution step by step, then conclude with either:
- Verification: PASS (if the answer is correct)
- Verification: FAIL (if the answer is incorrect)

Verification:"""

        self._device = next(self.base_model.parameters()).device
        console.log("[green]Base model loaded successfully[/green]")

    def load_adapter(self, adapter_path: Optional[str], logger: logging.Logger) -> None:
        """Load or swap to a different LoRA adapter. Pass None for base model."""
        from peft import PeftModel

        if adapter_path == self.current_adapter:
            logger.info(f"Adapter already loaded: {adapter_path or 'baseline'}")
            return

        if adapter_path is None:
            # Use base model without adapter
            logger.info("Switching to baseline (no adapter)")
            self.model = self.base_model
            self._peft_model = None
            self.current_adapter = None
            return

        adapter_path_obj = Path(adapter_path)
        if not adapter_path_obj.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        adapter_config = adapter_path_obj / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

        logger.info(f"Loading adapter: {adapter_path}")

        # Always create fresh PeftModel from base
        # This is simpler and more reliable than trying to swap adapters
        self._peft_model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
        )
        self.model = self._peft_model
        self.current_adapter = adapter_path
        logger.info(f"  ✓ Adapter loaded: {adapter_path_obj.name}")

    def generate(self, question: str, greedy: bool = True) -> Dict[str, Any]:
        """Generate forward and backward traces for a question."""
        # Forward pass
        forward_prompt = self.forward_prompt.format(question=question)
        forward_inputs = self.tokenizer(
            forward_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._device)

        gen_kwargs = {
            "max_new_tokens": self.max_forward_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if greedy:
            gen_kwargs.update({"do_sample": False, "temperature": None})
        else:
            gen_kwargs.update({"do_sample": True, "temperature": 0.7, "top_p": 0.95})

        with torch.inference_mode():
            forward_outputs = self.model.generate(**forward_inputs, **gen_kwargs)

        forward_text = self.tokenizer.decode(
            forward_outputs[0][forward_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Extract answer from forward trace
        extracted_answer = extract_final_answer(forward_text)

        # Backward pass
        backward_prompt = self.backward_prompt.format(
            question=question,
            answer=extracted_answer or "unknown",
        )
        backward_inputs = self.tokenizer(
            backward_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._device)

        backward_kwargs = gen_kwargs.copy()
        backward_kwargs["max_new_tokens"] = self.max_backward_tokens

        with torch.inference_mode():
            backward_outputs = self.model.generate(**backward_inputs, **backward_kwargs)

        backward_text = self.tokenizer.decode(
            backward_outputs[0][backward_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Extract verification
        verification = "UNKNOWN"
        if "PASS" in backward_text.upper():
            verification = "PASS"
        elif "FAIL" in backward_text.upper():
            verification = "FAIL"

        return {
            "forward_trace": forward_text,
            "backward_trace": backward_text,
            "extracted_answer": extracted_answer,
            "verification": verification,
        }


def load_eval_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load evaluation checkpoint for resume capability."""
    if checkpoint_path.exists():
        return json.loads(checkpoint_path.read_text())
    return {"completed_models": {}, "partial_results": {}}


def save_eval_checkpoint(checkpoint_path: Path, data: Dict[str, Any]) -> None:
    """Save evaluation checkpoint."""
    checkpoint_path.write_text(json.dumps(data, indent=2))


def evaluate_model(
    reasoner: AdapterSwappableReasoner,
    model_name: str,
    adapter_path: Optional[str],
    dataset,
    max_samples: int,
    logger: logging.Logger,
    checkpoint_data: Dict[str, Any],
    checkpoint_path: Path,
    output_file: Path,
) -> Dict[str, Any]:
    """Evaluate a single model with checkpointing."""
    from reasoning_lab.metrics.reasoning_metrics import (
        compute_accuracy,
        compute_acknowledgement_rate,
        compute_false_positive_rate,
        compute_verification_calibration,
    )

    # Load adapter
    reasoner.load_adapter(adapter_path, logger)

    # Check for partial results
    partial_key = f"partial_{model_name}"
    samples = checkpoint_data.get("partial_results", {}).get(partial_key, [])
    start_idx = len(samples)

    if start_idx > 0:
        logger.info(f"Resuming from sample {start_idx}/{max_samples}")

    # Limit samples
    total_samples = min(len(dataset), max_samples)

    logger.info(f"Evaluating {model_name}: {total_samples} samples")

    for idx in range(start_idx, total_samples):
        example = dataset[idx]
        question = example.get("question") or example.get("input")
        answer = example.get("answer", "")
        ground_truth = extract_ground_truth_answer(answer)

        try:
            outputs = reasoner.generate(question, greedy=True)

            predicted = outputs["extracted_answer"]
            is_correct = answers_match(predicted, ground_truth)

            sample_data = {
                "idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "is_correct": is_correct,
                "verification": outputs["verification"],
                "forward_trace": outputs["forward_trace"],
                "backward_trace": outputs["backward_trace"],
                "answer": answer,
            }
            samples.append(sample_data)

            # Log progress every 10 samples
            if (idx + 1) % 10 == 0:
                correct_so_far = sum(1 for s in samples if s.get("is_correct", False))
                logger.info(f"  [{idx+1}/{total_samples}] Accuracy so far: {correct_so_far}/{len(samples)} = {correct_so_far/len(samples):.1%}")

            # Save checkpoint every 50 samples
            if (idx + 1) % 50 == 0:
                checkpoint_data.setdefault("partial_results", {})[partial_key] = samples
                save_eval_checkpoint(checkpoint_path, checkpoint_data)
                logger.info(f"  Checkpoint saved at sample {idx+1}")

        except Exception as e:
            logger.error(f"  Error on sample {idx}: {e}")
            samples.append({
                "idx": idx,
                "question": question,
                "error": str(e),
                "is_correct": False,
            })

    # Compute metrics
    logger.info(f"Computing metrics for {model_name}...")

    metric_samples = [{
        "forward_trace": s.get("forward_trace", ""),
        "backward_trace": s.get("backward_trace", ""),
        "verification": s.get("verification", ""),
        "answer": s.get("answer", ""),
        "question": s.get("question", ""),
    } for s in samples]

    accuracy = compute_accuracy(metric_samples)
    ack_rate = compute_acknowledgement_rate(metric_samples)
    fpr = compute_false_positive_rate(metric_samples)
    calibration = compute_verification_calibration(metric_samples)

    results = {
        "metadata": {
            "model_name": model_name,
            "adapter_path": adapter_path,
            "num_samples": len(samples),
            "timestamp": datetime.now().isoformat(),
        },
        "results": {
            "gsm8k": {
                "accuracy": {"value": accuracy.value, "details": accuracy.details or {}},
                "acknowledgement_rate": {"value": ack_rate.value, "details": ack_rate.details or {}},
                "false_positive_rate": {"value": fpr.value, "details": fpr.details or {}},
                "verification_calibration": {"value": calibration.value, "details": calibration.details or {}},
            }
        },
    }

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2))
    logger.info(f"  ✓ Results saved to {output_file}")

    # Save predictions
    predictions_file = output_file.with_suffix(".predictions.json")
    predictions_file.write_text(json.dumps({"gsm8k": samples}, indent=2))

    # Clear partial results from checkpoint
    if partial_key in checkpoint_data.get("partial_results", {}):
        del checkpoint_data["partial_results"][partial_key]

    # Mark as completed
    checkpoint_data.setdefault("completed_models", {})[model_name] = True
    save_eval_checkpoint(checkpoint_path, checkpoint_data)

    logger.info(f"  Accuracy: {accuracy.value:.1%}")
    logger.info(f"  Ack Rate: {ack_rate.value:.1%}")
    logger.info(f"  FPR: {fpr.value:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Optimized batch evaluation with adapter swapping"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./experiment_outputs/evals"),
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("./experiment_outputs/runs"),
        help="Directory containing trained model adapters",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model path or HuggingFace repo ID",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples to evaluate per model",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["baseline", "forward_only", "backward_only", "hybrid_60_40"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if interrupted",
    )

    args = parser.parse_args()

    # Setup
    load_project_env()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = args.runs_dir.resolve()
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("OPTIMIZED BATCH EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Max samples: {args.max_samples}")

    # Checkpoint file
    checkpoint_path = output_dir / "eval_checkpoint.json"
    checkpoint_data = {}
    if args.resume:
        checkpoint_data = load_eval_checkpoint(checkpoint_path)
        if checkpoint_data.get("completed_models"):
            logger.info(f"Resuming - already completed: {list(checkpoint_data['completed_models'].keys())}")

    # Load dataset once
    logger.info("Loading GSM8K dataset...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    test_split = ds["test"]
    logger.info(f"Dataset loaded: {len(test_split)} test samples")

    # Get HF token
    hf_token = os.environ.get("HF_TOKEN")

    # Initialize reasoner with base model (loaded only once!)
    start_time = time.time()
    reasoner = AdapterSwappableReasoner(
        args.base_model,
        hf_token=hf_token,
        load_in_8bit=True,
    )
    load_time = time.time() - start_time
    logger.info(f"Base model loaded in {load_time:.1f}s")

    # Build model configs
    model_configs = []
    for model_name in args.models:
        if model_name == "baseline":
            model_configs.append({"name": "baseline", "adapter_path": None})
        else:
            adapter_path = runs_dir / f"{model_name}_dpo"
            if adapter_path.exists():
                model_configs.append({
                    "name": model_name,
                    "adapter_path": str(adapter_path),
                })
            else:
                logger.warning(f"Adapter not found: {adapter_path}")

    # Evaluate each model
    all_results = {}
    for model_cfg in model_configs:
        name = model_cfg["name"]

        # Skip if already completed
        if checkpoint_data.get("completed_models", {}).get(name):
            logger.info(f"Skipping {name} (already completed)")
            # Load existing results
            result_file = output_dir / f"{name}.json"
            if result_file.exists():
                all_results[name] = json.loads(result_file.read_text())
            continue

        logger.info("=" * 60)
        logger.info(f"Evaluating: {name}")
        logger.info("=" * 60)

        try:
            result_file = output_dir / f"{name}.json"
            results = evaluate_model(
                reasoner=reasoner,
                model_name=name,
                adapter_path=model_cfg.get("adapter_path"),
                dataset=test_split,
                max_samples=args.max_samples,
                logger=logger,
                checkpoint_data=checkpoint_data,
                checkpoint_path=checkpoint_path,
                output_file=result_file,
            )
            all_results[name] = results

        except Exception as e:
            logger.error(f"Failed to evaluate {name}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Print summary
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    logger.info("\nResults Summary:")
    for name, results in all_results.items():
        if "results" in results and "gsm8k" in results["results"]:
            gsm8k = results["results"]["gsm8k"]
            acc = gsm8k.get("accuracy", {}).get("value", "N/A")
            ack = gsm8k.get("acknowledgement_rate", {}).get("value", "N/A")
            logger.info(f"  {name}:")
            logger.info(f"    Accuracy: {acc:.1%}" if isinstance(acc, float) else f"    Accuracy: {acc}")
            logger.info(f"    Ack Rate: {ack:.1%}" if isinstance(ack, float) else f"    Ack Rate: {ack}")

    # Cleanup checkpoint on success
    if checkpoint_path.exists() and all(
        checkpoint_data.get("completed_models", {}).get(m["name"]) for m in model_configs
    ):
        checkpoint_path.unlink()
        logger.info("Evaluation checkpoint cleaned up (all models completed)")


if __name__ == "__main__":
    main()
