"""Generate high-quality DPO training pairs with rejection sampling.

This script generates better preference pairs by:
1. Using rejection sampling to get actual incorrect traces (not synthetic)
2. Generating multiple samples per question for self-consistency
3. Computing proper verdicts based on ground truth comparison
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.progress import track

from reasoning_lab.env import load_project_env
from reasoning_lab.inference.dual_reasoner import DualReasoner
from reasoning_lab.utils.text import (
    extract_final_answer,
    extract_ground_truth_answer,
    answers_match,
    extract_verification,
)

app = typer.Typer(add_completion=False, help=__doc__)
console: Any = Console()


@app.command()
def main(
    dataset_name: str = typer.Argument("openai/gsm8k", help="Source dataset to augment."),
    dataset_config: Optional[str] = typer.Option(
        "main", "--dataset-config", help="Optional dataset config/subset."
    ),
    split: str = typer.Option("train", help="Dataset split to use."),
    limit: Optional[int] = typer.Option(None, help="Optionally limit number of samples."),
    teacher_model: str = typer.Option(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Teacher model for generating traces.",
    ),
    torch_dtype: str = typer.Option("auto", help="Torch dtype (auto, bf16, fp16, float32)."),
    device_map: str = typer.Option("auto", help="Device placement strategy."),
    load_in_8bit: bool = typer.Option(False, help="Load in 8-bit quantization."),
    load_in_4bit: bool = typer.Option(False, help="Load in 4-bit quantization."),
    attn_implementation: Optional[str] = typer.Option(None, help="Attention implementation."),
    max_forward_tokens: int = typer.Option(512, help="Max tokens for forward reasoning."),
    max_backward_tokens: int = typer.Option(256, help="Max tokens for backward reasoning."),
    forward_temperature: float = typer.Option(0.7, help="Forward sampling temperature."),
    backward_temperature: float = typer.Option(0.3, help="Backward sampling temperature."),
    output_dir: Path = typer.Option(Path("data/processed"), help="Output directory."),
    hf_token_env: str = typer.Option("HF_TOKEN", help="Environment variable with HF token."),
    # Rejection sampling options
    rejection_sampling: bool = typer.Option(
        True, help="Use rejection sampling for negative examples."
    ),
    max_rejection_attempts: int = typer.Option(
        5, help="Max attempts for rejection sampling."
    ),
    # Multi-sample options
    samples_per_question: int = typer.Option(
        1, help="Number of samples per question (for self-consistency)."
    ),
    # Prompt templates
    forward_template: Optional[Path] = typer.Option(
        None, help="Path to forward prompt template."
    ),
    backward_template: Optional[Path] = typer.Option(
        None, help="Path to backward prompt template."
    ),
) -> None:
    """Generate DPO training pairs with optional rejection sampling."""
    load_project_env()
    from datasets import load_dataset

    token = os.getenv(hf_token_env)

    # Load prompt templates if provided
    fwd_template = str(forward_template) if forward_template else None
    bwd_template = str(backward_template) if backward_template else None

    reasoner = DualReasoner(
        teacher_model,
        hf_token=token,
        torch_dtype=torch_dtype,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        attn_implementation=attn_implementation,
        max_forward_tokens=max_forward_tokens,
        max_backward_tokens=max_backward_tokens,
        forward_sampling={
            "temperature": forward_temperature,
            "top_p": 0.95,
            "do_sample": forward_temperature > 0,
        },
        backward_sampling={
            "temperature": backward_temperature,
            "top_p": 0.9,
            "do_sample": backward_temperature > 0,
        },
        forward_prompt_template=fwd_template,
        backward_prompt_template=bwd_template,
    )

    # Load dataset
    load_kwargs = {}
    if dataset_config:
        load_kwargs["name"] = dataset_config

    ds: Any = load_dataset(dataset_name, split=split, **load_kwargs)
    if limit:
        ds = ds.select(range(limit))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    dpo_pairs_path = output_dir / "dpo_pairs.jsonl"
    stats_path = output_dir / "generation_stats.json"

    stats = {
        "total_questions": 0,
        "correct_traces": 0,
        "incorrect_traces": 0,
        "rejection_sampling_successes": 0,
        "rejection_sampling_failures": 0,
        "backward_calibrated": 0,
        "backward_miscalibrated": 0,
    }

    with dpo_pairs_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(track(ds, description="Generating DPO pairs")):
            question = sample.get("question") or sample.get("input")
            answer = sample.get("answer", "")
            ground_truth = extract_ground_truth_answer(answer)

            stats["total_questions"] += 1

            # Generate samples
            correct_sample = None
            incorrect_sample = None

            for _ in range(samples_per_question):
                # Try to get a correct sample
                if correct_sample is None:
                    correct_sample = reasoner.generate_with_rejection_sampling(
                        question,
                        ground_truth,
                        require_correct=True,
                        max_attempts=max_rejection_attempts if rejection_sampling else 1,
                    )

                # Try to get an incorrect sample for better contrastive signal
                if rejection_sampling and incorrect_sample is None:
                    incorrect_sample = reasoner.generate_with_rejection_sampling(
                        question,
                        ground_truth,
                        require_incorrect=True,
                        max_attempts=max_rejection_attempts,
                    )

            # Build DPO pair
            record = {
                "id": sample.get("id", idx),
                "question": question,
                "ground_truth": ground_truth,
            }

            if correct_sample:
                stats["correct_traces"] += 1
                stats["rejection_sampling_successes"] += 1
                record["chosen_forward"] = correct_sample.forward_trace
                record["chosen_answer"] = correct_sample.extracted_answer
                record["chosen_backward"] = correct_sample.backward_trace
                record["chosen_verification"] = correct_sample.verification

                # Check backward calibration
                is_pass = "pass" in correct_sample.verification.lower()
                if is_pass:
                    stats["backward_calibrated"] += 1
                else:
                    stats["backward_miscalibrated"] += 1
            else:
                stats["rejection_sampling_failures"] += 1
                # Fall back to any sample
                fallback = reasoner.generate(question)
                record["chosen_forward"] = fallback.forward_trace
                record["chosen_answer"] = fallback.extracted_answer
                record["chosen_backward"] = fallback.backward_trace
                record["chosen_verification"] = fallback.verification

            if incorrect_sample:
                stats["incorrect_traces"] += 1
                record["rejected_forward"] = incorrect_sample.forward_trace
                record["rejected_answer"] = incorrect_sample.extracted_answer
                record["rejected_backward"] = incorrect_sample.backward_trace
                record["rejected_verification"] = incorrect_sample.verification

                # Check backward calibration on incorrect
                is_fail = "fail" in incorrect_sample.verification.lower()
                if is_fail:
                    stats["backward_calibrated"] += 1
                else:
                    stats["backward_miscalibrated"] += 1
            else:
                # Use synthetic rejected if rejection sampling failed
                record["rejected_forward"] = (
                    f"I'm not sure how to solve this problem correctly. "
                    f"Let me attempt: {question[:100]}..."
                )
                record["rejected_answer"] = ""
                record["rejected_backward"] = "Verification: FAIL\nUnable to verify solution."
                record["rejected_verification"] = "Verification: FAIL"

            # Compute quality scores
            record["forward_correct"] = correct_sample is not None
            record["has_real_negative"] = incorrect_sample is not None
            record["quality_score"] = (
                1.0 if (correct_sample and incorrect_sample)
                else 0.7 if correct_sample
                else 0.3
            )

            f.write(json.dumps(record) + "\n")

    # Save stats
    stats_path.write_text(json.dumps(stats, indent=2))

    # Print summary
    console.print(f"\n[bold]Generation Complete[/bold]")
    console.print(f"  Total questions: {stats['total_questions']}")
    console.print(f"  Correct traces: {stats['correct_traces']}")
    console.print(f"  Incorrect traces (via rejection): {stats['incorrect_traces']}")
    console.print(f"  Backward calibrated: {stats['backward_calibrated']}")
    console.print(f"  Backward miscalibrated: {stats['backward_miscalibrated']}")

    if stats["total_questions"] > 0:
        console.print(f"\n[bold]Rates[/bold]")
        console.print(f"  Correct rate: {stats['correct_traces'] / stats['total_questions']:.1%}")
        total_backward = stats['backward_calibrated'] + stats['backward_miscalibrated']
        if total_backward > 0:
            console.print(f"  Backward calibration: {stats['backward_calibrated'] / total_backward:.1%}")

    console.print(f"\n[green]Saved to {dpo_pairs_path}[/green]")
    console.print(f"[green]Stats saved to {stats_path}[/green]")


if __name__ == "__main__":
    app()
