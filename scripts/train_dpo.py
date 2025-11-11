"""Train LLaMA 3.1 8B with combined forward/backward DPO."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from accelerate import Accelerator
from datasets import Dataset
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch

from reasoning_lab.config import load_config
from reasoning_lab.data.forward_backward_dataset import join_forward_backward, to_dpo_dataset
from reasoning_lab.env import load_project_env
from reasoning_lab.training.weighted_dpo_trainer import WeightedDPOTrainer

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


def _prepare_dataset(cfg) -> Dataset:
    include_backward = cfg.dpo.preference_source != "forward-only"
    if cfg.dpo.preference_source == "backward-only":
        include_backward = True

    forward_path = Path(cfg.data.forward_dataset_path)
    backward_path = Path(cfg.data.backward_dataset_path)

    missing = [str(p) for p in (forward_path, backward_path) if not Path(p).exists()]
    if missing:
        hint = (
            "Run `python scripts/bootstrap_pairs.py --output-dir data/processed` and rename the"
            " outputs to match the config (forward_reasoning.jsonl/backward_reasoning.jsonl)."
        )
        raise FileNotFoundError(
            "Missing required dataset files: " + ", ".join(missing) + f". Hint: {hint}"
        )

    examples = join_forward_backward(
        forward_path,
        backward_path,
        forward_weight=cfg.dpo.forward_weight,
        backward_weight=cfg.dpo.backward_weight,
    )

    dataset = to_dpo_dataset(examples, include_backward=include_backward)

    if cfg.dpo.preference_source == "backward-only":
        dataset = dataset.filter(lambda row: row["task_type"] == "backward")
    elif cfg.dpo.preference_source == "forward-only":
        dataset = dataset.filter(lambda row: row["task_type"] == "forward")

    return dataset


def _create_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model,
        use_fast=True,
        padding_side=cfg.model.get("tokenizer_padding_side", "left"),
        trust_remote_code=cfg.model.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _create_model(cfg, tokenizer, *, device_map, torch_dtype):
    load_kwargs = {
        "use_flash_attention_2": cfg.model.get("flash_attention", "auto") == "auto",
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }
    if cfg.model.get("load_in_8bit", False):
        load_kwargs["load_in_8bit"] = True
    if cfg.model.get("load_in_4bit", False):
        load_kwargs["load_in_4bit"] = True
    return AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        trust_remote_code=cfg.model.get("trust_remote_code", False),
        **load_kwargs,
    )


@app.command()
def main(
    config_path: Path = typer.Argument(..., help="Path to training YAML config."),
    hf_token_env: str = typer.Option("HF_TOKEN", help="Environment variable with the HF token."),
    resume_from: Optional[str] = typer.Option(None, help="Optional checkpoint path to resume."),
    device: str = typer.Option(
        "auto",
        help="Compute device to use: auto (default), gpu, or cpu.",
    ),
) -> None:
    load_project_env()
    cfg = load_config(config_path).raw

    hf_token = os.getenv(hf_token_env)
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    device_normalized = device.lower()
    if device_normalized not in {"auto", "gpu", "cpu"}:
        raise typer.BadParameter("device must be one of: auto, gpu, cpu")

    if device_normalized == "gpu" and not torch.cuda.is_available():
        raise typer.BadParameter("GPU selected but no CUDA device is available.")

    if device_normalized == "auto":
        use_cpu = not torch.cuda.is_available()
    else:
        use_cpu = device_normalized == "cpu"

    device_map = {"": "cpu"} if use_cpu else "auto"
    default_dtype = cfg.experiment.get("mixed_precision", "bf16")
    torch_dtype = "float32" if use_cpu else default_dtype

    dtype_alias = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if isinstance(torch_dtype, str):
        torch_dtype = dtype_alias.get(torch_dtype.lower(), torch_dtype)

    accelerator = Accelerator(cpu=use_cpu)
    console.log("Loading dataset...")
    dataset = _prepare_dataset(cfg)
    split = dataset.train_test_split(test_size=0.05, seed=cfg.experiment.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    console.log("Loading tokenizer and model...")
    tokenizer = _create_tokenizer(cfg)
    model = _create_model(cfg, tokenizer, device_map=device_map, torch_dtype=torch_dtype)

    ref_model = None
    if not cfg.dpo.reference_free:
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.base_model,
            trust_remote_code=cfg.model.get("trust_remote_code", False),
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

    use_bf16 = cfg.experiment.mixed_precision == "bf16" and not use_cpu
    use_fp16 = cfg.experiment.mixed_precision == "fp16" and not use_cpu

    training_args = TrainingArguments(
        output_dir=cfg.experiment.output_dir,
        learning_rate=cfg.training.lr,
        per_device_train_batch_size=cfg.data.batch_size_per_device,
        per_device_eval_batch_size=cfg.data.batch_size_per_device,
        gradient_accumulation_steps=cfg.data.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=cfg.training.max_steps,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_steps=cfg.training.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        report_to=["wandb"],
        run_name=cfg.experiment.name,
        resume_from_checkpoint=resume_from or cfg.training.resume_from_checkpoint,
    )

    trainer = WeightedDPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=cfg.dpo.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        loss_type="sigmoid",
        max_length=cfg.data.max_seq_length,
        max_prompt_length=cfg.data.max_seq_length,
        max_target_length=cfg.data.max_seq_length,
    )

    console.log("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(cfg.experiment.output_dir)
    tokenizer.save_pretrained(cfg.experiment.output_dir)
    console.log("[green]Training completed.[/green]")


if __name__ == "__main__":
    app()
