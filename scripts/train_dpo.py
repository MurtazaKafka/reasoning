"""Train LLaMA 3.1 8B with combined forward/backward DPO."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from datasets import Dataset
from rich.console import Console
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.dpo_config import DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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

    missing = [str(p) for p in (forward_path, backward_path) if not p.exists()]
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


def _is_cpu_only(device_map) -> bool:
    return isinstance(device_map, dict) and set(device_map.values()) == {"cpu"}


def _resolve_report_to(cfg) -> list[str]:
    report_to = cfg.experiment.get("report_to", ["wandb"])
    if isinstance(report_to, str):
        report_to = [report_to]
    if not isinstance(report_to, (list, tuple)):
        console.log("[yellow]experiment.report_to must be a string or list; defaulting to [][/yellow]")
        report_to = []

    normalized: list[str] = []
    for entry in report_to:
        if not entry:
            continue
        normalized.append(str(entry))

    if any(item.lower() == "wandb" for item in normalized):
        # Check if wandb is disabled via environment variable
        wandb_disabled_env = os.getenv("WANDB_DISABLED", "").upper()
        if wandb_disabled_env in ("TRUE", "1", "YES"):
            console.log("[yellow]WANDB_DISABLED is set; disabling Weights & Biases logging.[/yellow]")
            normalized = [item for item in normalized if item.lower() != "wandb"]
        else:
            # Check if wandb is installed
            try:
                import wandb  # noqa: F401
            except ImportError:
                console.log("[yellow]wandb not installed; disabling Weights & Biases logging.[/yellow]")
                normalized = [item for item in normalized if item.lower() != "wandb"]

    return normalized


def _is_flash_attention_available() -> bool:
    """Return True if flash-attn is installed and importable."""

    spec = importlib.util.find_spec("flash_attn")
    if spec is None:
        return False
    try:
        importlib.import_module("flash_attn")
        return True
    except Exception:
        return False


def _resolve_attention_kwargs(cfg, *, allow_flash: bool) -> dict:
    flash_pref = cfg.model.get("flash_attention", "auto")
    if flash_pref is None:
        return {}

    desired_impl: Optional[str] = None
    explicit_request = False

    if isinstance(flash_pref, bool):
        explicit_request = True
        if flash_pref:
            desired_impl = "flash_attention_2"
    else:
        value = str(flash_pref).strip().lower()
        if value in {"off", "disable", "disabled", "none", "false"}:
            return {}
        if value in {"sdpa", "scaled_dot_product"}:
            explicit_request = True
            return {"attn_implementation": "sdpa"}
        if value in {"eager", "standard"}:
            explicit_request = True
            return {"attn_implementation": "eager"}
        if value in {"flash", "flash2", "flash_attention", "flash_attention_2"}:
            explicit_request = True
            desired_impl = "flash_attention_2"
        elif value not in {"auto", ""}:
            console.log(f"[yellow]Unknown flash_attention value '{value}'; defaulting to auto.[/yellow]")

    if desired_impl == "flash_attention_2":
        if not allow_flash:
            if explicit_request:
                console.log(
                    "[yellow]FlashAttention requested but the current run is CPU-only. Falling back to SDPA.[/yellow]"
                )
            return {"attn_implementation": "sdpa"} if explicit_request else {}

        if _is_flash_attention_available():
            return {"attn_implementation": "flash_attention_2"}

        install_hint = (
            "pip install flash-attn --no-build-isolation"
            "  # requires CUDA 12+, compatible GPU, PyTorch 2.2+"
        )
        console.log(
            "[yellow]FlashAttention requested but the optional 'flash_attn' package isn't installed. "
            f"Falling back to SDPA. Install via: {install_hint}[/yellow]"
        )
        return {"attn_implementation": "sdpa"}

    if allow_flash and _is_flash_attention_available():
        return {"attn_implementation": "flash_attention_2"}

    return {}


def _build_model_load_kwargs(cfg, *, device_map, torch_dtype, include_quantization: bool) -> dict:
    allow_flash = not _is_cpu_only(device_map)
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }
    # Prefer safetensors checkpoints when available to avoid torch.load security check requirements on older PyTorch versions.
    if cfg.model.get("use_safetensors", True):
        load_kwargs.setdefault("use_safetensors", True)
    load_kwargs.update(_resolve_attention_kwargs(cfg, allow_flash=allow_flash))

    if include_quantization and allow_flash:
        if cfg.model.get("load_in_8bit", False):
            load_kwargs["load_in_8bit"] = True
        if cfg.model.get("load_in_4bit", False):
            load_kwargs["load_in_4bit"] = True
    elif include_quantization and cfg.model.get("load_in_8bit", False):
        console.log(
            "[yellow]Skipping 8-bit loading because the run is configured for CPU-only execution.[/yellow]"
        )
    elif include_quantization and cfg.model.get("load_in_4bit", False):
        console.log(
            "[yellow]Skipping 4-bit loading because the run is configured for CPU-only execution.[/yellow]"
        )

    return load_kwargs


def _should_enable_quantization(cfg) -> bool:
    wants_8bit = bool(cfg.model.get("load_in_8bit", False))
    wants_4bit = bool(cfg.model.get("load_in_4bit", False))
    if not (wants_8bit or wants_4bit):
        return False

    adapter_present = cfg.model.get("adapter") not in {None, "", "null"}
    if adapter_present:
        return True

    console.log(
        "[yellow]Quantized loading requested but no adapter is configured. "
        "Full-precision loading will be used so the model can be fine-tuned without PEFT.[/yellow]"
    )
    return False


def _create_lora_config(cfg):
    """Create LoRA configuration from config or defaults."""
    lora_cfg = cfg.get("lora", {})
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


def _create_model(cfg, tokenizer, *, device_map, torch_dtype):
    load_kwargs = _build_model_load_kwargs(
        cfg,
        device_map=device_map,
        torch_dtype=torch_dtype,
        include_quantization=_should_enable_quantization(cfg),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        trust_remote_code=cfg.model.get("trust_remote_code", False),
        **load_kwargs,
    )
    
    # Apply LoRA if adapter is configured or if using quantization
    use_lora = cfg.model.get("adapter") == "lora" or cfg.get("lora", {}).get("enabled", False)
    if use_lora or _should_enable_quantization(cfg):
        console.log("[cyan]Applying LoRA adapter for memory-efficient training...[/cyan]")
        if load_kwargs.get("load_in_8bit") or load_kwargs.get("load_in_4bit"):
            model = prepare_model_for_kbit_training(model)
        lora_config = _create_lora_config(cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


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
        "fp16": torch.float16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if isinstance(torch_dtype, str):
        torch_dtype = dtype_alias.get(torch_dtype.lower(), torch_dtype)

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
            **_build_model_load_kwargs(
                cfg,
                device_map=device_map,
                torch_dtype=torch_dtype,
                include_quantization=False,
            ),
        )

    use_bf16 = cfg.experiment.mixed_precision == "bf16" and not use_cpu
    use_fp16 = cfg.experiment.mixed_precision == "fp16" and not use_cpu

    dpo_args = DPOConfig(
        output_dir=cfg.experiment.output_dir,
        beta=cfg.dpo.beta,
        label_smoothing=cfg.dpo.label_smoothing,
        reference_free=cfg.dpo.reference_free,
        per_device_train_batch_size=cfg.data.batch_size_per_device,
        per_device_eval_batch_size=cfg.data.batch_size_per_device,
        gradient_accumulation_steps=cfg.data.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=cfg.training.max_steps if cfg.training.max_steps is not None else -1,
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_strategy="steps",
        logging_steps=cfg.training.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    report_to=_resolve_report_to(cfg),
        run_name=cfg.experiment.name,
        resume_from_checkpoint=resume_from or cfg.training.resume_from_checkpoint,
        max_length=cfg.data.max_seq_length,
        max_prompt_length=cfg.data.max_seq_length,
        max_completion_length=cfg.data.max_seq_length,
        loss_type=["sigmoid"],
        use_weighting=True,
    )

    trainer = WeightedDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    console.log("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from or cfg.training.resume_from_checkpoint)
    trainer.save_model(cfg.experiment.output_dir)
    tokenizer.save_pretrained(cfg.experiment.output_dir)
    console.log("[green]Training completed.[/green]")


if __name__ == "__main__":
    app()
