#!/usr/bin/env python3
"""
Fix corrupted/empty adapter_config.json in a LoRA checkpoint.

This script recreates the adapter_config.json based on the training config
when the original file is missing or empty.

Usage:
    python scripts/fix_adapter_config.py ./outputs/runs/llama31_8b_forward_backward_dpo/checkpoint-200
"""

import json
import sys
from pathlib import Path


def create_adapter_config(
    checkpoint_dir: str,
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> None:
    """Create adapter_config.json for a LoRA checkpoint."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    checkpoint_path = Path(checkpoint_dir)
    config_path = checkpoint_path / "adapter_config.json"

    # Standard PEFT LoRA config format
    adapter_config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": r,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }

    # Backup existing file if it exists and has content
    if config_path.exists() and config_path.stat().st_size > 0:
        backup_path = config_path.with_suffix(".json.bak")
        config_path.rename(backup_path)
        print(f"Backed up existing config to: {backup_path}")

    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"Created adapter_config.json at: {config_path}")
    print(f"Config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Target modules: {target_modules}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/fix_adapter_config.py <checkpoint_dir>")
        print("Example: python scripts/fix_adapter_config.py ./outputs/runs/llama31_8b_forward_backward_dpo/checkpoint-200")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]

    # Verify checkpoint exists
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Check for adapter weights
    weights_path = checkpoint_path / "adapter_model.safetensors"
    if not weights_path.exists():
        weights_path = checkpoint_path / "adapter_model.bin"
        if not weights_path.exists():
            print(f"Error: No adapter weights found in {checkpoint_dir}")
            print("Expected: adapter_model.safetensors or adapter_model.bin")
            sys.exit(1)

    print(f"Found adapter weights: {weights_path}")

    # Create config with settings matching dpo_hybrid.yaml
    create_adapter_config(
        checkpoint_dir,
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    print("\nDone! You can now load the adapter with:")
    print(f"  python scripts/eval_reasoning.py configs/eval_gsm8k.yaml")


if __name__ == "__main__":
    main()
