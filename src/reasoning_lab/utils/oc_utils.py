from __future__ import annotations

from typing import Any, Mapping

from omegaconf import OmegaConf


def ensure_native_lora_cfg(lora_cfg: Any) -> Mapping[str, Any]:
    """Return a native-Python mapping for a LoRA cfg that may be an OmegaConf object.

    This ensures lists, ints, and floats are plain Python types which are safe to
    serialize to JSON (e.g., when PEFT writes adapter_config.json).
    """
    if OmegaConf.is_config(lora_cfg):
        lora_cfg = OmegaConf.to_container(lora_cfg, resolve=True)

    # If the caller passed something other than a mapping, try to coerce
    if not isinstance(lora_cfg, Mapping):
        return {}

    # Ensure target_modules is a plain list
    target_modules = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    if not isinstance(target_modules, list):
        try:
            target_modules = list(target_modules)
        except Exception:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Build a sanitized dict with plain Python primitives
    sanitized = dict(lora_cfg)
    sanitized["target_modules"] = target_modules
    # Cast numeric fields to native types if present
    if "r" in sanitized:
        try:
            sanitized["r"] = int(sanitized["r"])
        except Exception:
            pass
    if "alpha" in sanitized:
        try:
            sanitized["alpha"] = int(sanitized["alpha"])
        except Exception:
            pass
    if "dropout" in sanitized:
        try:
            sanitized["dropout"] = float(sanitized["dropout"])
        except Exception:
            pass

    return sanitized
