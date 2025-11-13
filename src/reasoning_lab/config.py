"""Configuration utilities for experiment YAML files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf


@dataclass
class ExperimentConfig:
    """Wrapper around OmegaConf dictionaries with convenience helpers."""

    raw: DictConfig | ListConfig

    @property
    def output_dir(self) -> Path:
        return Path(OmegaConf.to_container(self.raw.experiment, resolve=True)["output_dir"]).expanduser()

    def to_dict(self) -> Dict[str, Any]:
        return OmegaConf.to_container(self.raw, resolve=True)  # type: ignore[return-value]


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    _ensure_env_resolver()
    conf = OmegaConf.load(path)
    return ExperimentConfig(raw=conf)


_ENV_RESOLVER_REGISTERED = False


def _env_resolver(key: Any, default: Any | None = None) -> Any:
    key_str = str(key)
    if key_str in os.environ:
        return os.environ[key_str]
    if default is not None:
        return default

    raise ValueError(
        f"Environment variable '{key_str}' is required but not set. "
        "Set it or provide a default in the config via ${env:VAR,default}."
    )


def _ensure_env_resolver() -> None:
    global _ENV_RESOLVER_REGISTERED
    if _ENV_RESOLVER_REGISTERED:
        return

    registered = False
    resolver_kwargs = {"use_cache": False}

    if hasattr(OmegaConf, "register_new_resolver"):
        try:
            OmegaConf.register_new_resolver("env", _env_resolver, **resolver_kwargs)
            registered = OmegaConf.has_resolver("env")
        except TypeError:
            # Older versions may not accept kwargs such as use_cache.
            OmegaConf.register_new_resolver("env", _env_resolver)
            registered = OmegaConf.has_resolver("env")

    if not registered and hasattr(OmegaConf, "register_resolver"):
        try:
            OmegaConf.register_resolver("env", _env_resolver, **resolver_kwargs)
        except TypeError:
            OmegaConf.register_resolver("env", _env_resolver)
        registered = OmegaConf.has_resolver("env")

    _ENV_RESOLVER_REGISTERED = registered


_ensure_env_resolver()
