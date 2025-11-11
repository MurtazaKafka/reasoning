"""Configuration utilities for experiment YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


@dataclass
class ExperimentConfig:
    """Wrapper around OmegaConf dictionaries with convenience helpers."""

    raw: OmegaConf

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

    conf = OmegaConf.load(path)
    return ExperimentConfig(raw=conf)
