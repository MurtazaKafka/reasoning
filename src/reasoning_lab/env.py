"""Environment helpers for reasoning lab."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

from dotenv import load_dotenv

PathLike = Union[str, Path]


def _candidate_paths(extra_paths: Iterable[PathLike] | None = None) -> list[Path]:
    paths: list[Path] = []
    if extra_paths:
        for item in extra_paths:
            candidate = Path(item).expanduser().resolve()
            if candidate not in paths:
                paths.append(candidate)
    default_path = Path.cwd() / ".env"
    if default_path not in paths:
        paths.append(default_path)
    return paths


_env_loaded = False


def load_project_env(extra_paths: Iterable[PathLike] | None = None) -> None:
    """Load the project .env once per process.

    Parameters
    ----------
    extra_paths:
        Optional iterable of additional paths to check before the default ``.env``.
    """

    global _env_loaded
    if _env_loaded:
        return

    for path in _candidate_paths(extra_paths):
        if path.exists():
            load_dotenv(path, override=False)
            _env_loaded = True
            return

    _env_loaded = True
