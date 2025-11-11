"""Utilities for downloading large models from Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.utils.tqdm import disable_progress_bars


def download_model(
    model_id: str,
    local_dir: str | Path,
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    max_workers: int | None = None,
    allow_patterns: Optional[list[str]] = None,
    resume_download: bool = True,
    hf_transfer: bool = True,
) -> Path:
    """Download a model snapshot to ``local_dir``.

    Parameters
    ----------
    model_id:
        Repository ID on the Hugging Face Hub (e.g., ``meta-llama/Meta-Llama-3.1-8B-Instruct``).
    local_dir:
        Target directory for the snapshot.
    revision:
        Optional commit hash or branch to pin the download.
    token:
        Hugging Face access token. If omitted, the function will read ``HF_TOKEN``
        from environment variables.
    max_workers:
        Limit the number of concurrent downloads. Defaults to hub default.
    allow_patterns:
        Optional subset of files to download. Useful for fetching only safetensors
        weights or tokenizer files.
    resume_download:
        Whether to resume partial downloads (defaults to ``True``).
    hf_transfer:
        Enable high-performance uploader/downloader when available.
    """

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    resolved_token = token or os.getenv("HF_TOKEN")
    if not resolved_token:
        raise ValueError(
            "No Hugging Face token provided. Set the HF_TOKEN environment variable "
            "or pass the token argument explicitly."
        )

    if hf_transfer:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    disable_progress_bars()  # keep logs cleaner for large downloads

    snapshot_path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision=revision,
        token=resolved_token,
        resume_download=resume_download,
        max_workers=max_workers,
        allow_patterns=allow_patterns,
    )

    return Path(snapshot_path)
