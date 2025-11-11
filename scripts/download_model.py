"""CLI for downloading LLaMA 3.1 8B (or related) weights locally."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from reasoning_lab.env import load_project_env
from reasoning_lab.io.model_downloader import download_model

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


def _print_summary(model_id: str, destination: Path, revision: Optional[str]) -> None:
    table = Table(title="Download Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Model", model_id)
    table.add_row("Revision", revision or "latest")
    table.add_row("Destination", str(destination))
    console.print(table)


@app.command()
def main(
    model_id: str = typer.Argument(
        ..., help="Hub repository ID, e.g., meta-llama/Meta-Llama-3.1-8B-Instruct"
    ),
    destination: Path = typer.Option(
        Path("models/llama-3.1-8b"),
        "--destination",
        "-d",
        help="Where to store the downloaded snapshot.",
    ),
    revision: Optional[str] = typer.Option(
        None,
        "--revision",
        "-r",
        help="Optional commit hash or branch to pin the download.",
    ),
    token_env: str = typer.Option(
        "HF_TOKEN",
        "--token-env",
        help="Environment variable containing the Hugging Face token.",
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        help="Maximum parallel download threads. Defaults to huggingface-hub settings.",
    ),
    safetensors_only: bool = typer.Option(
        True,
        help="Download only *.safetensors and tokenizer files to save space.",
    ),
    resume: bool = typer.Option(True, help="Resume partial downloads if present."),
) -> None:
    """Download model weights from the Hugging Face Hub."""

    load_project_env()
    token = os.getenv(token_env)
    if not token:
        raise typer.BadParameter(
            f"Environment variable {token_env} is not set. Export your Hugging Face token first."
        )

    allow_patterns = None
    if safetensors_only:
        allow_patterns = ["*.json", "*.safetensors", "tokenizer.*", "*.model"]

    snapshot_path = download_model(
        model_id=model_id,
        local_dir=destination,
        revision=revision,
        token=token,
        max_workers=max_workers,
        allow_patterns=allow_patterns,
        resume_download=resume,
    )

    _print_summary(model_id, snapshot_path, revision)
    console.print("[green]Download complete.[/green]")


if __name__ == "__main__":
    app()
