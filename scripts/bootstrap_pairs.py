"""Generate forward/backward reasoning pairs using a teacher model."""

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

app = typer.Typer(add_completion=False, help=__doc__)
console: Any = Console()


@app.command()
def main(
    dataset_name: str = typer.Argument("openai/gsm8k", help="Source dataset to augment."),
    dataset_config: Optional[str] = typer.Option(
        "main", "--dataset-config", help="Optional dataset config/subset (e.g., main, socratic)."
    ),
    split: str = typer.Option("train", help="Dataset split to use."),
    limit: Optional[int] = typer.Option(None, help="Optionally limit number of samples."),
    teacher_model: str = typer.Option(
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        help="Teacher model for generating forward/backward traces.",
    ),
    output_dir: Path = typer.Option(Path("data/augmented"), help="Where to write JSONL files."),
    hf_token_env: str = typer.Option("HF_TOKEN", help="Environment variable with HF token."),
    forward_filename: str = typer.Option(
        "forward_reasoning.jsonl", help="Filename for forward reasoning records."
    ),
    backward_filename: str = typer.Option(
        "backward_reasoning.jsonl", help="Filename for backward reasoning records."
    ),
) -> None:
    load_project_env()
    from datasets import load_dataset

    token = os.getenv(hf_token_env)
    reasoner = DualReasoner(teacher_model, hf_token=token)

    load_kwargs = {}
    if dataset_config:
        load_kwargs["name"] = dataset_config

    ds: Any = load_dataset(dataset_name, split=split, **load_kwargs)
    if limit:
        ds = ds.select(range(limit))

    output_dir.mkdir(parents=True, exist_ok=True)
    forward_path = output_dir / forward_filename
    backward_path = output_dir / backward_filename

    with forward_path.open("w", encoding="utf-8") as fwd_f, backward_path.open(
        "w", encoding="utf-8"
    ) as bwd_f:
        for idx, sample in enumerate(track(ds, description="Generating")):
            question = sample.get("question") or sample.get("input")
            answer = sample.get("answer", "")
            outputs = reasoner.generate(question)

            forward_record = {
                "id": sample.get("id", idx),
                "question": question,
                "answer": answer,
                "rationale": outputs.forward_trace,
                "verdict": True,
            }
            backward_record = {
                "id": sample.get("id", idx),
                "question": question,
                "answer": answer,
                "rationale": outputs.backward_trace,
                "verdict": outputs.verification.strip().endswith("PASS"),
            }
            fwd_f.write(json.dumps(forward_record) + "\n")
            bwd_f.write(json.dumps(backward_record) + "\n")

    console.print(f"[green]Saved augmented reasoning pairs to {output_dir}[/green]")


if __name__ == "__main__":
    app()
