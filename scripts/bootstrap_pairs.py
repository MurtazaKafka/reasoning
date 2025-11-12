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
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Teacher model for generating forward/backward traces.",
    ),
    torch_dtype: str = typer.Option(
        "auto",
        help="Torch dtype for the teacher model (auto, bf16, fp16, float32).",
    ),
    device_map: str = typer.Option("auto", help="Device placement strategy (auto, cpu, cuda:0, …)."),
    load_in_8bit: bool = typer.Option(
        False,
        help="Load the teacher in 8-bit (requires bitsandbytes on Linux).",
    ),
    load_in_4bit: bool = typer.Option(
        False,
        help="Load the teacher in 4-bit (requires bitsandbytes on Linux).",
    ),
    attn_implementation: Optional[str] = typer.Option(
        None,
        help="Override attention implementation (e.g., sdpa, flash_attention_2).",
    ),
    max_forward_tokens: int = typer.Option(
        384,
        help="Maximum tokens to generate for the forward reasoning trace.",
    ),
    max_backward_tokens: int = typer.Option(
        192,
        help="Maximum tokens to generate for the backward reasoning trace.",
    ),
    forward_temperature: float = typer.Option(0.7, help="Sampling temperature for forward reasoning."),
    forward_top_p: float = typer.Option(0.95, help="Nucleus sampling top-p for forward reasoning."),
    backward_temperature: float = typer.Option(0.3, help="Sampling temperature for backward reasoning."),
    backward_top_p: float = typer.Option(0.9, help="Nucleus sampling top-p for backward reasoning."),
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
    reasoner = DualReasoner(
        teacher_model,
        hf_token=token,
        torch_dtype=torch_dtype,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        attn_implementation=attn_implementation,
        max_forward_tokens=max_forward_tokens,
        max_backward_tokens=max_backward_tokens,
        forward_sampling={
            "temperature": forward_temperature,
            "top_p": forward_top_p,
            "do_sample": forward_temperature > 0,
        },
        backward_sampling={
            "temperature": backward_temperature,
            "top_p": backward_top_p,
            "do_sample": backward_temperature > 0,
        },
    )

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
