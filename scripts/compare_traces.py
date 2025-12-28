"""Compare per-example forward/backward traces between two evaluation configs.

Usage examples:
  # Dry-run (no model loads): write questions + gold answers only
  python scripts/compare_traces.py configs/eval_baseline.yaml configs/eval_gsm8k.yaml \
      --max-samples 100 --output outputs/evals/gsm8k_samples_compare.csv --dry-run

  # Full run (will load models and generate traces) - ensure you have GPU/token
  python scripts/compare_traces.py configs/eval_baseline.yaml configs/eval_gsm8k.yaml \
      --max-samples 100 --output outputs/evals/gsm8k_samples_compare.csv

The script writes a CSV with columns:
  idx, question, gold_answer,
  a_forward, a_backward, a_verification,
  b_forward, b_backward, b_verification

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import csv
from rich.console import Console

from reasoning_lab.config import load_config

app = typer.Typer(add_completion=False)
console = Console()


def _load_samples_from_config(cfg_path: Path, max_samples: int) -> List[Dict[str, Any]]:
    cfg = load_config(cfg_path).raw
    from datasets import load_dataset

    dataset_cfg = cfg.datasets[0]
    ds = load_dataset(dataset_cfg.path, dataset_cfg.get("subset"))
    split = ds[dataset_cfg.split]
    if dataset_cfg.get("few_shot_k"):
        split = split.select(range(dataset_cfg.few_shot_k))
    max_available = len(split)
    n = min(max_samples, max_available)
    console.log(f"Loading {n} examples from {dataset_cfg.path} ({dataset_cfg.split})")
    rows = []
    for i, ex in enumerate(split.select(range(n))):
        q = ex.get("question") or ex.get("input") or ex.get("prompt") or ""
        a = ex.get("answer") or ex.get("output") or ""
        rows.append({"idx": i, "question": q, "gold_answer": a})
    return rows


@app.command()
def main(
    config_a: Path = typer.Argument(..., help="Path to eval config A (baseline)."),
    config_b: Path = typer.Argument(..., help="Path to eval config B (hybrid)."),
    max_samples: int = typer.Option(100, help="Maximum examples to compare."),
    output: Path = typer.Option(Path("outputs/evals/trace_compare.csv"), help="Output CSV path."),
    hf_token_env: str = typer.Option("HF_TOKEN", help="Env var with HF token."),
    dry_run: bool = typer.Option(False, help="If set, do not load models or run generations."),
) -> None:
    load_cfg_a = load_config(config_a).raw
    load_cfg_b = load_config(config_b).raw

    samples = _load_samples_from_config(config_a, max_samples)

    output.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "idx",
        "question",
        "gold_answer",
        "a_forward",
        "a_backward",
        "a_verification",
        "b_forward",
        "b_backward",
        "b_verification",
    ]

    # If dry-run, write placeholders only
    if dry_run:
        console.log("Dry-run mode: writing questions and gold answers only.")
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in samples:
                writer.writerow({
                    "idx": row["idx"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "a_forward": "",
                    "a_backward": "",
                    "a_verification": "",
                    "b_forward": "",
                    "b_backward": "",
                    "b_verification": "",
                })
        console.log(f"Wrote dry-run CSV to {output}")
        raise typer.Exit()

    # Otherwise, run generation for both configs
    token = None
    import os

    if hf_token_env in os.environ:
        token = os.environ[hf_token_env]

    from reasoning_lab.inference.dual_reasoner import DualReasoner

    console.log("Instantiating reasoner A (config A)...")
    model_a = load_cfg_a.model.get("base_model") or load_cfg_a.model.get("checkpoint_path")
    adapter_a = load_cfg_a.model.get("adapter_path")
    reasoner_a = DualReasoner(model_a, adapter_path=adapter_a, hf_token=token)

    console.log("Instantiating reasoner B (config B)...")
    model_b = load_cfg_b.model.get("base_model") or load_cfg_b.model.get("checkpoint_path")
    adapter_b = load_cfg_b.model.get("adapter_path")
    reasoner_b = DualReasoner(model_b, adapter_path=adapter_b, hf_token=token)

    rows_out: List[Dict[str, Optional[str]]] = []
    for row in samples:
        q = row["question"]
        console.log(f"Generating idx={row['idx']}")
        out_a = reasoner_a.generate(q)
        out_b = reasoner_b.generate(q)
        rows_out.append(
            {
                "idx": row["idx"],
                "question": q,
                "gold_answer": row["gold_answer"],
                "a_forward": out_a.forward_trace,
                "a_backward": out_a.backward_trace,
                "a_verification": out_a.verification,
                "b_forward": out_b.forward_trace,
                "b_backward": out_b.backward_trace,
                "b_verification": out_b.verification,
            }
        )

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    console.log(f"Wrote comparison CSV to {output}")


if __name__ == "__main__":
    app()
