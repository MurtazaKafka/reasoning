"""Evaluate forward/backward reasoning model on benchmarks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import typer
from rich.console import Console
from rich.table import Table

from reasoning_lab.config import load_config
from reasoning_lab.env import load_project_env
from reasoning_lab.inference.dual_reasoner import DualReasoner
from reasoning_lab.metrics.reasoning_metrics import (
    MetricResult,
    compute_acknowledgement_rate,
    compute_accuracy,
    compute_self_consistency,
)

app = typer.Typer(add_completion=False, help=__doc__)
console: Any = Console()


_METRIC_REGISTRY: Dict[str, Callable[[Iterable[Dict[str, str]]], MetricResult]] = {
    "accuracy": compute_accuracy,
    "acknowledgement_rate": compute_acknowledgement_rate,
    "self_consistency": compute_self_consistency,
}


@app.command()
def main(
    config_path: Path = typer.Argument(..., help="Evaluation config path."),
    hf_token_env: str = typer.Option("HF_TOKEN", help="Env var with HF token."),
    output_json: Path = typer.Option(Path("outputs/eval_results.json"), help="File to save metrics."),
) -> None:
    load_project_env()
    from datasets import load_dataset

    cfg: Any = load_config(config_path).raw
    token = None
    if hf_token_env in os.environ:
        token = os.environ[hf_token_env]

    reasoner = DualReasoner(
        cfg.model.checkpoint_path,
        hf_token=token,
        torch_dtype=cfg.model.get("dtype", "bf16"),
    )

    all_results: Dict[str, List[MetricResult]] = {}

    for dataset_cfg in cfg.datasets:
        console.log(f"Evaluating {dataset_cfg.name} ({dataset_cfg.split})...")
        ds = load_dataset(dataset_cfg.path, dataset_cfg.get("subset"))
        split: Any = ds[dataset_cfg.split]
        if dataset_cfg.get("few_shot_k"):
            split = split.select(range(dataset_cfg.few_shot_k))
        if cfg.experiment.get("max_samples_per_task"):
            max_samples = int(cfg.experiment.max_samples_per_task)
            try:
                available = len(split)
            except TypeError:
                available = max_samples
            split = split.select(range(min(available, max_samples)))

        samples = []
        for example in console.track(split, description=f"{dataset_cfg.name} inference"):
            question = example.get("question") or example.get("input")
            answer = example.get("answer", "")
            outputs = reasoner.generate(question)
            samples.append(
                {
                    "forward_trace": outputs.forward_trace,
                    "backward_trace": outputs.backward_trace,
                    "verification": outputs.verification,
                    "answer": answer,
                }
            )

        metric_results = []
        for metric_cfg in cfg.metrics:
            metric_fn = _METRIC_REGISTRY[metric_cfg["name"]]
            result = metric_fn(samples)
            metric_results.append(result)

        all_results[dataset_cfg.name] = metric_results
        _print_table(dataset_cfg.name, metric_results)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    serialized = {
        task: {metric.name: metric.value for metric in metrics}
        for task, metrics in all_results.items()
    }
    output_json.write_text(json.dumps(serialized, indent=2))
    console.print(f"[green]Saved metrics to {output_json}[/green]")


def _print_table(task_name: str, metrics: List[MetricResult]) -> None:
    table = Table(title=f"Results: {task_name}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for metric in metrics:
        table.add_row(metric.name, f"{metric.value:.4f}")
    console.print(table)


if __name__ == "__main__":
    app()
