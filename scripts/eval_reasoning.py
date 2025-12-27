"""Evaluate forward/backward reasoning model on benchmarks.

Features:
- Multi-sample generation for self-consistency
- Detailed per-sample predictions saved for analysis
- Comprehensive metrics including verification calibration
- Support for multiple models in batch evaluation
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from reasoning_lab.config import load_config
from reasoning_lab.env import load_project_env
from reasoning_lab.inference.dual_reasoner import DualReasoner
from reasoning_lab.metrics.reasoning_metrics import (
    MetricResult,
    compute_acknowledgement_rate,
    compute_accuracy,
    compute_self_consistency,
    compute_false_positive_rate,
    compute_verification_calibration,
    compute_all_metrics,
)
from reasoning_lab.utils.text import extract_final_answer, extract_ground_truth_answer, answers_match

app = typer.Typer(add_completion=False, help=__doc__)
console: Any = Console()


_METRIC_REGISTRY: Dict[str, Callable[[Iterable[Dict[str, str]]], MetricResult]] = {
    "accuracy": compute_accuracy,
    "acknowledgement_rate": compute_acknowledgement_rate,
    "self_consistency": compute_self_consistency,
    "false_positive_rate": compute_false_positive_rate,
    "verification_calibration": compute_verification_calibration,
}


@app.command()
def main(
    config_path: Path = typer.Argument(..., help="Evaluation config path."),
    hf_token_env: str = typer.Option("HF_TOKEN", help="Env var with HF token."),
    output_json: Optional[Path] = typer.Option(
        None, help="File to save metrics. Defaults to outputs/evals/<experiment_name>.json"
    ),
    save_predictions: bool = typer.Option(
        True, help="Save individual predictions for error analysis."
    ),
    num_samples: int = typer.Option(
        1, help="Number of samples per question for self-consistency."
    ),
    greedy: bool = typer.Option(
        False, help="Use greedy decoding (temperature=0) for reproducibility."
    ),
) -> None:
    """Run evaluation on configured benchmarks."""
    load_project_env()
    from datasets import load_dataset

    cfg: Any = load_config(config_path).raw

    # Set output path based on experiment name if not specified
    if output_json is None:
        exp_name = cfg.experiment.get("name", "eval_results")
        output_json = Path(f"outputs/evals/{exp_name}.json")

    token = None
    if hf_token_env in os.environ:
        token = os.environ[hf_token_env]

    # Support both old (checkpoint_path) and new (base_model + adapter_path) config styles
    model_path = cfg.model.get("base_model") or cfg.model.get("checkpoint_path")
    adapter_path = cfg.model.get("adapter_path")

    # Override sampling if greedy mode
    forward_sampling = None
    backward_sampling = None
    if greedy:
        forward_sampling = {"temperature": 0.0, "do_sample": False}
        backward_sampling = {"temperature": 0.0, "do_sample": False}

    reasoner = DualReasoner(
        model_path,
        adapter_path=adapter_path,
        hf_token=token,
        torch_dtype=cfg.model.get("dtype", "bf16"),
        load_in_8bit=cfg.model.get("load_in_8bit", False),
        load_in_4bit=cfg.model.get("load_in_4bit", False),
        forward_sampling=forward_sampling,
        backward_sampling=backward_sampling,
    )

    all_results: Dict[str, List[MetricResult]] = {}
    all_predictions: Dict[str, List[Dict[str, Any]]] = {}

    for dataset_cfg in cfg.datasets:
        console.log(f"\n[bold]Evaluating {dataset_cfg.name}[/bold] ({dataset_cfg.split})...")

        try:
            ds = load_dataset(dataset_cfg.path, dataset_cfg.get("subset"))
            split: Any = ds[dataset_cfg.split]
        except Exception as e:
            console.log(f"[red]Failed to load {dataset_cfg.name}: {e}[/red]")
            continue

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
        predictions = []

        for idx, example in enumerate(track(split, description=f"{dataset_cfg.name}")):
            question = example.get("question") or example.get("input")
            answer = example.get("answer", "")
            ground_truth = extract_ground_truth_answer(answer)

            # Generate one or more samples
            if num_samples > 1:
                multi_output = reasoner.generate_multiple(question, num_samples=num_samples)
                # Use majority vote answer
                predicted = multi_output.majority_answer
                consistency = multi_output.consistency_score

                # Use first sample for backward trace
                first_sample = multi_output.samples[0] if multi_output.samples else None
                forward_trace = first_sample.forward_trace if first_sample else ""
                backward_trace = first_sample.backward_trace if first_sample else ""
                verification = first_sample.verification if first_sample else ""
            else:
                outputs = reasoner.generate(question)
                predicted = outputs.extracted_answer
                forward_trace = outputs.forward_trace
                backward_trace = outputs.backward_trace
                verification = outputs.verification
                consistency = 1.0

            # Check correctness
            is_correct = answers_match(predicted, ground_truth)

            sample_data = {
                "forward_trace": forward_trace,
                "backward_trace": backward_trace,
                "verification": verification,
                "answer": answer,
                "question": question,
            }
            samples.append(sample_data)

            # Detailed prediction for analysis
            if save_predictions:
                predictions.append({
                    "idx": idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "verification": verification,
                    "consistency": consistency,
                    "forward_trace": forward_trace[:500],  # Truncate for storage
                    "backward_trace": backward_trace[:300],
                })

        # Compute metrics
        metric_results = []
        for metric_cfg in cfg.metrics:
            metric_name = metric_cfg["name"] if isinstance(metric_cfg, dict) else metric_cfg
            if metric_name in _METRIC_REGISTRY:
                metric_fn = _METRIC_REGISTRY[metric_name]
                result = metric_fn(samples)
                metric_results.append(result)

        all_results[dataset_cfg.name] = metric_results
        all_predictions[dataset_cfg.name] = predictions
        _print_table(dataset_cfg.name, metric_results)

    # Save results
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Serialize metrics
    serialized = {
        "metadata": {
            "model": model_path,
            "adapter": adapter_path,
            "timestamp": datetime.now().isoformat(),
            "num_samples": num_samples,
            "greedy": greedy,
        },
        "results": {},
    }

    for task, metrics in all_results.items():
        serialized["results"][task] = {}
        for metric in metrics:
            serialized["results"][task][metric.name] = {
                "value": metric.value,
                "details": metric.details if metric.details else {},
            }

    output_json.write_text(json.dumps(serialized, indent=2))
    console.print(f"\n[green]Saved metrics to {output_json}[/green]")

    # Save predictions for error analysis
    if save_predictions:
        predictions_path = output_json.with_suffix(".predictions.json")
        predictions_path.write_text(json.dumps(all_predictions, indent=2))
        console.print(f"[green]Saved predictions to {predictions_path}[/green]")

    # Print summary
    _print_summary(all_results)


def _print_table(task_name: str, metrics: List[MetricResult]) -> None:
    """Print metrics table for a single task."""
    table = Table(title=f"Results: {task_name}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Details", justify="left")

    for metric in metrics:
        details_str = ""
        if metric.details:
            key_details = {k: v for k, v in metric.details.items() if not k.startswith("_")}
            if key_details:
                details_str = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in list(key_details.items())[:3])
        table.add_row(metric.name, f"{metric.value:.4f}", details_str)

    console.print(table)


def _print_summary(all_results: Dict[str, List[MetricResult]]) -> None:
    """Print overall summary across all tasks."""
    console.print("\n[bold]Summary[/bold]")

    # Compute averages across tasks
    metric_sums: Dict[str, List[float]] = {}
    for task, metrics in all_results.items():
        for metric in metrics:
            if metric.name not in metric_sums:
                metric_sums[metric.name] = []
            metric_sums[metric.name].append(metric.value)

    for metric_name, values in metric_sums.items():
        avg = sum(values) / len(values)
        console.print(f"  Average {metric_name}: {avg:.4f}")


@app.command("batch")
def batch_eval(
    config_path: Path = typer.Argument(..., help="Batch evaluation config."),
    output_dir: Path = typer.Option(
        Path("outputs/evals"), help="Output directory for all results."
    ),
) -> None:
    """Evaluate multiple models from a batch config."""
    load_project_env()
    from datasets import load_dataset

    cfg: Any = load_config(config_path).raw

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if config has multiple models
    if not hasattr(cfg, "models"):
        console.print("[red]Batch config must have 'models' section[/red]")
        raise typer.Exit(1)

    for model_cfg in cfg.models:
        model_name = model_cfg.get("name", "unknown")
        console.print(f"\n[bold blue]{'='*50}[/bold blue]")
        console.print(f"[bold]Evaluating: {model_name}[/bold]")
        console.print(f"[bold blue]{'='*50}[/bold blue]")

        try:
            # Create reasoner for this model
            reasoner = DualReasoner(
                model_cfg.get("base_model"),
                adapter_path=model_cfg.get("adapter_path"),
                torch_dtype=cfg.inference.get("dtype", "bf16"),
                load_in_8bit=cfg.inference.get("load_in_8bit", False),
            )

            all_results: Dict[str, Dict[str, Any]] = {}

            for dataset_cfg in cfg.datasets:
                console.log(f"  Evaluating {dataset_cfg.name}...")

                try:
                    ds = load_dataset(dataset_cfg.path, dataset_cfg.get("subset"))
                    split = ds[dataset_cfg.split]
                except Exception as e:
                    console.log(f"  [red]Failed: {e}[/red]")
                    continue

                if cfg.experiment.get("max_samples_per_task"):
                    max_samples = int(cfg.experiment.max_samples_per_task)
                    split = split.select(range(min(len(split), max_samples)))

                samples = []
                for example in track(split, description=f"  {dataset_cfg.name}"):
                    question = example.get("question") or example.get("input")
                    answer = example.get("answer", "")
                    outputs = reasoner.generate(question)
                    samples.append({
                        "forward_trace": outputs.forward_trace,
                        "backward_trace": outputs.backward_trace,
                        "verification": outputs.verification,
                        "answer": answer,
                    })

                # Compute all metrics
                results = compute_all_metrics(samples)
                all_results[dataset_cfg.name] = {
                    name: {"value": r.value, "details": r.details or {}}
                    for name, r in results.items()
                }

            # Save results for this model
            output_path = output_dir / f"{model_name}.json"
            output_path.write_text(json.dumps(all_results, indent=2))
            console.print(f"  [green]Saved to {output_path}[/green]")

        except Exception as e:
            console.print(f"  [red]Error evaluating {model_name}: {e}[/red]")
            continue

    console.print(f"\n[green]Batch evaluation complete. Results in {output_dir}[/green]")


if __name__ == "__main__":
    app()
