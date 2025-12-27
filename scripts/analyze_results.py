"""Analyze and visualize experimental results for paper figures.

Usage:
    python scripts/analyze_results.py outputs/evals/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


def load_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all JSON result files from directory."""
    results = {}
    for json_file in results_dir.glob("*.json"):
        model_name = json_file.stem
        with json_file.open() as f:
            results[model_name] = json.load(f)
    return results


def create_main_results_table(results: Dict[str, Dict[str, Any]]) -> Table:
    """Create main results table for paper."""
    table = Table(title="Main Results: Forward-Backward Reasoning with DPO")

    # Add columns
    table.add_column("Model", style="bold")
    table.add_column("GSM8K Acc", justify="right")
    table.add_column("MATH Acc", justify="right")
    table.add_column("Ack. Rate", justify="right")
    table.add_column("FP Rate", justify="right")
    table.add_column("Calib. F1", justify="right")

    # Model order for presentation
    model_order = [
        "baseline",
        "forward_only",
        "backward_only",
        "hybrid_60_40",
        "hybrid_50_50",
        "hybrid_80_20",
    ]

    model_names = {
        "baseline": "LLaMA 3.1 8B (Base)",
        "forward_only": "Forward-Only DPO",
        "backward_only": "Backward-Only DPO",
        "hybrid_60_40": "Hybrid 60/40 (Ours)",
        "hybrid_50_50": "Hybrid 50/50",
        "hybrid_80_20": "Hybrid 80/20",
    }

    for model_key in model_order:
        if model_key not in results:
            continue

        model_results = results[model_key]
        name = model_names.get(model_key, model_key)

        # Extract metrics (handle both old and new format)
        def get_metric(task: str, metric: str) -> str:
            if task not in model_results:
                return "-"
            task_results = model_results[task]
            if metric not in task_results:
                return "-"
            val = task_results[metric]
            if isinstance(val, dict):
                val = val.get("value", 0)
            return f"{val:.1%}" if val else "-"

        gsm8k_acc = get_metric("gsm8k", "accuracy")
        math_acc = get_metric("math_algebra", "accuracy")
        ack_rate = get_metric("gsm8k", "acknowledgement_rate")
        fp_rate = get_metric("gsm8k", "false_positive_rate")
        calib_f1 = get_metric("gsm8k", "verification_calibration")

        # Highlight best hybrid
        style = "green" if model_key == "hybrid_60_40" else None
        table.add_row(name, gsm8k_acc, math_acc, ack_rate, fp_rate, calib_f1, style=style)

    return table


def create_ablation_table(results: Dict[str, Dict[str, Any]]) -> Table:
    """Create ablation study table."""
    table = Table(title="Ablation: Effect of Forward/Backward Weight Ratio")

    table.add_column("Weight Ratio (F/B)", style="bold")
    table.add_column("GSM8K Accuracy", justify="right")
    table.add_column("Acknowledgement Rate", justify="right")
    table.add_column("Verification F1", justify="right")

    ratios = [
        ("100/0", "forward_only"),
        ("80/20", "hybrid_80_20"),
        ("60/40", "hybrid_60_40"),
        ("50/50", "hybrid_50_50"),
        ("0/100", "backward_only"),
    ]

    for ratio_name, model_key in ratios:
        if model_key not in results:
            continue

        model_results = results[model_key]

        def get_val(metric: str) -> str:
            if "gsm8k" not in model_results:
                return "-"
            val = model_results["gsm8k"].get(metric, {})
            if isinstance(val, dict):
                val = val.get("value", 0)
            return f"{val:.1%}" if val else "-"

        table.add_row(
            ratio_name,
            get_val("accuracy"),
            get_val("acknowledgement_rate"),
            get_val("verification_calibration"),
        )

    return table


def generate_latex_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Generate LaTeX table for paper."""
    latex = r"""
\begin{table}[t]
\centering
\caption{Main Results: Forward-Backward Reasoning with DPO on LLaMA 3.1 8B}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{GSM8K} & \textbf{MATH} & \textbf{Ack. Rate} & \textbf{FP Rate} & \textbf{Calib. F1} \\
\midrule
"""

    model_order = [
        ("baseline", "LLaMA 3.1 8B (Base)"),
        ("forward_only", "Forward-Only DPO"),
        ("backward_only", "Backward-Only DPO"),
        ("hybrid_60_40", "\\textbf{Hybrid 60/40 (Ours)}"),
    ]

    for model_key, name in model_order:
        if model_key not in results:
            continue

        model_results = results[model_key]

        def fmt(task: str, metric: str) -> str:
            if task not in model_results:
                return "-"
            val = model_results[task].get(metric, {})
            if isinstance(val, dict):
                val = val.get("value", 0)
            return f"{val*100:.1f}" if val else "-"

        latex += f"{name} & {fmt('gsm8k', 'accuracy')} & {fmt('math_algebra', 'accuracy')} & "
        latex += f"{fmt('gsm8k', 'acknowledgement_rate')} & {fmt('gsm8k', 'false_positive_rate')} & "
        latex += f"{fmt('gsm8k', 'verification_calibration')} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def compute_statistics(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute key statistics for the paper."""
    stats = {}

    # Improvement of hybrid over forward-only
    if "forward_only" in results and "hybrid_60_40" in results:
        fwd = results["forward_only"]
        hyb = results["hybrid_60_40"]

        if "gsm8k" in fwd and "gsm8k" in hyb:
            fwd_acc = fwd["gsm8k"].get("accuracy", {})
            hyb_acc = hyb["gsm8k"].get("accuracy", {})
            if isinstance(fwd_acc, dict):
                fwd_acc = fwd_acc.get("value", 0)
            if isinstance(hyb_acc, dict):
                hyb_acc = hyb_acc.get("value", 0)

            if fwd_acc and hyb_acc:
                stats["gsm8k_improvement"] = (hyb_acc - fwd_acc) / fwd_acc
                stats["gsm8k_absolute_improvement"] = hyb_acc - fwd_acc

            fwd_ack = fwd["gsm8k"].get("acknowledgement_rate", {})
            hyb_ack = hyb["gsm8k"].get("acknowledgement_rate", {})
            if isinstance(fwd_ack, dict):
                fwd_ack = fwd_ack.get("value", 0)
            if isinstance(hyb_ack, dict):
                hyb_ack = hyb_ack.get("value", 0)

            if fwd_ack is not None and hyb_ack is not None:
                stats["ack_rate_improvement"] = hyb_ack - fwd_ack

    return stats


@app.command()
def main(
    results_dir: Path = typer.Argument(..., help="Directory containing result JSON files"),
    output_latex: Path = typer.Option(None, help="Output LaTeX table file"),
    output_stats: Path = typer.Option(None, help="Output statistics JSON file"),
) -> None:
    """Analyze experimental results and generate tables."""

    if not results_dir.exists():
        console.print(f"[red]Results directory not found: {results_dir}[/red]")
        raise typer.Exit(1)

    results = load_results(results_dir)

    if not results:
        console.print(f"[yellow]No result files found in {results_dir}[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Found {len(results)} result files[/bold]\n")

    # Print main results table
    main_table = create_main_results_table(results)
    console.print(main_table)
    console.print()

    # Print ablation table
    ablation_table = create_ablation_table(results)
    console.print(ablation_table)
    console.print()

    # Compute and print key statistics
    stats = compute_statistics(results)
    if stats:
        console.print("[bold]Key Statistics for Paper:[/bold]")
        if "gsm8k_improvement" in stats:
            console.print(f"  GSM8K relative improvement: {stats['gsm8k_improvement']:.1%}")
        if "gsm8k_absolute_improvement" in stats:
            console.print(f"  GSM8K absolute improvement: {stats['gsm8k_absolute_improvement']:.1%}")
        if "ack_rate_improvement" in stats:
            console.print(f"  Acknowledgement rate improvement: {stats['ack_rate_improvement']:.1%}")

    # Save LaTeX if requested
    if output_latex:
        latex = generate_latex_table(results)
        output_latex.write_text(latex)
        console.print(f"\n[green]LaTeX table saved to {output_latex}[/green]")

    # Save stats if requested
    if output_stats:
        output_stats.write_text(json.dumps(stats, indent=2))
        console.print(f"[green]Statistics saved to {output_stats}[/green]")


if __name__ == "__main__":
    app()
