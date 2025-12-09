#!/usr/bin/env python3
"""
Visualization script for comparing reasoning model experiments.

Generates comparison charts for:
- Baseline vs Fine-tuned models
- Forward-only vs Backward-only vs Hybrid training
- Accuracy, Self-consistency, and Acknowledgement rate metrics

Usage:
    python scripts/visualize_results.py --results-dir outputs/evals --output-dir outputs/figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# Define color palette for consistency
COLORS = {
    "baseline": "#808080",      # Gray
    "forward_only": "#2196F3",  # Blue
    "backward_only": "#FF9800", # Orange
    "hybrid": "#4CAF50",        # Green
}

LABELS = {
    "baseline": "Baseline (No DPO)",
    "forward_only": "Forward-Only DPO",
    "backward_only": "Backward-Only DPO",
    "hybrid": "Hybrid DPO",
}


def load_results(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load evaluation results from JSON files."""
    results = {}
    
    # Expected result files
    expected_files = {
        "baseline": ["eval_gsm8k_baseline.json", "eval_baseline.json", "baseline_results.json"],
        "forward_only": ["eval_gsm8k_forward_only.json", "eval_forward_only.json", "forward_only_results.json"],
        "backward_only": ["eval_gsm8k_backward_only.json", "eval_backward_only.json", "backward_only_results.json"],
        "hybrid": ["eval_gsm8k_hybrid.json", "eval_hybrid.json", "hybrid_results.json", "eval_results.json"],
    }
    
    for condition, possible_files in expected_files.items():
        for filename in possible_files:
            filepath = results_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    data = json.load(f)
                # Handle different result formats
                if "gsm8k" in data:
                    results[condition] = data["gsm8k"]
                elif "metrics" in data:
                    results[condition] = data["metrics"]
                else:
                    results[condition] = data
                print(f"Loaded {condition} results from {filepath}")
                break
    
    return results


def create_bar_chart(
    results: Dict[str, Dict[str, float]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> None:
    """Create a bar chart comparing conditions for a single metric."""
    conditions = []
    values = []
    colors = []
    
    # Order: baseline first, then alphabetical
    order = ["baseline", "forward_only", "backward_only", "hybrid"]
    
    for cond in order:
        if cond in results and metric in results[cond]:
            conditions.append(LABELS.get(cond, cond))
            values.append(results[cond][metric])
            colors.append(COLORS.get(cond, "#666666"))
    
    if not conditions:
        print(f"No data for metric: {metric}")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.2%}" if val <= 1 else f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right", fontsize=11)
    ax.set_ylim(0, max(values) * 1.15)
    
    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_grouped_bar_chart(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    title: str,
    output_path: Path,
    figsize: tuple = (12, 6),
) -> None:
    """Create a grouped bar chart comparing all metrics across conditions."""
    order = ["baseline", "forward_only", "backward_only", "hybrid"]
    conditions = [c for c in order if c in results]
    
    if not conditions:
        print("No results to plot")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(metrics))
    width = 0.8 / len(conditions)
    
    for i, cond in enumerate(conditions):
        values = [results[cond].get(m, 0) for m in metrics]
        offset = (i - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=LABELS.get(cond, cond),
            color=COLORS.get(cond, "#666666"),
            edgecolor="black",
            linewidth=0.8,
        )
    
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 1.1)
    
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_improvement_chart(
    results: Dict[str, Dict[str, float]],
    metric: str,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> None:
    """Create a chart showing improvement over baseline."""
    if "baseline" not in results or metric not in results["baseline"]:
        print(f"No baseline data for {metric}")
        return
    
    baseline_val = results["baseline"][metric]
    
    conditions = []
    improvements = []
    colors = []
    
    for cond in ["forward_only", "backward_only", "hybrid"]:
        if cond in results and metric in results[cond]:
            conditions.append(LABELS.get(cond, cond))
            improvement = results[cond][metric] - baseline_val
            improvements.append(improvement)
            colors.append(COLORS.get(cond, "#666666"))
    
    if not conditions:
        print(f"No data to compare for {metric}")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, improvements, color=colors, edgecolor="black", linewidth=1.2)
    
    # Color bars based on positive/negative
    for bar, imp in zip(bars, improvements):
        if imp < 0:
            bar.set_color("#f44336")  # Red for negative
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.annotate(
            f"{val:+.1%}" if abs(val) <= 1 else f"{val:+.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -15),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=12,
            fontweight="bold",
        )
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_ylabel(f"Improvement in {metric.replace('_', ' ').title()}", fontsize=12)
    ax.set_title(f"Improvement Over Baseline ({metric.replace('_', ' ').title()})", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right", fontsize=11)
    
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(results: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Generate a markdown summary table."""
    metrics = ["accuracy", "self_consistency", "acknowledgement_rate"]
    order = ["baseline", "forward_only", "backward_only", "hybrid"]
    conditions = [c for c in order if c in results]
    
    lines = ["# Experiment Results Summary\n"]
    lines.append("| Condition | " + " | ".join(m.replace("_", " ").title() for m in metrics) + " |")
    lines.append("|" + "|".join(["---"] * (len(metrics) + 1)) + "|")
    
    for cond in conditions:
        row = [LABELS.get(cond, cond)]
        for m in metrics:
            val = results[cond].get(m, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.2%}" if val <= 1 else f"{val:.2f}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")
    
    # Add improvement section
    if "baseline" in results:
        lines.append("\n## Improvement Over Baseline\n")
        lines.append("| Condition | " + " | ".join(m.replace("_", " ").title() for m in metrics) + " |")
        lines.append("|" + "|".join(["---"] * (len(metrics) + 1)) + "|")
        
        baseline = results["baseline"]
        for cond in conditions:
            if cond == "baseline":
                continue
            row = [LABELS.get(cond, cond)]
            for m in metrics:
                if m in results[cond] and m in baseline:
                    diff = results[cond][m] - baseline[m]
                    row.append(f"{diff:+.2%}" if abs(diff) <= 1 else f"{diff:+.2f}")
                else:
                    row.append("N/A")
            lines.append("| " + " | ".join(row) + " |")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize reasoning experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/evals"),
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory to save figures",
    )
    args = parser.parse_args()
    
    # Also check outputs root for eval_results.json
    results_dirs = [args.results_dir, Path("outputs")]
    
    all_results = {}
    for rdir in results_dirs:
        if rdir.exists():
            all_results.update(load_results(rdir))
    
    if not all_results:
        print(f"No results found in {args.results_dir} or outputs/")
        print("\nRun evaluations first:")
        print("  python scripts/eval_reasoning.py configs/eval_baseline.yaml")
        print("  python scripts/eval_reasoning.py configs/eval_forward_only.yaml")
        print("  python scripts/eval_reasoning.py configs/eval_backward_only.yaml")
        print("  python scripts/eval_reasoning.py configs/eval_hybrid.yaml")
        return
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nFound results for: {list(all_results.keys())}")
    
    # Generate individual metric charts
    metrics = ["accuracy", "self_consistency", "acknowledgement_rate"]
    for metric in metrics:
        create_bar_chart(
            all_results,
            metric,
            f"GSM8K {metric.replace('_', ' ').title()} by Training Method",
            metric.replace("_", " ").title(),
            args.output_dir / f"{metric}_comparison.png",
        )
    
    # Generate grouped bar chart
    create_grouped_bar_chart(
        all_results,
        metrics,
        "GSM8K Performance Comparison",
        args.output_dir / "all_metrics_comparison.png",
    )
    
    # Generate improvement charts
    for metric in metrics:
        create_improvement_chart(
            all_results,
            metric,
            args.output_dir / f"{metric}_improvement.png",
        )
    
    # Generate summary table
    generate_summary_table(all_results, args.output_dir / "results_summary.md")
    
    print(f"\n✅ All figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
