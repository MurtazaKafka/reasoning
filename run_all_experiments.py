#!/usr/bin/env python3
"""
Complete Experiment Pipeline for Forward-Backward Reasoning with DPO

This script runs the entire experimental pipeline:
1. Data generation with rejection sampling
2. Training all model variants (forward-only, backward-only, hybrid)
3. Evaluation on all benchmarks
4. Results analysis and report generation

All outputs are saved persistently to disk. Safe to run on clusters where
browser sessions may disconnect.

Usage:
    python run_all_experiments.py --output-dir ./experiment_outputs

For SLURM:
    sbatch run_experiments.slurm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)

    # File handler - captures everything
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_file}")
    return logger


def run_command(
    cmd: List[str],
    logger: logging.Logger,
    cwd: Optional[Path] = None,
    timeout: int = 43200,  # 12 hour default timeout
) -> bool:
    """Run a command with streaming output. Returns True on success."""
    cmd_str = " ".join(cmd)
    logger.info(f"Running: {cmd_str}")

    try:
        # Use Popen for streaming output instead of buffering
        process = subprocess.Popen(
            cmd,
            cwd=cwd or SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
        )

        start_time = time.time()
        output_lines = []

        # Stream output in real-time
        assert process.stdout is not None  # Guaranteed by stdout=PIPE
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                process.kill()
                logger.error(f"Command timed out after {timeout/3600:.1f} hours")
                return False

            # Read line with timeout
            line = process.stdout.readline()
            if line:
                line = line.rstrip()
                output_lines.append(line)
                # Log output at INFO level so it's visible
                logger.info(f"  {line}")
            elif process.poll() is not None:
                # Process finished
                break

        # Get any remaining output
        remaining = process.stdout.read()
        if remaining:
            for line in remaining.strip().split("\n"):
                if line:
                    logger.info(f"  {line}")

        if process.returncode != 0:
            logger.error(f"Command failed with code {process.returncode}")
            return False

        logger.info("  ✓ Success")
        return True

    except Exception as e:
        logger.error(f"Command failed: {e}")
        return False


def save_checkpoint(checkpoint_file: Path, stage: str, status: Dict[str, Any]) -> None:
    """Save checkpoint to resume from if interrupted."""
    checkpoint = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "status": status,
    }
    checkpoint_file.write_text(json.dumps(checkpoint, indent=2))


def load_checkpoint(checkpoint_file: Path) -> Optional[Dict[str, Any]]:
    """Load checkpoint if exists."""
    if checkpoint_file.exists():
        return json.loads(checkpoint_file.read_text())
    return None


def stage_1_generate_data(
    output_dir: Path,
    logger: logging.Logger,
    data_limit: int = 2000,
    skip_if_exists: bool = True,
) -> bool:
    """Stage 1: Generate training data with rejection sampling."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Generation")
    logger.info("=" * 60)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    dpo_pairs_file = data_dir / "dpo_pairs.jsonl"
    if skip_if_exists and dpo_pairs_file.exists():
        logger.info(f"Data already exists at {dpo_pairs_file}, skipping generation")
        return True

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "scripts" / "generate_dpo_pairs.py"),
        "--limit", str(data_limit),
        "--output-dir", str(data_dir),
        "--rejection-sampling",
        "--max-rejection-attempts", "5",
    ]

    success = run_command(cmd, logger)

    if success:
        # Also run legacy format for compatibility
        legacy_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "scripts" / "bootstrap_pairs.py"),
            "--limit", str(data_limit),
            "--output-dir", str(data_dir),
        ]
        run_command(legacy_cmd, logger)

    return success


def stage_2_train_models(
    output_dir: Path,
    logger: logging.Logger,
    configs: List[str],
    skip_if_exists: bool = True,
) -> Dict[str, bool]:
    """Stage 2: Train all model variants."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Model Training")
    logger.info("=" * 60)

    results = {}
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables
    env = os.environ.copy()
    env["OUTPUT_DIR"] = str(output_dir)

    for config_name in configs:
        config_path = SCRIPT_DIR / "configs" / "experiments" / f"train_{config_name}.yaml"

        if not config_path.exists():
            # Try without train_ prefix
            config_path = SCRIPT_DIR / "configs" / f"dpo_{config_name}.yaml"

        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}")
            results[config_name] = False
            continue

        # Check if already trained
        checkpoint_dir = runs_dir / f"{config_name}_dpo"
        if skip_if_exists and checkpoint_dir.exists():
            # Check for adapter files
            adapter_files = list(checkpoint_dir.glob("**/adapter_model.*"))
            if adapter_files:
                logger.info(f"Model {config_name} already trained, skipping")
                results[config_name] = True
                continue

        logger.info(f"Training: {config_name}")

        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "scripts" / "train_dpo.py"),
            str(config_path),
        ]

        # Run with modified environment using streaming
        try:
            process = subprocess.Popen(
                cmd,
                cwd=SCRIPT_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            start_time_train = time.time()
            assert process.stdout is not None

            while True:
                elapsed = time.time() - start_time_train
                if elapsed > 43200:  # 12 hour timeout for training
                    process.kill()
                    logger.error(f"Training timed out after 12 hours")
                    success = False
                    break

                line = process.stdout.readline()
                if line:
                    logger.info(f"  {line.rstrip()}")
                elif process.poll() is not None:
                    break

            if process.returncode is not None:
                success = process.returncode == 0
            else:
                success = False

        except Exception as e:
            logger.error(f"Training failed: {e}")
            success = False

        results[config_name] = success
        logger.info(f"  {'✓ Success' if success else '✗ Failed'}")

        # Save intermediate checkpoint
        save_checkpoint(
            output_dir / "checkpoint.json",
            "training",
            {"completed": [k for k, v in results.items() if v]}
        )

    return results


def stage_3_evaluate_models(
    output_dir: Path,
    logger: logging.Logger,
    model_configs: List[Dict[str, Any]],
    max_samples: int = 500,
) -> Dict[str, bool]:
    """Stage 3: Evaluate all trained models using optimized batch evaluation."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Model Evaluation (Optimized Batch)")
    logger.info("=" * 60)

    evals_dir = output_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"

    # Build model list for the optimized eval script
    model_names = [cfg["name"] for cfg in model_configs]

    # Check which models are already evaluated
    models_to_eval = []
    results = {}
    for cfg in model_configs:
        name = cfg["name"]
        result_file = evals_dir / f"{name}.json"
        if result_file.exists():
            logger.info(f"  {name}: Results exist, skipping")
            results[name] = True
        else:
            models_to_eval.append(name)

    if not models_to_eval:
        logger.info("All models already evaluated")
        return results

    logger.info(f"Models to evaluate: {models_to_eval}")

    # Use the optimized batch evaluation script
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "scripts" / "eval_all_models.py"),
        "--output-dir", str(evals_dir),
        "--runs-dir", str(runs_dir),
        "--max-samples", str(max_samples),
        "--models", *models_to_eval,
        "--resume",  # Always enable resume for robustness
    ]

    success = run_command(cmd, logger, timeout=43200)  # 12 hour timeout

    # Check which models were successfully evaluated
    for name in models_to_eval:
        result_file = evals_dir / f"{name}.json"
        results[name] = result_file.exists()
        if results[name]:
            logger.info(f"  {name}: ✓ Evaluated")
        else:
            logger.error(f"  {name}: ✗ Failed")

    return results


def stage_4_analyze_results(output_dir: Path, logger: logging.Logger) -> bool:
    """Stage 4: Analyze results and generate reports."""
    logger.info("=" * 60)
    logger.info("STAGE 4: Results Analysis")
    logger.info("=" * 60)

    evals_dir = output_dir / "evals"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "scripts" / "analyze_results.py"),
        str(evals_dir),
        "--output-latex", str(reports_dir / "main_results.tex"),
        "--output-stats", str(reports_dir / "statistics.json"),
    ]

    success = run_command(cmd, logger)

    # Also create a summary file
    if success:
        create_summary_report(output_dir, logger)

    return success


def create_summary_report(output_dir: Path, logger: logging.Logger) -> None:
    """Create a human-readable summary report."""
    reports_dir = output_dir / "reports"
    summary_file = reports_dir / "EXPERIMENT_SUMMARY.txt"

    lines = [
        "=" * 70,
        "FORWARD-BACKWARD REASONING WITH DPO - EXPERIMENT SUMMARY",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Load statistics if available
    stats_file = reports_dir / "statistics.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
        lines.append("KEY RESULTS:")
        lines.append("-" * 40)

        if "gsm8k_improvement" in stats:
            lines.append(f"GSM8K Relative Improvement: {stats['gsm8k_improvement']:.1%}")
        if "gsm8k_absolute_improvement" in stats:
            lines.append(f"GSM8K Absolute Improvement: {stats['gsm8k_absolute_improvement']:.1%}")
        if "ack_rate_improvement" in stats:
            lines.append(f"Acknowledgement Rate Improvement: {stats['ack_rate_improvement']:.1%}")
        lines.append("")

    # List all result files
    evals_dir = output_dir / "evals"
    if evals_dir.exists():
        lines.append("EVALUATION RESULTS:")
        lines.append("-" * 40)
        for result_file in sorted(evals_dir.glob("*.json")):
            if not result_file.name.endswith(".predictions.json"):
                try:
                    results = json.loads(result_file.read_text())
                    model_name = result_file.stem

                    # Extract key metrics
                    if "results" in results:
                        results = results["results"]

                    if "gsm8k" in results:
                        gsm8k = results["gsm8k"]
                        acc = gsm8k.get("accuracy", {})
                        if isinstance(acc, dict):
                            acc = acc.get("value", "N/A")
                        ack = gsm8k.get("acknowledgement_rate", {})
                        if isinstance(ack, dict):
                            ack = ack.get("value", "N/A")

                        lines.append(f"  {model_name}:")
                        lines.append(f"    Accuracy: {acc:.1%}" if isinstance(acc, float) else f"    Accuracy: {acc}")
                        lines.append(f"    Ack Rate: {ack:.1%}" if isinstance(ack, float) else f"    Ack Rate: {ack}")
                except Exception as e:
                    lines.append(f"  {result_file.stem}: Error reading results")
        lines.append("")

    # List checkpoint info
    lines.append("OUTPUT LOCATIONS:")
    lines.append("-" * 40)
    lines.append(f"  Logs: {output_dir / 'logs'}")
    lines.append(f"  Trained Models: {output_dir / 'runs'}")
    lines.append(f"  Evaluation Results: {output_dir / 'evals'}")
    lines.append(f"  Reports: {output_dir / 'reports'}")
    lines.append("")
    lines.append("=" * 70)

    summary_file.write_text("\n".join(lines))
    logger.info(f"Summary report saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete forward-backward reasoning experiments"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./experiment_outputs"),
        help="Directory for all outputs (default: ./experiment_outputs)"
    )
    parser.add_argument(
        "--data-limit",
        type=int,
        default=2000,
        help="Number of training samples to generate (default: 2000)"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=500,
        help="Number of evaluation samples per benchmark (default: 500)"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data generation if already exists"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only run evaluation"
    )
    parser.add_argument(
        "--only-eval",
        action="store_true",
        help="Only run evaluation (assumes models are trained)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["forward_only", "backward_only", "hybrid_60_40"],
        help="Training configs to run (default: forward_only backward_only hybrid_60_40)"
    )

    args = parser.parse_args()

    # Setup
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("FORWARD-BACKWARD REASONING EXPERIMENT PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training configs: {args.configs}")
    logger.info(f"Data limit: {args.data_limit}")
    logger.info(f"Eval samples: {args.eval_samples}")

    # Check for checkpoint
    checkpoint_file = output_dir / "checkpoint.json"
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_file)
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint['stage']}")

    start_time = time.time()
    results = {
        "data_generation": None,
        "training": {},
        "evaluation": {},
        "analysis": None,
    }

    try:
        # Stage 1: Data Generation
        if not args.only_eval and not args.skip_training:
            results["data_generation"] = stage_1_generate_data(
                output_dir,
                logger,
                data_limit=args.data_limit,
                skip_if_exists=args.skip_data or args.resume,
            )
            save_checkpoint(checkpoint_file, "data_generation", results)

        # Stage 2: Training
        if not args.only_eval and not args.skip_training:
            results["training"] = stage_2_train_models(
                output_dir,
                logger,
                configs=args.configs,
                skip_if_exists=args.resume,
            )
            save_checkpoint(checkpoint_file, "training", results)

        # Stage 3: Evaluation
        # Build model configs for evaluation
        model_configs = [
            {"name": "baseline", "adapter_path": None},
        ]

        runs_dir = output_dir / "runs"
        for config_name in args.configs:
            adapter_path = runs_dir / f"{config_name}_dpo"
            if adapter_path.exists() or not args.skip_training:
                model_configs.append({
                    "name": config_name,
                    "adapter_path": str(adapter_path) if adapter_path.exists() else None,
                })

        results["evaluation"] = stage_3_evaluate_models(
            output_dir,
            logger,
            model_configs,
            max_samples=args.eval_samples,
        )
        save_checkpoint(checkpoint_file, "evaluation", results)

        # Stage 4: Analysis
        results["analysis"] = stage_4_analyze_results(output_dir, logger)
        save_checkpoint(checkpoint_file, "complete", results)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        save_checkpoint(checkpoint_file, "interrupted", results)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        save_checkpoint(checkpoint_file, "failed", results)
        raise

    # Final summary
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("")
    logger.info("Key output files:")
    logger.info(f"  - Summary: {output_dir / 'reports' / 'EXPERIMENT_SUMMARY.txt'}")
    logger.info(f"  - LaTeX table: {output_dir / 'reports' / 'main_results.tex'}")
    logger.info(f"  - Evaluation JSONs: {output_dir / 'evals' / '*.json'}")

    # Save final results
    final_results_file = output_dir / "final_results.json"
    final_results_file.write_text(json.dumps({
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "results": results,
    }, indent=2))

    logger.info(f"  - Final results: {final_results_file}")

    return 0 if all([
        results.get("data_generation", True),
        all(results.get("training", {}).values()) if results.get("training") else True,
        all(results.get("evaluation", {}).values()) if results.get("evaluation") else True,
    ]) else 1


if __name__ == "__main__":
    sys.exit(main())
