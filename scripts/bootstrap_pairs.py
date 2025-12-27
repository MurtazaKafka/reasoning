"""Generate forward/backward reasoning pairs using a teacher model."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.progress import track

from reasoning_lab.env import load_project_env
from reasoning_lab.inference.dual_reasoner import DualReasoner
from reasoning_lab.utils.text import (
    extract_final_answer,
    extract_ground_truth_answer,
    answers_match,
    extract_verification,
)

app = typer.Typer(add_completion=False, help=__doc__)
console: Any = Console()


def _is_degenerate_output(text: str, max_repetitions: int = 5) -> bool:
    """Check if output shows degenerate repetition patterns.

    Detects oscillating PASS/FAIL, repeated phrases, etc.
    """
    # Check for oscillating PASS/FAIL pattern
    pass_fail_pattern = r"(PASS|FAIL)"
    matches = re.findall(pass_fail_pattern, text, re.IGNORECASE)
    if len(matches) > max_repetitions:
        return True

    # Check for repeated sentences (sign of generation loop)
    sentences = re.split(r'[.!?]\s+', text)
    if len(sentences) > 3:
        sentence_counts = {}
        for s in sentences:
            s_clean = s.strip().lower()
            if len(s_clean) > 20:  # Only check substantial sentences
                sentence_counts[s_clean] = sentence_counts.get(s_clean, 0) + 1
                if sentence_counts[s_clean] > 2:
                    return True

    return False


def _compute_forward_verdict(predicted: str, ground_truth: str) -> bool:
    """Compute whether forward reasoning produced correct answer."""
    return answers_match(predicted, ground_truth)


def _compute_backward_verdict(backward_trace: str, forward_correct: bool) -> bool:
    """Compute whether backward verification is calibrated.

    A good backward trace should:
    - Say PASS when forward answer is correct
    - Say FAIL when forward answer is incorrect
    """
    verification = extract_verification(backward_trace)
    backward_says_pass = "pass" in verification.lower()

    # Verdict is True if backward agrees with ground truth
    if forward_correct and backward_says_pass:
        return True  # Correctly verified correct answer
    elif not forward_correct and not backward_says_pass:
        return True  # Correctly identified incorrect answer
    else:
        return False  # Miscalibrated verification


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
    filter_degenerate: bool = typer.Option(
        True, help="Filter out degenerate outputs (repeated text, oscillating PASS/FAIL)."
    ),
    min_trace_length: int = typer.Option(
        50, help="Minimum character length for valid reasoning traces."
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

    # Statistics for logging
    stats = {
        "total": 0,
        "forward_correct": 0,
        "backward_calibrated": 0,
        "filtered_degenerate": 0,
        "filtered_short": 0,
    }

    with forward_path.open("w", encoding="utf-8") as fwd_f, backward_path.open(
        "w", encoding="utf-8"
    ) as bwd_f:
        for idx, sample in enumerate(track(ds, description="Generating")):
            question = sample.get("question") or sample.get("input")
            answer = sample.get("answer", "")
            ground_truth = extract_ground_truth_answer(answer)

            outputs = reasoner.generate(question)
            stats["total"] += 1

            # Check for degenerate outputs
            if filter_degenerate:
                if _is_degenerate_output(outputs.forward_trace):
                    stats["filtered_degenerate"] += 1
                    console.print(f"[yellow]Skipping {idx}: degenerate forward trace[/yellow]")
                    continue
                if _is_degenerate_output(outputs.backward_trace):
                    stats["filtered_degenerate"] += 1
                    console.print(f"[yellow]Skipping {idx}: degenerate backward trace[/yellow]")
                    continue

            # Check minimum length
            if len(outputs.forward_trace) < min_trace_length:
                stats["filtered_short"] += 1
                continue
            if len(outputs.backward_trace) < min_trace_length:
                stats["filtered_short"] += 1
                continue

            # Extract predicted answer and compute verdicts
            predicted_answer = extract_final_answer(outputs.forward_trace)
            forward_correct = _compute_forward_verdict(predicted_answer, ground_truth)
            backward_calibrated = _compute_backward_verdict(outputs.backward_trace, forward_correct)

            if forward_correct:
                stats["forward_correct"] += 1
            if backward_calibrated:
                stats["backward_calibrated"] += 1

            forward_record = {
                "id": sample.get("id", idx),
                "question": question,
                "answer": answer,
                "predicted_answer": predicted_answer,
                "rationale": outputs.forward_trace,
                "verdict": forward_correct,  # True only if answer matches ground truth
                "score": 1.0 if forward_correct else 0.5,  # Confidence score for weighting
            }
            backward_record = {
                "id": sample.get("id", idx),
                "question": question,
                "answer": answer,
                "rationale": outputs.backward_trace,
                "verification_result": outputs.verification,
                "forward_was_correct": forward_correct,
                "verdict": backward_calibrated,  # True if verification matches reality
                "score": 1.0 if backward_calibrated else 0.5,
            }
            fwd_f.write(json.dumps(forward_record) + "\n")
            bwd_f.write(json.dumps(backward_record) + "\n")

    # Print summary statistics
    valid_samples = stats["total"] - stats["filtered_degenerate"] - stats["filtered_short"]
    console.print(f"\n[bold]Generation Statistics:[/bold]")
    console.print(f"  Total samples processed: {stats['total']}")
    console.print(f"  Valid samples written: {valid_samples}")
    console.print(f"  Filtered (degenerate): {stats['filtered_degenerate']}")
    console.print(f"  Filtered (too short): {stats['filtered_short']}")
    if valid_samples > 0:
        console.print(f"  Forward accuracy: {stats['forward_correct'] / valid_samples:.1%}")
        console.print(f"  Backward calibration: {stats['backward_calibrated'] / valid_samples:.1%}")
    console.print(f"\n[green]Saved augmented reasoning pairs to {output_dir}[/green]")


if __name__ == "__main__":
    app()
