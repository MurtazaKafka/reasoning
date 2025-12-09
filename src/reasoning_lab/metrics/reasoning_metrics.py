"""Metrics for evaluating forward/backward reasoning models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from reasoning_lab.utils.text import (
    answers_match,
    extract_final_answer,
    extract_ground_truth_answer,
    extract_verification,
)


@dataclass
class MetricResult:
    name: str
    value: float


def compute_accuracy(samples: Iterable[Dict[str, str]]) -> MetricResult:
    """Compute accuracy with robust numeric answer matching.
    
    Handles various answer formats:
    - GSM8K format (#### X)
    - LaTeX boxed (\\boxed{X})
    - Natural language ("The answer is X")
    """
    total = 0
    correct = 0
    for sample in samples:
        total += 1
        prediction = extract_final_answer(sample["forward_trace"])
        ground_truth = extract_ground_truth_answer(sample["answer"])
        if answers_match(prediction, ground_truth):
            correct += 1
    return MetricResult(name="accuracy", value=correct / max(total, 1))


def compute_acknowledgement_rate(samples: Iterable[Dict[str, str]]) -> MetricResult:
    """Compute rate at which model acknowledges incorrect answers via backward verification."""
    total_wrong = 0
    acknowledged = 0
    for sample in samples:
        prediction = extract_final_answer(sample["forward_trace"])
        ground_truth = extract_ground_truth_answer(sample["answer"])
        verification = extract_verification(sample["backward_trace"])
        
        # Skip correct predictions
        if answers_match(prediction, ground_truth):
            continue
        
        total_wrong += 1
        # Check if backward pass flagged the answer as FAIL
        if "fail" in verification.lower():
            acknowledged += 1
    
    rate = acknowledged / max(total_wrong, 1)
    return MetricResult(name="acknowledgement_rate", value=rate)


def compute_self_consistency(samples: Iterable[Dict[str, str]]) -> MetricResult:
    """Compute self-consistency based on unique extracted answers.
    
    Higher diversity (lower consistency) may indicate uncertainty.
    """
    answers = [extract_final_answer(sample["forward_trace"]) for sample in samples]
    unique = len(set(answers))
    total = len(answers)
    # Return ratio of unique answers (1.0 = all different, lower = more consistent)
    return MetricResult(name="self_consistency", value=unique / max(total, 1))
