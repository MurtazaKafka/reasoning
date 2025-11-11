"""Metrics for evaluating forward/backward reasoning models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from reasoning_lab.utils.text import extract_final_answer, extract_verification


@dataclass
class MetricResult:
    name: str
    value: float


def compute_accuracy(samples: Iterable[Dict[str, str]]) -> MetricResult:
    total = 0
    correct = 0
    for sample in samples:
        total += 1
        prediction = extract_final_answer(sample["forward_trace"])
        if prediction.strip() == sample["answer"].strip():
            correct += 1
    return MetricResult(name="accuracy", value=correct / max(total, 1))


def compute_acknowledgement_rate(samples: Iterable[Dict[str, str]]) -> MetricResult:
    total_wrong = 0
    acknowledged = 0
    for sample in samples:
        prediction = extract_final_answer(sample["forward_trace"])
        verification = extract_verification(sample["backward_trace"])
        if prediction.strip() == sample["answer"].strip():
            continue
        total_wrong += 1
        if verification.lower().endswith("fail"):
            acknowledged += 1
    rate = acknowledged / max(total_wrong, 1)
    return MetricResult(name="acknowledgement_rate", value=rate)


def compute_self_consistency(samples: Iterable[Dict[str, str]]) -> MetricResult:
    # Placeholder: use average number of distinct rationales.
    traces = [sample["forward_trace"] for sample in samples]
    unique = len(set(traces))
    total = len(traces)
    return MetricResult(name="self_consistency", value=unique / max(total, 1))
