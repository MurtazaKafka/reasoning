"""Metrics for evaluating forward/backward reasoning models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

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
    details: Dict[str, float] | None = None


def compute_accuracy(samples: Iterable[Dict[str, str]]) -> MetricResult:
    """Compute accuracy with robust numeric answer matching.

    Handles various answer formats:
    - GSM8K format (#### X)
    - LaTeX boxed (\\boxed{X})
    - Natural language ("The answer is X")
    - Fractions
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
    """Compute rate at which model acknowledges incorrect answers via backward verification.

    This is the key metric for measuring hallucination awareness:
    - Among incorrect forward predictions, what % did backward reasoning flag as FAIL?

    Returns:
        MetricResult with acknowledgement_rate value and details breakdown.
    """
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
    return MetricResult(
        name="acknowledgement_rate",
        value=rate,
        details={
            "total_incorrect": float(total_wrong),
            "acknowledged_as_fail": float(acknowledged),
        },
    )


def compute_false_positive_rate(samples: Iterable[Dict[str, str]]) -> MetricResult:
    """Compute rate at which model incorrectly flags correct answers as FAIL.

    This measures over-conservatism in verification:
    - Among correct forward predictions, what % did backward reasoning flag as FAIL?

    A good model should have low false positive rate (don't reject correct answers).
    """
    total_correct = 0
    false_positives = 0

    for sample in samples:
        prediction = extract_final_answer(sample["forward_trace"])
        ground_truth = extract_ground_truth_answer(sample["answer"])
        verification = extract_verification(sample["backward_trace"])

        # Only consider correct predictions
        if not answers_match(prediction, ground_truth):
            continue

        total_correct += 1
        # Check if backward pass incorrectly flagged as FAIL
        if "fail" in verification.lower():
            false_positives += 1

    rate = false_positives / max(total_correct, 1)
    return MetricResult(
        name="false_positive_rate",
        value=rate,
        details={
            "total_correct": float(total_correct),
            "incorrectly_flagged_fail": float(false_positives),
        },
    )


def compute_verification_calibration(samples: Iterable[Dict[str, str]]) -> MetricResult:
    """Compute overall calibration of backward verification.

    Measures how well the backward verification correlates with actual correctness.
    Returns F1-like score where:
    - True Positive: Incorrect answer correctly flagged as FAIL
    - True Negative: Correct answer correctly passed as PASS
    - False Positive: Correct answer incorrectly flagged as FAIL
    - False Negative: Incorrect answer incorrectly passed as PASS
    """
    tp = 0  # Incorrect answer, correctly says FAIL
    tn = 0  # Correct answer, correctly says PASS
    fp = 0  # Correct answer, incorrectly says FAIL
    fn = 0  # Incorrect answer, incorrectly says PASS

    for sample in samples:
        prediction = extract_final_answer(sample["forward_trace"])
        ground_truth = extract_ground_truth_answer(sample["answer"])
        verification = extract_verification(sample["backward_trace"])

        is_correct = answers_match(prediction, ground_truth)
        says_fail = "fail" in verification.lower()

        if is_correct and not says_fail:
            tn += 1
        elif is_correct and says_fail:
            fp += 1
        elif not is_correct and says_fail:
            tp += 1
        else:  # not is_correct and not says_fail
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)

    # Precision: Of those flagged FAIL, how many were actually wrong?
    precision = tp / max(tp + fp, 1)

    # Recall: Of those actually wrong, how many were flagged FAIL?
    recall = tp / max(tp + fn, 1)

    # F1 Score
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return MetricResult(
        name="verification_calibration",
        value=f1,
        details={
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": float(tp),
            "true_negatives": float(tn),
            "false_positives": float(fp),
            "false_negatives": float(fn),
        },
    )


def compute_self_consistency(
    samples: Iterable[Dict[str, str]],
    *,
    group_by_question: bool = True,
) -> MetricResult:
    """Compute self-consistency based on unique extracted answers.

    If group_by_question is True (default), groups samples by question and
    measures consistency within each group. This is the proper way to measure
    self-consistency when you have multiple samples per question.

    If group_by_question is False, treats all samples as one group
    (original behavior, useful when each sample is a unique question).

    Returns:
        MetricResult with consistency score (1.0 = all same, lower = more diverse)
    """
    samples_list = list(samples)

    if not samples_list:
        return MetricResult(name="self_consistency", value=1.0)

    if group_by_question:
        # Group samples by question
        groups: Dict[str, List[str]] = defaultdict(list)
        for sample in samples_list:
            question = sample.get("question", sample.get("forward_trace", ""))[:200]
            answer = extract_final_answer(sample["forward_trace"])
            groups[question].append(answer)

        # Compute consistency per group
        consistencies = []
        for question, answers in groups.items():
            if len(answers) > 1:
                unique = len(set(answers))
                # Consistency = 1 - diversity
                # diversity = (unique - 1) / (total - 1)
                consistency = 1 - (unique - 1) / (len(answers) - 1)
                consistencies.append(consistency)
            else:
                # Single sample = perfect consistency
                consistencies.append(1.0)

        avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 1.0

        return MetricResult(
            name="self_consistency",
            value=avg_consistency,
            details={
                "num_groups": float(len(groups)),
                "avg_samples_per_group": float(len(samples_list) / max(len(groups), 1)),
            },
        )
    else:
        # Original behavior: all samples in one group
        answers = [extract_final_answer(sample["forward_trace"]) for sample in samples_list]
        unique = len(set(answers))
        total = len(answers)
        # Return ratio of unique answers (1.0 = all different, lower = more consistent)
        diversity = unique / max(total, 1)
        return MetricResult(
            name="self_consistency",
            value=1 - diversity,  # Convert to consistency (higher = better)
            details={"unique_answers": float(unique), "total_answers": float(total)},
        )


def compute_all_metrics(samples: Iterable[Dict[str, str]]) -> Dict[str, MetricResult]:
    """Compute all reasoning metrics on the given samples.

    Returns a dictionary of metric name -> MetricResult.
    """
    samples_list = list(samples)  # Materialize for multiple iterations

    return {
        "accuracy": compute_accuracy(samples_list),
        "acknowledgement_rate": compute_acknowledgement_rate(samples_list),
        "false_positive_rate": compute_false_positive_rate(samples_list),
        "verification_calibration": compute_verification_calibration(samples_list),
        "self_consistency": compute_self_consistency(samples_list),
    }
