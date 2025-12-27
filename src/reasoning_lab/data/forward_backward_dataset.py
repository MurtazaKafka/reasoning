"""Dataset utilities for combining forward and backward reasoning traces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from datasets import Dataset


@dataclass
class ReasoningExample:
    """Normalized schema for forward/backward reasoning tuples."""

    uid: str
    question: str
    answer: str
    forward_chosen: str
    forward_rejected: str
    backward_chosen: str
    backward_rejected: str
    forward_weight: float = 1.0
    backward_weight: float = 1.0
    # Additional metadata for analysis
    forward_correct: bool = True
    backward_calibrated: bool = True
    predicted_answer: str = ""


@dataclass
class JsonlRecord:
    uid: str
    question: str
    answer: str
    rationale: str
    verdict: bool
    score: float | None = None
    predicted_answer: str = ""
    verification_result: str = ""
    forward_was_correct: bool | None = None


def _load_jsonl(path: str | Path) -> Iterator[JsonlRecord]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line)
            yield JsonlRecord(
                uid=str(data["id"]),
                question=data.get("question", ""),
                answer=data.get("answer", ""),
                rationale=data.get("rationale") or data.get("reasoning", ""),
                verdict=bool(data.get("verdict", data.get("label", True))),
                score=data.get("score"),
                predicted_answer=data.get("predicted_answer", ""),
                verification_result=data.get("verification_result", ""),
                forward_was_correct=data.get("forward_was_correct"),
            )


def join_forward_backward(
    forward_path: str | Path,
    backward_path: str | Path,
    *,
    forward_fail_fallback: Optional[str] = None,
    backward_fail_fallback: Optional[str] = None,
    forward_weight: float = 1.0,
    backward_weight: float = 1.0,
    use_score_weighting: bool = True,
    filter_incorrect_forward: bool = False,
    filter_miscalibrated_backward: bool = False,
) -> List[ReasoningExample]:
    """Join forward/backward JSONL files on the shared ``id`` field.

    Parameters
    ----------
    forward_path : Path
        Path to forward reasoning JSONL file.
    backward_path : Path
        Path to backward reasoning JSONL file.
    forward_fail_fallback : str, optional
        Text to use as rejected sample for forward reasoning.
    backward_fail_fallback : str, optional
        Text to use as rejected sample for backward verification.
    forward_weight : float
        Base weight for forward reasoning samples.
    backward_weight : float
        Base weight for backward verification samples.
    use_score_weighting : bool
        If True, use the score field from JSONL to modulate weights.
    filter_incorrect_forward : bool
        If True, only include samples where forward reasoning was correct.
    filter_miscalibrated_backward : bool
        If True, only include samples where backward verification was calibrated.

    Returns
    -------
    List[ReasoningExample]
        Joined examples ready for DPO training.
    """

    forward_records = {rec.uid: rec for rec in _load_jsonl(forward_path)}
    backward_records = {rec.uid: rec for rec in _load_jsonl(backward_path)}

    examples: List[ReasoningExample] = []
    stats = {"total": 0, "filtered_forward": 0, "filtered_backward": 0}

    for uid, fwd in forward_records.items():
        bwd = backward_records.get(uid)
        if bwd is None:
            continue

        stats["total"] += 1

        # Optional filtering
        if filter_incorrect_forward and not fwd.verdict:
            stats["filtered_forward"] += 1
            continue
        if filter_miscalibrated_backward and not bwd.verdict:
            stats["filtered_backward"] += 1
            continue

        # Compute weights
        if use_score_weighting and fwd.score is not None:
            fwd_weight = forward_weight * fwd.score
        else:
            fwd_weight = forward_weight if fwd.verdict else forward_weight * 0.5

        if use_score_weighting and bwd.score is not None:
            bwd_weight = backward_weight * bwd.score
        else:
            bwd_weight = backward_weight if bwd.verdict else backward_weight * 0.5

        # Create rejected samples
        # For forward: a generic error message (could be improved with actual wrong traces)
        forward_rejected = forward_fail_fallback or (
            f"I apologize, but I made an error in my reasoning. "
            f"Let me reconsider the problem: {fwd.question[:100]}..."
        )

        # For backward: opposite of what the model said
        if "pass" in bwd.verification_result.lower():
            backward_rejected = "Verification: FAIL\nThe solution contains errors that need correction."
        else:
            backward_rejected = backward_fail_fallback or "Verification: FAIL"

        examples.append(
            ReasoningExample(
                uid=uid,
                question=fwd.question,
                answer=fwd.answer,
                forward_chosen=fwd.rationale,
                forward_rejected=forward_rejected,
                backward_chosen=bwd.rationale,
                backward_rejected=backward_rejected,
                forward_weight=fwd_weight,
                backward_weight=bwd_weight,
                forward_correct=fwd.verdict,
                backward_calibrated=bwd.verdict,
                predicted_answer=fwd.predicted_answer,
            )
        )

    return examples


def to_dpo_dataset(
    examples: Iterable[ReasoningExample],
    *,
    include_backward: bool = True,
) -> Dataset:
    """Convert joined examples into a Hugging Face dataset suitable for DPO training."""

    prompts: list[str] = []
    chosens: list[str] = []
    rejecteds: list[str] = []
    weights: list[float] = []
    task_types: list[str] = []

    for ex in examples:
        forward_prompt = (
            "You are a forward reasoner. Given the problem, produce a clear chain of thought and final"
            " answer. Problem: "
            f"{ex.question}\n"
        )
        forward_output = f"{ex.forward_chosen}\nFinal Answer: {ex.answer}"

        prompts.append(forward_prompt)
        chosens.append(forward_output)
        rejecteds.append(ex.forward_rejected)
        weights.append(ex.forward_weight)
        task_types.append("forward")

        if include_backward:
            backward_prompt = (
                "You are a backward verifier. Starting from the candidate answer, reason backwards to"
                " check validity. Respond with Verification: PASS/FAIL. Problem: "
                f"{ex.question}\nCandidate Answer: {ex.answer}"
            )
            prompts.append(backward_prompt)
            chosens.append(ex.backward_chosen)
            rejecteds.append(ex.backward_rejected)
            weights.append(ex.backward_weight)
            task_types.append("backward")

    data = {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
        "weight": weights,
        "task_type": task_types,
    }

    return Dataset.from_dict(data)


def load_dpo_pairs_jsonl(
    path: str | Path,
    *,
    forward_weight: float = 0.6,
    backward_weight: float = 0.4,
    min_quality_score: float = 0.0,
) -> Dataset:
    """Load DPO pairs from the new JSONL format (from generate_dpo_pairs.py).

    This format contains both chosen and rejected traces in a single record,
    with real negative examples from rejection sampling.

    Args:
        path: Path to the dpo_pairs.jsonl file.
        forward_weight: Weight for forward reasoning samples.
        backward_weight: Weight for backward verification samples.
        min_quality_score: Minimum quality score to include a sample.

    Returns:
        HuggingFace Dataset ready for DPO training.
    """
    prompts: list[str] = []
    chosens: list[str] = []
    rejecteds: list[str] = []
    weights: list[float] = []
    task_types: list[str] = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            # Skip low quality samples
            quality = record.get("quality_score", 1.0)
            if quality < min_quality_score:
                continue

            question = record["question"]
            ground_truth = record.get("ground_truth", "")

            # Forward reasoning pair
            forward_prompt = (
                f"You are a forward reasoner. Solve the problem step by step.\n"
                f"Problem: {question}\n"
            )
            prompts.append(forward_prompt)
            chosens.append(record.get("chosen_forward", ""))
            rejecteds.append(record.get("rejected_forward", ""))

            # Weight by quality and whether we have real negatives
            has_real_neg = record.get("has_real_negative", False)
            weight = forward_weight * quality
            if has_real_neg:
                weight *= 1.2  # Boost weight for real negative examples
            weights.append(weight)
            task_types.append("forward")

            # Backward verification pair
            if record.get("chosen_backward"):
                chosen_answer = record.get("chosen_answer", ground_truth)
                backward_prompt = (
                    f"You are a verifier. Check if the answer is correct.\n"
                    f"Problem: {question}\n"
                    f"Candidate Answer: {chosen_answer}\n"
                )
                prompts.append(backward_prompt)
                chosens.append(record.get("chosen_backward", ""))
                rejecteds.append(record.get("rejected_backward", "Verification: FAIL"))
                weights.append(backward_weight * quality)
                task_types.append("backward")

    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
        "weight": weights,
        "task_type": task_types,
    })
