"""Dataset utilities for combining forward and backward reasoning traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass
class JsonlRecord:
    uid: str
    question: str
    answer: str
    rationale: str
    verdict: bool
    score: float | None = None


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
            )


def join_forward_backward(
    forward_path: str | Path,
    backward_path: str | Path,
    *,
    forward_fail_fallback: Optional[str] = None,
    backward_fail_fallback: Optional[str] = None,
    forward_weight: float = 1.0,
    backward_weight: float = 1.0,
) -> List[ReasoningExample]:
    """Join forward/backward JSONL files on the shared ``id`` field."""

    forward_records = {rec.uid: rec for rec in _load_jsonl(forward_path)}
    backward_records = {rec.uid: rec for rec in _load_jsonl(backward_path)}

    examples: List[ReasoningExample] = []
    for uid, fwd in forward_records.items():
        bwd = backward_records.get(uid)
        if bwd is None:
            continue

        forward_rejected = forward_fail_fallback or f"The earlier reasoning was flawed for question: {fwd.question}"
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
                forward_weight=forward_weight if fwd.verdict else forward_weight * 0.5,
                backward_weight=backward_weight if bwd.verdict else backward_weight * 0.5,
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
