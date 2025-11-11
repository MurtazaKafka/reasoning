"""Text parsing helpers for reasoning traces."""

from __future__ import annotations


def extract_final_answer(text: str) -> str:
    for line in reversed(text.splitlines()):
        if "Final Answer:" in line:
            return line.split("Final Answer:", 1)[1].strip()
    return text.splitlines()[-1].strip()


def extract_verification(text: str) -> str:
    for line in reversed(text.splitlines()):
        lower = line.strip().lower()
        if lower.startswith("verification"):
            return line.strip()
    return "Verification: UNKNOWN"
