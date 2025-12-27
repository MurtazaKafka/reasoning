"""Text parsing helpers for reasoning traces."""

from __future__ import annotations

import re


def _eval_fraction(frac_str: str) -> float | None:
    """Safely evaluate a fraction string like '1/2' or '3/4'."""
    try:
        if "/" in frac_str:
            parts = frac_str.split("/")
            if len(parts) == 2:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    return num / denom
        return float(frac_str)
    except (ValueError, ZeroDivisionError):
        return None


def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from text, handling multiple formats.

    Supports:
    - GSM8K format: #### 42
    - LaTeX boxed: \\boxed{42} or $\\boxed{42}$
    - LaTeX frac: \\frac{1}{2}
    - "The answer is X" / "The final answer is X"
    - "Final Answer: X"
    - Fractions: 1/2, 3/4
    - Percentages: 50%
    - Last number in text as fallback
    """
    if not text:
        return None

    # Priority 1: GSM8K format "#### <number>"
    gsm8k_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if gsm8k_match:
        return gsm8k_match.group(1).replace(",", "")

    # Priority 2: LaTeX \frac{a}{b} anywhere in text (check before boxed to handle nested)
    frac_latex_global = re.search(r"\\frac\{(-?\d+)\}\{(\d+)\}", text)
    if frac_latex_global:
        result = _eval_fraction(f"{frac_latex_global.group(1)}/{frac_latex_global.group(2)}")
        if result is not None:
            if result == int(result):
                return str(int(result))
            return str(result)

    # Priority 3: LaTeX \boxed{<number or expression>}
    # Use a more permissive regex that handles nested braces
    boxed_match = re.search(r"\\boxed\{(.+?)\}(?:\s*\$)?", text)
    if boxed_match:
        boxed_content = boxed_match.group(1).strip()
        # Check for plain fraction like 3/4 in boxed
        if "/" in boxed_content:
            frac_match = re.search(r"(-?\d+)\s*/\s*(\d+)", boxed_content)
            if frac_match:
                result = _eval_fraction(f"{frac_match.group(1)}/{frac_match.group(2)}")
                if result is not None:
                    # Return as decimal string
                    if result == int(result):
                        return str(int(result))
                    return str(result)
        # Extract plain number
        num_match = re.search(r"(-?[\d,]+(?:\.\d+)?)", boxed_content)
        if num_match:
            return num_match.group(1).replace(",", "")
        return boxed_content.strip()

    # Priority 3: "The answer is X" or "The final answer is X"
    answer_is_match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*\$?(-?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if answer_is_match:
        return answer_is_match.group(1).replace(",", "")

    # Priority 4: "Final Answer: X"
    final_answer_match = re.search(
        r"Final\s+Answer[:\s]*\$?(-?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if final_answer_match:
        return final_answer_match.group(1).replace(",", "")

    # Priority 5: Percentage (convert to decimal if needed)
    percent_match = re.search(r"(-?[\d,]+(?:\.\d+)?)\s*%", text)
    if percent_match:
        # Return the number as-is (caller can decide to divide by 100)
        return percent_match.group(1).replace(",", "")

    # Priority 6: Standalone fraction like "1/2" or "= 3/4" at end
    fraction_match = re.search(r"[=:\s]\s*(-?\d+)\s*/\s*(\d+)\s*$", text.strip())
    if fraction_match:
        result = _eval_fraction(f"{fraction_match.group(1)}/{fraction_match.group(2)}")
        if result is not None:
            if result == int(result):
                return str(int(result))
            return str(result)

    # Fallback: Last number in the text
    all_numbers = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    if all_numbers:
        return all_numbers[-1].replace(",", "")

    return None


def normalize_numeric_answer(answer: str | None) -> str:
    """Normalize a numeric answer for comparison.

    - Removes commas, whitespace, dollar signs
    - Handles decimal trailing zeros
    - Handles fractions
    - Returns empty string if None
    """
    if answer is None:
        return ""

    # Remove common formatting
    cleaned = re.sub(r"[$,\s]", "", str(answer))

    # Handle fractions
    if "/" in cleaned:
        result = _eval_fraction(cleaned)
        if result is not None:
            if result == int(result):
                return str(int(result))
            # Round to avoid floating point issues
            return str(round(result, 6))

    # Try to parse as number and normalize
    try:
        num = float(cleaned)
        # Return integer if whole number, else float
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return cleaned


def extract_final_answer(text: str) -> str:
    """Extract final answer from reasoning trace.

    Returns the extracted numeric answer or the last line as fallback.
    """
    numeric = extract_numeric_answer(text)
    if numeric:
        return numeric

    # Fallback to last non-empty line
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def extract_ground_truth_answer(answer_text: str) -> str:
    """Extract the ground truth answer from GSM8K format.

    GSM8K answers look like:
    "Some reasoning...\n#### 42"
    """
    numeric = extract_numeric_answer(answer_text)
    if numeric:
        return numeric
    return answer_text.strip()


def answers_match(prediction: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth.

    First extracts numeric values, then normalizes for comparison.
    Handles various formats (#### X, \\boxed{X}, plain numbers, fractions, etc.)
    """
    # Check if inputs are simple fractions - normalize them directly
    def is_simple_fraction(s: str) -> bool:
        return bool(re.match(r"^\s*-?\d+\s*/\s*\d+\s*$", s.strip()))

    # For simple fraction strings, use normalize directly instead of extract
    if is_simple_fraction(prediction):
        pred_normalized = normalize_numeric_answer(prediction)
    else:
        pred_num = extract_numeric_answer(prediction)
        pred_normalized = normalize_numeric_answer(pred_num) if pred_num else normalize_numeric_answer(prediction)

    if is_simple_fraction(ground_truth):
        gt_normalized = normalize_numeric_answer(ground_truth)
    else:
        gt_num = extract_numeric_answer(ground_truth)
        gt_normalized = normalize_numeric_answer(gt_num) if gt_num else normalize_numeric_answer(ground_truth)

    if not pred_normalized or not gt_normalized:
        # Fallback to exact string match
        return prediction.strip() == ground_truth.strip()

    # Try numeric comparison for floating point tolerance
    try:
        pred_float = float(pred_normalized)
        gt_float = float(gt_normalized)
        # Use relative tolerance for floating point comparison
        return abs(pred_float - gt_float) < 1e-6 * max(abs(gt_float), 1)
    except ValueError:
        pass

    return pred_normalized == gt_normalized


def extract_verification(text: str) -> str:
    """Extract verification result (PASS/FAIL) from backward reasoning trace."""
    for line in reversed(text.splitlines()):
        lower = line.strip().lower()
        if lower.startswith("verification"):
            return line.strip()
        # Also check for standalone PASS/FAIL with context
        if "pass" in lower or "fail" in lower:
            if "verification" in lower or "verify" in lower or "result" in lower:
                return line.strip()

    # Check for PASS/FAIL anywhere if not found in standard format
    text_lower = text.lower()
    if "verification: pass" in text_lower or "verification:pass" in text_lower:
        return "Verification: PASS"
    if "verification: fail" in text_lower or "verification:fail" in text_lower:
        return "Verification: FAIL"

    return "Verification: UNKNOWN"
