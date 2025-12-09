"""Test text extraction utilities."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reasoning_lab.utils.text import (
    extract_numeric_answer,
    extract_ground_truth_answer,
    answers_match,
    normalize_numeric_answer,
)


def test_gsm8k_format():
    """Test extraction from GSM8K #### format."""
    text = "Some calculation\n#### 72"
    assert extract_numeric_answer(text) == "72"
    
    text2 = "Natalia sold 48+24 = 72 clips.\n#### 72"
    assert extract_ground_truth_answer(text2) == "72"


def test_boxed_format():
    """Test extraction from LaTeX boxed format."""
    text = r"The final answer is: $\boxed{72}$"
    assert extract_numeric_answer(text) == "72"
    
    text2 = r"Therefore \boxed{42}"
    assert extract_numeric_answer(text2) == "42"


def test_answer_is_format():
    """Test 'the answer is X' format."""
    text = "The answer is 42"
    assert extract_numeric_answer(text) == "42"
    
    text2 = "the final answer is: $10"
    assert extract_numeric_answer(text2) == "10"


def test_final_answer_format():
    """Test 'Final Answer: X' format."""
    text = "Final Answer: 5"
    assert extract_numeric_answer(text) == "5"


def test_answers_match():
    """Test answer matching with various formats."""
    # Same number, different formats
    assert answers_match("72", "#### 72")
    assert answers_match("72", "72")
    assert answers_match("42", "The answer is 42")
    
    # With commas
    assert answers_match("1000", "1,000")
    
    # Floats
    assert answers_match("10.0", "10")
    assert answers_match("10.5", "10.5")


def test_normalize():
    """Test numeric normalization."""
    assert normalize_numeric_answer("72") == "72"
    assert normalize_numeric_answer("1,000") == "1000"
    assert normalize_numeric_answer("$42") == "42"
    assert normalize_numeric_answer("10.0") == "10"


if __name__ == "__main__":
    test_gsm8k_format()
    print("✓ GSM8K format extraction works")
    
    test_boxed_format()
    print("✓ Boxed format extraction works")
    
    test_answer_is_format()
    print("✓ 'Answer is' format extraction works")
    
    test_final_answer_format()
    print("✓ 'Final Answer' format extraction works")
    
    test_answers_match()
    print("✓ Answer matching works")
    
    test_normalize()
    print("✓ Normalization works")
    
    print("\n✅ All tests passed!")
