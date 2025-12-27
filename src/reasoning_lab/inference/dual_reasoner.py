"""Forward/backward inference helpers."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_lab.utils.text import extract_final_answer, extract_verification

try:  # Optional dependency; available when bitsandbytes is installed.
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - executed when bitsandbytes absent.
    BitsAndBytesConfig = None  # type: ignore[assignment]

try:  # Optional dependency; available when peft is installed.
    from peft import PeftModel
except ImportError:  # pragma: no cover - executed when peft absent.
    PeftModel = None  # type: ignore[assignment]


# Default prompt templates
DEFAULT_FORWARD_PROMPT = """You are an expert problem solver. Solve the following problem step by step.
Show your reasoning clearly, then provide the final answer.

Problem: {question}

Solution:"""

DEFAULT_BACKWARD_PROMPT = """You are a careful verifier. Given a problem and a candidate answer, verify whether the answer is correct by reasoning backwards from the answer.

Problem: {question}
Candidate Answer: {answer}

Verify the solution step by step, then conclude with either:
- Verification: PASS (if the answer is correct)
- Verification: FAIL (if the answer is incorrect)

Verification:"""


def _is_local_path(path: str) -> bool:
    """Check if path is a local filesystem path vs HuggingFace repo ID."""
    return os.path.exists(path) or path.startswith((".", "/", "~"))


def _load_prompt_template(path: Optional[str | Path]) -> Optional[str]:
    """Load a prompt template from file if it exists."""
    if path is None:
        return None
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


@dataclass
class ReasoningOutputs:
    """Single forward/backward reasoning output."""
    forward_trace: str
    backward_trace: str
    verification: str
    extracted_answer: str = ""


@dataclass
class MultiSampleOutputs:
    """Multiple samples for self-consistency evaluation."""
    question: str
    samples: List[ReasoningOutputs] = field(default_factory=list)
    majority_answer: str = ""
    consistency_score: float = 0.0

    def compute_majority(self) -> str:
        """Compute majority vote answer from samples."""
        if not self.samples:
            return ""
        answers = [s.extracted_answer for s in self.samples if s.extracted_answer]
        if not answers:
            return ""
        from collections import Counter
        counter = Counter(answers)
        self.majority_answer = counter.most_common(1)[0][0]
        # Consistency = fraction of samples agreeing with majority
        self.consistency_score = counter[self.majority_answer] / len(answers)
        return self.majority_answer


class DualReasoner:
    """Generate paired forward and backward reasoning traces."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        adapter_path: Optional[str] = None,
        device_map: str | Dict[str, Any] | None = "auto",
        torch_dtype: str | torch.dtype | None = "auto",
        hf_token: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        attn_implementation: Optional[str] = None,
        max_forward_tokens: int = 512,
        max_backward_tokens: int = 256,
        forward_sampling: Optional[Dict[str, Any]] = None,
        backward_sampling: Optional[Dict[str, Any]] = None,
        forward_prompt_template: Optional[str] = None,
        backward_prompt_template: Optional[str] = None,
    ) -> None:
        if load_in_4bit and load_in_8bit:
            raise ValueError("Choose only one of load_in_4bit or load_in_8bit.")

        # Only pass token for remote HuggingFace repos, not local paths
        is_local = _is_local_path(model_name_or_path)
        token_arg = None if is_local else hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            token=token_arg,
            use_fast=True,
            local_files_only=is_local,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        resolved_dtype = self._normalize_dtype(torch_dtype)
        if load_in_4bit and resolved_dtype == torch.bfloat16:
            resolved_dtype = torch.float16
        model_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": resolved_dtype,
            "token": token_arg,
            "local_files_only": is_local,
        }

        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        quantization_config = self._build_quant_config(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        # Load LoRA adapter if provided
        if adapter_path:
            if PeftModel is None:
                raise ImportError(
                    "peft is required to load LoRA adapters. Install it with: pip install peft"
                )
            # Validate adapter checkpoint exists and has required files
            adapter_config_path = Path(adapter_path) / "adapter_config.json"
            if not adapter_config_path.exists():
                raise FileNotFoundError(
                    f"LoRA adapter config not found at {adapter_config_path}. "
                    "The checkpoint may be incomplete or corrupted."
                )
            if adapter_config_path.stat().st_size == 0:
                raise ValueError(
                    f"LoRA adapter config at {adapter_config_path} is empty. "
                    "The checkpoint may have been corrupted during save. "
                    "Please re-run training to generate a valid checkpoint."
                )
            print(f"Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            # Optionally merge for faster inference (if not using quantization)
            if not (load_in_8bit or load_in_4bit):
                print("Merging LoRA weights into base model...")
                self.model = self.model.merge_and_unload()

        self.model.eval()
        self._device = self._infer_primary_device()

        self.max_forward_tokens = max_forward_tokens
        self.max_backward_tokens = max_backward_tokens

        default_forward = {"temperature": 0.7, "top_p": 0.95, "do_sample": True}
        if forward_sampling:
            default_forward.update(forward_sampling)
        self.forward_sampling = default_forward

        default_backward = {"temperature": 0.3, "top_p": 0.9, "do_sample": True}
        if backward_sampling:
            default_backward.update(backward_sampling)
        self.backward_sampling = default_backward

        # Load or use default prompt templates
        loaded_forward = _load_prompt_template(forward_prompt_template)
        loaded_backward = _load_prompt_template(backward_prompt_template)
        self.forward_template = loaded_forward or DEFAULT_FORWARD_PROMPT
        self.backward_template = loaded_backward or DEFAULT_BACKWARD_PROMPT

    @torch.inference_mode()
    def generate(
        self,
        question: str,
        *,
        candidate_answer: Optional[str] = None,
        skip_backward: bool = False,
    ) -> ReasoningOutputs:
        """Generate a single forward/backward reasoning pair.

        Args:
            question: The problem to solve.
            candidate_answer: Optional pre-specified answer for backward verification.
            skip_backward: If True, skip backward verification (faster for forward-only eval).

        Returns:
            ReasoningOutputs with forward trace, backward trace, and verification result.
        """
        # Format forward prompt
        forward_prompt = self.forward_template.format(question=question)

        inputs = self._tokenize_to_device(forward_prompt)
        forward_output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_forward_tokens,
            **self.forward_sampling,
        )
        forward_text = self.tokenizer.decode(forward_output[0], skip_special_tokens=True)

        # Extract answer from forward trace
        final_answer = candidate_answer or extract_final_answer(forward_text)

        if skip_backward:
            return ReasoningOutputs(
                forward_trace=forward_text,
                backward_trace="",
                verification="Verification: SKIPPED",
                extracted_answer=final_answer,
            )

        # Format backward prompt
        backward_prompt = self.backward_template.format(
            question=question,
            answer=final_answer,
        )
        backward_inputs = self._tokenize_to_device(backward_prompt)
        backward_output = self.model.generate(
            **backward_inputs,
            max_new_tokens=self.max_backward_tokens,
            **self.backward_sampling,
        )
        backward_text = self.tokenizer.decode(backward_output[0], skip_special_tokens=True)

        verification = extract_verification(backward_text)

        return ReasoningOutputs(
            forward_trace=forward_text,
            backward_trace=backward_text,
            verification=verification,
            extracted_answer=final_answer,
        )

    @torch.inference_mode()
    def generate_multiple(
        self,
        question: str,
        *,
        num_samples: int = 5,
        skip_backward: bool = False,
    ) -> MultiSampleOutputs:
        """Generate multiple samples for self-consistency evaluation.

        Args:
            question: The problem to solve.
            num_samples: Number of independent samples to generate.
            skip_backward: If True, skip backward verification for speed.

        Returns:
            MultiSampleOutputs with all samples and majority vote answer.
        """
        result = MultiSampleOutputs(question=question)

        for _ in range(num_samples):
            sample = self.generate(question, skip_backward=skip_backward)
            result.samples.append(sample)

        result.compute_majority()
        return result

    @torch.inference_mode()
    def generate_with_rejection_sampling(
        self,
        question: str,
        ground_truth: str,
        *,
        max_attempts: int = 10,
        require_correct: bool = True,
        require_incorrect: bool = False,
    ) -> Optional[ReasoningOutputs]:
        """Generate a sample that matches the desired correctness criterion.

        Useful for creating balanced DPO datasets with both correct and incorrect traces.

        Args:
            question: The problem to solve.
            ground_truth: The correct answer for comparison.
            max_attempts: Maximum generation attempts before giving up.
            require_correct: If True, only return samples with correct answers.
            require_incorrect: If True, only return samples with incorrect answers.

        Returns:
            ReasoningOutputs matching the criterion, or None if not found.
        """
        from reasoning_lab.utils.text import answers_match

        if require_correct and require_incorrect:
            raise ValueError("Cannot require both correct and incorrect simultaneously.")

        for _ in range(max_attempts):
            sample = self.generate(question)
            is_correct = answers_match(sample.extracted_answer, ground_truth)

            if require_correct and is_correct:
                return sample
            elif require_incorrect and not is_correct:
                return sample
            elif not require_correct and not require_incorrect:
                return sample  # Return any sample

        return None

    def _normalize_dtype(self, torch_dtype: str | torch.dtype | None) -> torch.dtype:
        if torch_dtype is None or torch_dtype == "auto":
            return torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype

        dtype_alias = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "full": torch.float32,
        }
        key = torch_dtype.lower()
        if key not in dtype_alias:
            raise ValueError(f"Unsupported torch_dtype value: {torch_dtype}")
        return dtype_alias[key]

    def _build_quant_config(
        self,
        *,
        load_in_8bit: bool,
        load_in_4bit: bool,
    ) -> Any | None:
        if not load_in_8bit and not load_in_4bit:
            return None
        if BitsAndBytesConfig is None:
            warnings.warn(
                "bitsandbytes is not installed; ignoring quantized loading request.",
                stacklevel=2,
            )
            return None

        kwargs: Dict[str, Any] = {}
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True

        return BitsAndBytesConfig(**kwargs)

    def _infer_primary_device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return torch.device(self.model.device)  # type: ignore[arg-type]

        try:
            first_param = next(self.model.parameters())
            return first_param.device
        except StopIteration:
            pass

        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for target in device_map.values():
                if isinstance(target, torch.device):
                    return target
                if isinstance(target, str):
                    return torch.device(target)

        return torch.device("cpu")

    def _tokenize_to_device(self, prompt: str) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self._device) for key, value in inputs.items()}
