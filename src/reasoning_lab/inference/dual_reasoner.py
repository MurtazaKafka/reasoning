"""Forward/backward inference helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_lab.utils.text import extract_final_answer, extract_verification

try:  # Optional dependency; available when bitsandbytes is installed.
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - executed when bitsandbytes absent.
    BitsAndBytesConfig = None  # type: ignore[assignment]


@dataclass
class ReasoningOutputs:
    forward_trace: str
    backward_trace: str
    verification: str


class DualReasoner:
    """Generate paired forward and backward reasoning traces."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
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
    ) -> None:
        if load_in_4bit and load_in_8bit:
            raise ValueError("Choose only one of load_in_4bit or load_in_8bit.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            token=hf_token,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        resolved_dtype = self._normalize_dtype(torch_dtype)
        if load_in_4bit and resolved_dtype == torch.bfloat16:
            resolved_dtype = torch.float16
        model_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": resolved_dtype,
            "token": hf_token,
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

    @torch.inference_mode()
    def generate(self, question: str, *, candidate_answer: Optional[str] = None) -> ReasoningOutputs:
        forward_prompt = (
            "You are a deliberate mathematician. Solve the problem step by step before giving the final"
            " answer. Problem:" f" {question}\n"
        )
        inputs = self._tokenize_to_device(forward_prompt)
        forward_output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_forward_tokens,
            **self.forward_sampling,
        )
        forward_text = self.tokenizer.decode(forward_output[0], skip_special_tokens=True)

        final_answer = candidate_answer or extract_final_answer(forward_text)

        backward_prompt = (
            "You are a verifier. Starting with the candidate answer, reason backwards to confirm the"
            " solution. Report `Verification: PASS` or `Verification: FAIL`. Problem: "
            f"{question}\nCandidate Answer: {final_answer}"
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
        )

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

