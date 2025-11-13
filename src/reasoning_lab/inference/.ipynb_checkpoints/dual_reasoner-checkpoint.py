"""Forward/backward inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_lab.utils.text import extract_final_answer, extract_verification


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
        device_map: str | Dict[str, int] | None = "auto",
        torch_dtype: str | torch.dtype = torch.bfloat16,
        hf_token: Optional[str] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=hf_token,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, question: str, *, candidate_answer: Optional[str] = None) -> ReasoningOutputs:
        forward_prompt = (
            "You are a deliberate mathematician. Solve the problem step by step before giving the final"
            " answer. Problem:" f" {question}\n"
        )
        inputs = self.tokenizer(forward_prompt, return_tensors="pt").to(self.model.device)
        forward_output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
        forward_text = self.tokenizer.decode(forward_output[0], skip_special_tokens=True)

        final_answer = candidate_answer or extract_final_answer(forward_text)

        backward_prompt = (
            "You are a verifier. Starting with the candidate answer, reason backwards to confirm the"
            " solution. Report `Verification: PASS` or `Verification: FAIL`. Problem: "
            f"{question}\nCandidate Answer: {final_answer}"
        )
        backward_inputs = self.tokenizer(backward_prompt, return_tensors="pt").to(self.model.device)
        backward_output = self.model.generate(
            **backward_inputs,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )
        backward_text = self.tokenizer.decode(backward_output[0], skip_special_tokens=True)

        verification = extract_verification(backward_text)

        return ReasoningOutputs(
            forward_trace=forward_text,
            backward_trace=backward_text,
            verification=verification,
        )

