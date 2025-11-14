"""Extensions of TRL's DPOTrainer with per-sample weights."""

from __future__ import annotations

from typing import Any

import torch
from trl import DPOTrainer


class WeightedDPOTrainer(DPOTrainer):
    """Adds support for ``weight`` column to modulate the DPO loss."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        dpo_args = kwargs.get("args", None)
        if dpo_args is None and len(args) >= 3:
            dpo_args = args[2]
        if dpo_args is None:
            raise ValueError("WeightedDPOTrainer requires a DPOConfig passed via the 'args' parameter.")

        dpo_args.use_weighting = True
        super().__init__(*args, **kwargs)
        self.use_weighting = True

    def concatenated_forward(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        is_ref_model: bool = False,
    ) -> dict[str, torch.Tensor]:
        output = super().concatenated_forward(model, batch, is_ref_model=is_ref_model)

        if not is_ref_model and "weight" in batch:
            weights = batch["weight"]
            device = getattr(self.args, "device", None)
            if device is None:
                device = next(model.parameters()).device

            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.float32, device=device)
            else:
                weights = weights.to(device=device, dtype=torch.float32)

            policy_weights = output.get("policy_weights")
            if policy_weights is None:
                policy_weights = torch.ones_like(weights, dtype=torch.float32, device=device)
            else:
                policy_weights = policy_weights.to(device=device, dtype=torch.float32)

            output["policy_weights"] = policy_weights * weights

        return output
