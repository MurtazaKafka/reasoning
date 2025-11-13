"""Extensions of TRL's DPOTrainer with per-sample weights."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from trl import DPOTrainer


class WeightedDPOTrainer(DPOTrainer):
    """Adds support for ``weight`` column to modulate the DPO loss."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        sample_weights = inputs.pop("weight", None)
        if sample_weights is not None:
            sample_weights = sample_weights.to(model.device)

        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        if sample_weights is not None:
            # Broadcast weights to match batch size if needed
            if sample_weights.ndim == 1:
                sample_weights = sample_weights.view(-1, 1)
            loss = (loss * sample_weights.squeeze()).mean()
        else:
            loss = loss.mean()

        if return_outputs:
            return loss, outputs
        return loss, None
