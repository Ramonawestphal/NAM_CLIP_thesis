from __future__ import annotations

import torch
import torch.nn as nn

from src.nam.nam import NAM

_bce = nn.BCEWithLogitsLoss()


def output_penalty(model: NAM, x: torch.Tensor) -> torch.Tensor:
    """Mean squared output penalty computed with dropout OFF (eval mode).

    Matches reference graph_builder.py:110-117: calc_outputs(training=False).
    Gradients flow normally; only dropout is suppressed.
    """
    was_training = model.training
    model.eval()
    outputs = model.calc_outputs(x)  # list of K tensors (B,), no dropout
    if was_training:
        model.train()
    stacked = torch.stack(outputs, dim=1)  # (B, K)
    return stacked.pow(2).mean()


def total_loss(
    model: NAM,
    x: torch.Tensor,
    y: torch.Tensor,
    output_reg: float,
    l2_reg: float = 0.0,
) -> torch.Tensor:
    """BCE + output_reg * output_penalty. l2_reg unused (λ2=0 per contract)."""
    logits = model(x)
    loss = _bce(logits, y.float())
    if output_reg > 0.0:
        loss = loss + output_reg * output_penalty(model, x)
    return loss
