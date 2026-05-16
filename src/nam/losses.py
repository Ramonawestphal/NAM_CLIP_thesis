from __future__ import annotations

import torch
import torch.nn as nn

from src.nam.nam import NAM

_bce = nn.BCEWithLogitsLoss()


def output_penalty(model: NAM, x: torch.Tensor) -> torch.Tensor:
    """Output penalty per paper eq. (3): (1/K) Σ_k mean_n[f_k(x_kn)²].

    Uses the current model.training state — no eval() switch. Gradients flow.
    The BCE term and the penalty term in total_loss therefore see the same
    stochastic forward pass, removing the inconsistency in the previous design.
    """
    outputs = model.calc_outputs(x)  # honours current train/eval state
    K = len(outputs)
    stacked = torch.stack(outputs, dim=1)  # (B, K)
    # eq. (3): (1/K) Σ_k mean_n[f_k²]
    return stacked.pow(2).mean(dim=0).sum() / K


def total_loss(
    model: NAM,
    x: torch.Tensor,
    y: torch.Tensor,
    output_reg: float,
    l2_reg: float = 0.0,
) -> torch.Tensor:
    """Training loss with a single calc_outputs call shared by BCE and penalty.

    BCE term and output penalty see the same per-feature outputs (same stochastic
    draw), so gradients are consistent. Feature dropout is applied here via the
    model's existing dropout module so the logit matches NAM.forward().
    l2_reg is reserved but unused (λ₂=0 per contract).
    """
    outputs = model.calc_outputs(x)        # list of K tensors (B,)
    K = len(outputs)
    stacked = torch.stack(outputs, dim=1)  # (B, K), pre-feature-dropout

    # Feature dropout: plain Bernoulli zeroing without inverted scaling (fix #3).
    # Mirrors NAM.forward() exactly so training with total_loss is consistent.
    if model.training and model.feature_dropout_p > 0:
        mask = (torch.rand(K, device=stacked.device) > model.feature_dropout_p).float()
        dropped = stacked * mask
    else:
        dropped = stacked

    logits = dropped.sum(dim=1) + model.bias
    loss = _bce(logits, y.float())

    if output_reg > 0.0:
        # Penalty on pre-dropout per-feature outputs — same calc_outputs call, no extra pass
        # eq. (3): (1/K) Σ_k mean_n[f_k²]
        loss = loss + output_reg * stacked.pow(2).mean(dim=0).sum() / K

    return loss
