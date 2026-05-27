"""
Concurvity regularization for differentiable GAMs.

Implements R_perp from:
  Siems et al. (2023) "Curve Your Enthusiasm: Concurvity Regularization in
  Differentiable Generalized Additive Models." NeurIPS 2023. arXiv:2305.11475

  Training objective: L_total = L_task + lambda * R_perp

Multiclass extension (this module): for K output classes, R_perp^{(k)} is
computed per class using that class's shape function outputs, then averaged:

    R_perp_total = (1/K) * sum_{k=1}^{K} R_perp^{(k)}

where R_perp^{(k)} = mean_{i<j} |Corr(f_{i,k}(X_i), f_{j,k}(X_j))|,
with Pearson correlation taken over the batch dimension.

Both functions are fully vectorized (no Python loops over feature pairs).
"""

from __future__ import annotations

import torch


def pairwise_concurvity(shape_outputs: torch.Tensor) -> torch.Tensor:
    """R_perp for a scalar-output GAM (Siems et al. 2023, Eq. 2).

    Computes the mean absolute pairwise Pearson correlation between shape
    function outputs across all p*(p-1)/2 unordered feature pairs.

    Args:
        shape_outputs: (B, p) — f_i(X_i) for all p features over a batch.

    Returns:
        Scalar in [0, 1].
    """
    B, p = shape_outputs.shape
    mean = shape_outputs.mean(dim=0, keepdim=True)                  # (1, p)
    std  = shape_outputs.std(dim=0, keepdim=True).clamp(min=1e-8)   # (1, p)
    z    = (shape_outputs - mean) / std                             # (B, p)
    # Full (p, p) Pearson correlation matrix: corr[i,j] = z[:,i]·z[:,j] / B
    corr = (z.T @ z) / B                                           # (p, p)
    # Mean |corr| over upper triangle (i < j) — normalises by p*(p-1)/2
    mask = torch.triu(
        torch.ones(p, p, device=shape_outputs.device, dtype=torch.bool),
        diagonal=1,
    )
    return corr[mask].abs().mean()


def multiclass_concurvity(shape_outputs: torch.Tensor) -> torch.Tensor:
    """Multiclass extension of Siems et al. (2023) for K-class NAMs.

    For each class k, computes R_perp^{(k)} over that class's shape function
    outputs, then averages across classes:

        R_perp_total = (1/K) * sum_{k=1}^K  mean_{i<j} |Corr(f_{i,k}, f_{j,k})|

    Fully vectorized: builds all K correlation matrices simultaneously via
    einsum, then indexes the upper triangle mask in one operation.

    Args:
        shape_outputs: (B, p, K) — f_{i,k}(X_i) for all features i, classes k.

    Returns:
        Scalar in [0, 1], averaged over classes and feature pairs.
    """
    B, p, K = shape_outputs.shape
    mean = shape_outputs.mean(dim=0, keepdim=True)                  # (1, p, K)
    std  = shape_outputs.std(dim=0, keepdim=True).clamp(min=1e-8)   # (1, p, K)
    z    = (shape_outputs - mean) / std                             # (B, p, K)
    # corr[k, i, j] = z[:,i,k]·z[:,j,k] / B  — all K matrices at once
    corr = torch.einsum('bik,bjk->kij', z, z) / B                  # (K, p, p)
    # Upper triangle mask shared across classes
    mask = torch.triu(
        torch.ones(p, p, device=shape_outputs.device, dtype=torch.bool),
        diagonal=1,
    )
    # corr[:, mask] → (K, n_pairs); mean over all K * n_pairs elements
    return corr[:, mask].abs().mean()
