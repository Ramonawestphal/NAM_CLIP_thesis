"""
Group LASSO sparsity regularization for multiclass NAM.

Implements the Group LASSO penalty and proximal operator from:
  Xu, S., Bu, Z., Chaudhari, P. & Barnett, I.J. (2022)
  "Sparse Neural Additive Model: Interpretable Deep Learning with
   Feature Selection via Group Sparsity."  arXiv:2202.12482 (ICLR 2022).

  Full training objective:
      L_total = L_task + lambda_concurvity * R_perp + lambda_sparsity * R_sparse

  R_sparse = sum_{i=1}^{p} ||theta_i||_2   (Group LASSO; Yuan & Lin 2006)

  Optimization: the sparsity term is implemented via PROXIMAL GRADIENT
  DESCENT, NOT by adding R_sparse to the loss and differentiating.
  Xu et al. (2022) Theorem 1 proves that proximal gradient achieves exact
  support recovery (groups are driven to exactly zero), while plain
  subgradient only achieves near-zero and causes uniform parameter shrinkage.

  In practice: after each optimizer.step(), call apply_proximal_step().
  group_lasso_penalty() is used only as a diagnostic metric, not in the loss.

Group definition (Xu et al. 2022, §3):
  Each "group" = one feature's sub-network.  theta_i is the flattened
  concatenation of ALL trainable parameters of sub-network i (hidden-layer
  weights, hidden-layer biases, output-layer weights, output-layer biases).
  The global NAM bias (shared across features) is NOT included.

Multiclass note:
  The output projection maps to K classes; those weights are included in
  the group, so the ENTIRE sub-network drops when its group is zeroed —
  consistent with Xu et al.'s framework, which is output-dimension agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.models.nam_multiclass import NAMMulticlass


# ── Core regularization term ──────────────────────────────────────────────────

def group_lasso_penalty(model: "NAMMulticlass") -> torch.Tensor:
    """R_sparse = sum_i ||theta_i||_2  (Xu et al. 2022, Eq. 3).

    Iterates over the p feature sub-networks of the model.  For each,
    concatenates all trainable parameter tensors into a single vector and
    computes its L2 norm.  The norms are summed across sub-networks.

    The global NAM bias (model.bias) is NOT included — it is shared across
    all features and does not belong to any sub-network group.

    Args:
        model: NAMMulticlass instance.  Must expose feature_subnetworks().

    Returns:
        Scalar tensor (requires_grad=True when called outside torch.no_grad()).
    """
    total = torch.zeros(1, device=next(model.parameters()).device)
    for _, subnet in model.feature_subnetworks():
        params = [p.flatten() for p in subnet.parameters() if p.requires_grad]
        if params:
            group_vec = torch.cat(params)
            total = total + torch.linalg.norm(group_vec, ord=2)
    return total.squeeze(0)


# ── Diagnostic utility ────────────────────────────────────────────────────────

@torch.no_grad()
def feature_group_norms(model: "NAMMulticlass") -> Dict[str, float]:
    """L2 norm of each sub-network's parameter vector.

    Used at end of training to report which concepts survived selection.
    Groups with norm below a threshold (default 1e-4) are considered zeroed.

    Args:
        model: NAMMulticlass instance in eval() mode recommended.

    Returns:
        Dict mapping concept_name → ||theta_i||_2 (float).
    """
    norms: Dict[str, float] = {}
    for name, subnet in model.feature_subnetworks():
        params = [p.flatten() for p in subnet.parameters() if p.requires_grad]
        if params:
            group_vec = torch.cat(params)
            norms[name] = float(torch.linalg.norm(group_vec, ord=2).item())
        else:
            norms[name] = 0.0
    return norms


# ── Proximal gradient step ────────────────────────────────────────────────────

def apply_proximal_step(
    model: "NAMMulticlass",
    lr: float,
    sparsity_lambda: float,
    eps: float = 1e-12,
) -> None:
    """Proximal gradient step for Group LASSO (Xu et al. 2022, Theorem 1).

    Applies block-soft-thresholding to each feature sub-network after
    optimizer.step().  For sub-network i with parameter vector theta_i:

        norm_i     = ||theta_i||_2 + eps
        shrink_i   = max(0.0,  1.0 - lr * sparsity_lambda / norm_i)
        theta_i   <- shrink_i * theta_i

    When shrink_i == 0 the entire group is zeroed exactly (exact support
    recovery).  This is the key difference from plain subgradient: subgradient
    drives groups toward zero but never reaches it; proximal zeroing is exact.

    Must be called AFTER optimizer.step(), not before.  The penalty must NOT
    also be added to the loss — that would double-count the regularization.
    group_lasso_penalty() is a diagnostic only when proximal is active.

    Args:
        model:           NAMMulticlass instance.
        lr:              Current learning rate (optimizer.param_groups[0]['lr']).
                         Use the current value, not the initial one; ReduceLROnPlateau
                         changes lr during training and the proximal threshold scales
                         with lr * sparsity_lambda.
        sparsity_lambda: Group LASSO regularization strength.
        eps:             Numerical stability term added to norm denominator.
    """
    with torch.no_grad():
        for _, subnet in model.feature_subnetworks():
            params = [p for p in subnet.parameters() if p.requires_grad]
            if not params:
                continue
            group_vec  = torch.cat([p.flatten() for p in params])
            norm       = torch.linalg.norm(group_vec, ord=2).item()
            shrinkage  = max(0.0, 1.0 - lr * sparsity_lambda / (norm + eps))
            if shrinkage == 0.0:
                for p in params:
                    p.zero_()
            else:
                for p in params:
                    p.mul_(shrinkage)
