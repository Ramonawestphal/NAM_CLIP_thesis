"""
Multiclass Neural Additive Model.

Extends the binary NAM in src/nam/nam.py to C-class multinomial output without
modifying the original. Each concept sub-network maps a scalar input to R^C;
contributions are summed across K concepts to produce (B, C) logits.

Architecture mirrors the Agarwal et al. 2021 design:
  - Hidden dims: configurable, default (64, 64, 32)
  - Activations: ReLU (default) or ExU (configurable)
  - Dropout: applied after each hidden layer
  - Output linear: Linear(last_hidden, C, bias=False); global bias (C,) handles intercept

Usage:
    model = NAMMulticlass(n_features=72, num_classes=7)
    logits = model(x)                         # (B, 72) → (B, 7)
    contribs = model.shape_outputs(x)         # (B, 72) → (B, 72, 7)
    f_i = model.concept_contributions(x_i, i) # (N,) → (N, 7)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class FeatureNNMulticlass(nn.Module):
    """Single-concept sub-network: scalar → R^C.

    Builds a [hidden_dims] MLP with ReLU activations and inter-layer dropout,
    then a final Linear(last_dim, num_classes, bias=False).
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dims: Sequence[int] = (64, 64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 1
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        # bias=False: intercept is absorbed by NAMMulticlass.bias (shape C)
        layers.append(nn.Linear(in_dim, num_classes, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) → (B, C)"""
        return self.net(x.unsqueeze(1))


class NAMMulticlass(nn.Module):
    """Multiclass Neural Additive Model.

    K concept sub-networks each produce (B, C) contributions; summing gives
    the (B, C) multinomial logits.  The per-class global bias is a learnable
    parameter of shape (C,).

    Args:
        n_features:   Number of concept scores (K=72 for BiomedCLIP v5).
        num_classes:  Number of output classes (C=7 for HAM10000).
        hidden_dims:  Hidden layer widths for each sub-network.
        dropout:      Dropout probability inside sub-networks.
        feature_dropout: Probability of zeroing an entire concept contribution
                        during training (plain Bernoulli, no rescaling).
                        Set to 0.0 to disable (default).
    """

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (64, 64, 32),
        dropout: float = 0.1,
        feature_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.num_classes = num_classes
        self.feature_dropout_p = float(feature_dropout)
        self.feature_nns = nn.ModuleList([
            FeatureNNMulticlass(num_classes, hidden_dims, dropout)
            for _ in range(n_features)
        ])
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(
        self,
        x: torch.Tensor,
        return_shape_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """x: (B, K) → logits (B, C).

        Args:
            x: Input concept scores, shape (B, K).
            return_shape_outputs: If True, also return the pre-sum per-feature
                contributions as a (B, K, C) tensor.  Default False keeps the
                return signature unchanged for all existing callers.

        Returns:
            logits (B, C), or (logits (B, C), shape_outputs (B, K, C)) when
            return_shape_outputs=True.
        """
        # stack all sub-network outputs: (B, K, C)
        stacked = torch.stack(
            [nn(x[:, i]) for i, nn in enumerate(self.feature_nns)], dim=1
        )
        if self.training and self.feature_dropout_p > 0:
            # Zero entire concept contributions (no rescaling, matches binary NAM)
            mask = (
                torch.rand(stacked.shape[1], device=stacked.device)
                > self.feature_dropout_p
            ).float()
            stacked = stacked * mask.unsqueeze(0).unsqueeze(-1)
        logits = stacked.sum(dim=1) + self.bias  # (B, C)
        if return_shape_outputs:
            return logits, stacked
        return logits

    def shape_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Per-concept contributions without feature dropout or bias.

        x: (B, K) → (B, K, C)

        Use this to inspect how each concept contributes to each class.
        Dropout is NOT applied here (call model.eval() first).
        """
        return torch.stack(
            [nn(x[:, i]) for i, nn in enumerate(self.feature_nns)], dim=1
        )

    def concept_contributions(
        self,
        x_i: torch.Tensor | np.ndarray,
        concept_idx: int,
    ) -> torch.Tensor:
        """Evaluate the shape function for one concept across a range of values.

        Args:
            x_i:         1D array/tensor of concept score values (N grid points).
            concept_idx: Which sub-network to query (0 ≤ idx < K).

        Returns:
            Tensor of shape (N, C) — the contribution of this concept to each class
            logit at each input value.  Dropout is off (switches to eval temporarily).
        """
        was_training = self.training
        self.eval()
        device = self.bias.device
        if isinstance(x_i, np.ndarray):
            xt = torch.as_tensor(x_i, dtype=torch.float32, device=device)
        else:
            xt = x_i.to(device=device, dtype=torch.float32)
        if xt.dim() > 1:
            xt = xt.reshape(-1)
        with torch.no_grad():
            out = self.feature_nns[concept_idx](xt)  # (N, C)
        if was_training:
            self.train()
        return out
