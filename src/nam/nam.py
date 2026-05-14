from __future__ import annotations

import torch
import torch.nn as nn

from src.nam.feature_nn import FeatureNN


class NAM(nn.Module):
    """Neural Additive Model: K feature sub-networks summed with feature dropout."""

    def __init__(
        self,
        n_features: int,
        dropout: float = 0.1,
        feature_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.feature_nns = nn.ModuleList(
            [FeatureNN(dropout=dropout) for _ in range(n_features)]
        )
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.bias = nn.Parameter(torch.zeros(1))

    def calc_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return per-feature outputs [(B,), ...] honouring current training state."""
        return [nn(x[:, i]) for i, nn in enumerate(self.feature_nns)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K) → logits (B,)"""
        outputs = self.calc_outputs(x)        # K tensors of shape (B,)
        stacked = torch.stack(outputs, dim=1) # (B, K)
        stacked = self.feature_dropout(stacked)
        return stacked.sum(dim=1) + self.bias
