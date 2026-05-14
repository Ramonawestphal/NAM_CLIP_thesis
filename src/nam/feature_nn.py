from __future__ import annotations

import torch
import torch.nn as nn


class FeatureNN(nn.Module):
    """Single-feature sub-network: 3 hidden Dense-ReLU-Dropout layers + Linear(32,1,bias=False)."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) or (B,1) → (B,)"""
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.net(x).squeeze(1)
