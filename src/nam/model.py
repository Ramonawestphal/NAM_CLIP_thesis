from __future__ import annotations

import torch
import torch.nn as nn


class _FeatureSubnetwork(nn.Module):
    """One scalar output subnetwork per input feature (1-D path for NAM)."""

    def __init__(self, hidden_dims: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 1
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(inplace=True)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(-1)).squeeze(-1)


class NeuralAdditiveModel(nn.Module):
    """Additive ensemble of per-feature networks plus optional bias."""

    def __init__(
        self,
        num_features: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        output_bias: bool = True,
    ) -> None:
        super().__init__()
        self.subnets = nn.ModuleList(
            [_FeatureSubnetwork(hidden_dims) for _ in range(num_features)]
        )
        self.bias = nn.Parameter(torch.zeros(1)) if output_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("x must be [batch, num_features]")
        parts = torch.stack([net(x[:, i]) for i, net in enumerate(self.subnets)], dim=1)
        out = parts.sum(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out
