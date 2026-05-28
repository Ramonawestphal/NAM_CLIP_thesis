from __future__ import annotations

import numpy as np
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
        # Plain float — feature dropout uses a Bernoulli mask without inverted scaling.
        # nn.Dropout would scale survivors by 1/(1-p), inflating contributions by ~5%;
        # the paper's intent is to zero an entire feature's contribution, not rescale.
        self.feature_dropout_p = float(feature_dropout)
        self.bias = nn.Parameter(torch.zeros(1))

    def calc_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return per-feature outputs [(B,), ...] honouring current training state.

        Does NOT apply feature dropout or bias — used for penalty computation and plots.
        """
        return [nn(x[:, i]) for i, nn in enumerate(self.feature_nns)]

    def feature_forward(
        self,
        k: int,
        x: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Evaluate shape function f_k on a 1D vector of values for feature k only.

        Returns raw sub-network outputs (not mean-centered). Dropout is off (eval mode).
        Does not add the global bias or touch other features.

        Used by ``src.utils.plotting`` for Figure 4–style panels; centring is applied
        there by subtracting the training-set mean per ensemble member.
        """
        was_training = self.training
        self.eval()
        device = self.bias.device
        if isinstance(x, np.ndarray):
            xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            xt = x.to(device=device, dtype=torch.float32)
        if xt.dim() > 1:
            xt = xt.reshape(-1)
        with torch.no_grad():
            out = self.feature_nns[k](xt)
        if was_training:
            self.train()
        return out

    def center_shape_functions(self, x_train: torch.Tensor) -> None:
        """Mean-center shape functions on training data. Call once after training, before plotting.

        For each feature net k, computes offset_k = E_n[f_k(x_kn)] over x_train.
        Stores offsets as a buffer (self.shape_fn_offsets) so they survive state_dict
        save/load, then absorbs their sum into self.bias so forward-pass predictions
        are unchanged.

        Plotting code subtracts shape_fn_offsets[k] from calc_outputs()[k] to obtain
        the mean-centered f̃_k used in Figure 4.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            outputs = self.calc_outputs(x_train)  # list of K tensors (N,)
        means = torch.stack([o.mean() for o in outputs])  # (K,)
        self.register_buffer("shape_fn_offsets", means)
        self.bias.data = self.bias.data + means.sum()
        if was_training:
            self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K) → logits (B,)"""
        outputs = self.calc_outputs(x)         # K tensors of shape (B,)
        stacked = torch.stack(outputs, dim=1)  # (B, K)
        if self.training and self.feature_dropout_p > 0:
            # Plain zeroing: each feature's contribution is independently dropped
            # with probability feature_dropout_p, without rescaling survivors.
            mask = (torch.rand(stacked.shape[1], device=stacked.device)
                    > self.feature_dropout_p).float()
            stacked = stacked * mask
        return stacked.sum(dim=1) + self.bias
