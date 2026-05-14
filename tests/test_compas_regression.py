from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.data.compas import compas_numeric_feature_matrix, load_compas_frame
from src.nam import NeuralAdditiveModel


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "compas_tiny.csv"


def test_compas_fixture_loads_and_shapes_match() -> None:
    df = load_compas_frame(FIXTURE)
    X_df, names = compas_numeric_feature_matrix(df)
    assert names == list(X_df.columns)
    assert len(df) == X_df.shape[0]
    assert "two_year_recid" in df.columns


def test_nam_forward_and_single_step_over_compas_fixture() -> None:
    df = load_compas_frame(FIXTURE)
    X_df, _ = compas_numeric_feature_matrix(df)
    y = torch.tensor(df["two_year_recid"].values, dtype=torch.float32)
    X = torch.tensor(X_df.values, dtype=torch.float32)
    model = NeuralAdditiveModel(num_features=X.shape[1], hidden_dims=(16, 16))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    logits = model(X)
    loss0 = float(loss_fn(logits, y).detach())
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    loss1 = float(loss_fn(model(X), y).detach())
    assert loss1 <= loss0 + 1e-4
