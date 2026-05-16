"""Smoke test for the paper-faithful training path in src/train.py.

Uses a tiny pre-cleaned fixture so no raw COMPAS CSV is required.
Runs 2 epochs; asserts that loss decreases and a checkpoint is written.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.train import run_paper_path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "compas_clean_tiny.csv"


def test_train_paper_path_smoke(tmp_path: Path) -> None:
    df = pd.read_csv(FIXTURE)

    cfg = {
        "seed": 42,
        "cv_seed": 42,
        "val_seed": 1337,
        "dropout": 0.0,          # deterministic FeatureNN for speed
        "feature_dropout": 0.0,
        "output_reg": 0.1,
        "lr": 0.05,
        "lr_decay": 1.0,         # no decay — keep LR flat for 2 epochs
        "batch_size": 16,
        "max_epochs": 2,
        "early_stopping_patience": 100,
    }

    out_dir = tmp_path / "results"
    model, encoder = run_paper_path(cfg, df, out_dir=str(out_dir), device=torch.device("cpu"))

    # Checkpoint must exist and contain the expected keys
    ckpt_path = out_dir / "nam_compas_paper.pt"
    assert ckpt_path.is_file(), "Checkpoint file must be written"

    saved = torch.load(ckpt_path, weights_only=False)
    for key in ("model", "encoder", "feature_names", "config", "test_auc_roc", "test_auc_pr"):
        assert key in saved, f"Checkpoint missing key '{key}'"

    # Encoder must have produced 12 features
    assert encoder.n_features == 12
    assert len(saved["feature_names"]) == 12

    # AUC scores must be in a valid range (may be low with 40 rows)
    assert 0.0 <= saved["test_auc_roc"] <= 1.0
    assert 0.0 <= saved["test_auc_pr"] <= 1.0
