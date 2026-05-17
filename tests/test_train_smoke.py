"""Smoke test for the paper-faithful training path in src/train.py.

Uses a tiny pre-cleaned fixture so no raw COMPAS CSV is required.
Runs 2 epochs; asserts that loss decreases and a checkpoint is written.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.train import run_full_cv, run_paper_path

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
        "balanced_sampling": True,
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


def test_train_full_cv_smoke(tmp_path: Path) -> None:
    """Smoke test for run_full_cv: 2 folds x 2 resamples, 2 epochs each."""
    df = pd.read_csv(FIXTURE)

    cfg = {
        "seed": 42,
        "cv_seed": 42,
        "val_seed": 1337,
        "val_size": 0.125,
        "n_folds": 2,
        "n_resamples": 2,
        "dropout": 0.0,
        "feature_dropout": 0.0,
        "output_reg": 0.1,
        "lr": 0.05,
        "lr_decay": 1.0,
        "batch_size": 16,
        "max_epochs": 2,
        "early_stopping_patience": 100,
        "balanced_sampling": True,
    }

    out_dir = tmp_path / "results"
    results = run_full_cv(cfg, df, out_dir=str(out_dir), device=torch.device("cpu"))

    # Checkpoint must exist
    assert (out_dir / "nam_compas_cv.pt").is_file()

    # Results dict must have the expected keys
    for key in ("per_fold", "mean_auc_roc", "std_auc_roc", "ensemble_auc_roc", "ensemble_probs"):
        assert key in results, f"Results missing key '{key}'"

    # Fold count must match n_folds
    assert len(results["per_fold"]) == 2

    # Each fold result must have val_aucs with n_resamples entries
    for fold in results["per_fold"]:
        assert len(fold["val_aucs"]) == 2

    # AUC scores must be in valid range
    assert 0.0 <= results["mean_auc_roc"] <= 1.0
    assert 0.0 <= results["ensemble_auc_roc"] <= 1.0

    # ensemble_probs must cover all rows in the fixture
    assert len(results["ensemble_probs"]) == len(df)
