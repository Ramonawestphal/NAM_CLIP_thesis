"""Unit tests for shape-function plotting (no saved checkpoints)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch

from src.nam.nam import NAM
from src.utils.plotting import (
    evaluate_shape_function,
    feature_importance,
    infer_binary_feature_indices,
    plot_all_shape_functions,
)


@pytest.fixture
def tiny_ensemble() -> tuple[list[NAM], np.ndarray]:
    seed_everything = __import__(
        "src.utils.seeding", fromlist=["seed_everything"]
    ).seed_everything
    seed_everything(0)
    K, N, M = 4, 40, 3
    X = np.random.randn(N, K).astype(np.float32)
    models = [NAM(n_features=K, dropout=0.0, feature_dropout=0.0) for _ in range(M)]
    for m in models:
        m.eval()
    return models, X


def test_feature_forward_shape(tiny_ensemble):
    models, X = tiny_ensemble
    grid = np.linspace(-1.0, 1.0, 50)
    out = models[0].feature_forward(0, grid)
    assert out.shape == (50,)
    out2 = models[0].feature_forward(1, X[:, 1])
    assert out2.shape == (X.shape[0],)


def test_evaluate_shape_function_centred_mean_zero(tiny_ensemble):
    models, X = tiny_ensemble
    grid = np.linspace(-1.0, 1.0, 30)
    f = evaluate_shape_function(models, 0, grid, X)
    assert f.shape == (len(models), len(grid))
    # Centred on training column → mean over train points ≈ 0 per member
    for m in range(len(models)):
        f_train = evaluate_shape_function([models[m]], 0, X[:, 0], X)
        assert abs(f_train[0].mean()) < 1e-4


def test_feature_importance_positive(tiny_ensemble):
    models, X = tiny_ensemble
    imp = feature_importance(models, X)
    assert imp.shape == (X.shape[1],)
    assert np.all(imp >= 0)


def test_infer_binary_columns():
    X = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.6], [0.0, 1.0, 0.7]])
    assert infer_binary_feature_indices(X) == {0, 1}


def test_plot_all_shape_functions_runs(tmp_path, tiny_ensemble):
    models, X = tiny_ensemble
    names = [f"f{i}" for i in range(X.shape[1])]
    out = tmp_path / "shapes.png"
    fig, imp_df = plot_all_shape_functions(
        ensemble=models,
        X_train=X,
        feature_names=names,
        task="binary",
        top_k=3,
        n_cols=2,
        categorical_features=set(),
        save_path=str(out),
    )
    import matplotlib.pyplot as plt

    plt.close(fig)
    assert out.is_file()
    assert len(imp_df) == X.shape[1]
    assert list(imp_df.columns) == ["feature", "importance"]
