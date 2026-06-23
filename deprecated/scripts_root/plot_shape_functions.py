"""Plot NAM shape-function panels from a saved paper-path checkpoint.

Usage (from repo root, venv active):

    python scripts/plot_shape_functions.py

Requires:
    results/nam_compas_paper.pt
    results/compas_clean_v1.csv  (or data/compas/compas_clean_v1.csv)

For a true multi-model ensemble (20–100 members), pass multiple checkpoints or
extend training to save all CV resample weights; this script uses a single model
as a one-member ensemble by default.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from src.nam.nam import NAM
from src.utils.plotting import compas_categorical_indices, plot_all_shape_functions

ROOT = _ROOT
CKPT = ROOT / "results" / "nam_compas_paper.pt"
CLEAN_CSV = ROOT / "results" / "compas_clean_v1.csv"
if not CLEAN_CSV.is_file():
    CLEAN_CSV = ROOT / "data" / "compas" / "compas_clean_v1.csv"
OUT = ROOT / "results" / "shape_functions.png"


def load_training_matrix(
    cfg: dict,
    encoder,
    clean_csv: Path,
) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(clean_csv)
    y_all = df["two_year_recid"].values.astype(np.float32)
    X_df = df.drop(columns=["two_year_recid"])

    sss_outer = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=int(cfg.get("cv_seed", 42))
    )
    trval_idx, _ = next(sss_outer.split(X_df, y_all))
    X_trval = X_df.iloc[trval_idx]
    y_trval = y_all[trval_idx]

    sss_inner = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(cfg.get("val_size", 0.125)),
        random_state=int(cfg.get("val_seed", 1337)),
    )
    tr_idx, _ = next(sss_inner.split(X_trval, y_trval))
    X_train_df = X_trval.iloc[tr_idx]
    X_train = encoder.transform(X_train_df)
    return X_train, list(encoder.feature_names_)


def main() -> None:
    if not CKPT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")
    if not CLEAN_CSV.is_file():
        raise FileNotFoundError(f"Clean COMPAS CSV not found: {CLEAN_CSV}")

    saved = torch.load(CKPT, weights_only=False)
    encoder = saved["encoder"]
    feature_names: list[str] = saved["feature_names"]
    cfg: dict = saved["config"]

    model = NAM(
        n_features=len(feature_names),
        dropout=float(cfg.get("dropout", 0.1)),
        feature_dropout=float(cfg.get("feature_dropout", 0.05)),
    )
    model.load_state_dict(saved["model"])
    model.eval()

    X_train, names = load_training_matrix(cfg, encoder, CLEAN_CSV)
    assert names == feature_names

    # Single checkpoint → ensemble of size 1; replace with list of M models when available.
    ensemble = [model]

    cat_idx = compas_categorical_indices(feature_names)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, imp_df = plot_all_shape_functions(
        ensemble=ensemble,
        X_train=X_train,
        feature_names=feature_names,
        task="binary",
        top_k=12,
        categorical_features=cat_idx,
        save_path=str(OUT),
    )
    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    plt.close(fig)

    print(imp_df.head(20).to_string(index=False))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
