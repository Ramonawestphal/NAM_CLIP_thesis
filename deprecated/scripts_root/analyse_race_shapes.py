"""Analyse and plot centred race shape functions from a saved NAM checkpoint.

Usage:
    python scripts/analyse_race_shapes.py

Requires:
    - results/nam_compas_paper.pt   (saved by run_paper_path)
    - results/compas_clean_v1.csv   (saved by load_compas)

Reconstructs the same training split (deterministic seeds) so
center_shape_functions can be called on the correct data.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that src.* modules can be found
# when torch.load deserialises the pickled CompasEncoder stored in the checkpoint.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")  # headless — saves to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results" / "nam_compas_paper.pt"
CLEAN_CSV = ROOT / "results" / "compas_clean_v1.csv"
OUT_PNG = ROOT / "results" / "race_shape_functions.png"


def main() -> None:
    # ------------------------------------------------------------------ load
    saved = torch.load(CKPT, weights_only=False)
    encoder = saved["encoder"]
    feature_names: list[str] = saved["feature_names"]
    cfg: dict = saved["config"]

    from src.nam.nam import NAM
    model = NAM(
        n_features=12,
        dropout=float(cfg.get("dropout", 0.1)),
        feature_dropout=float(cfg.get("feature_dropout", 0.05)),
    )
    model.load_state_dict(saved["model"])
    model.eval()

    # ---------------------------------------------- reconstruct training split
    df = pd.read_csv(CLEAN_CSV)
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

    X_train = torch.tensor(encoder.transform(X_train_df), dtype=torch.float32)

    # --------------------------------------- centre shape functions on train set
    model.center_shape_functions(X_train)
    offsets = model.shape_fn_offsets  # (K,)

    # -------------------------------------------------- evaluate race features
    # After Fix B: feature_names[0:6] = race OHE, [6:7] = sex OHE, [8:11] = continuous
    race_names = [n for n in feature_names if n.startswith("race_")]
    race_indices = [feature_names.index(n) for n in race_names]

    # For each race feature k, evaluate f_k on a fine grid in [-1, +1]
    grid = torch.linspace(-1.0, 1.0, 200).unsqueeze(1)  # (200, 1)

    print("\n" + "=" * 60)
    print("Race shape function values at x = +1.0 (centred)")
    print("  (positive = increases predicted recidivism risk)")
    print("=" * 60)

    contributions: dict[str, float] = {}
    with torch.no_grad():
        for name, k in zip(race_names, race_indices):
            f_grid = model.feature_nns[k](grid).squeeze() - offsets[k]
            f_at_plus1 = (model.feature_nns[k](torch.tensor([[1.0]])) - offsets[k]).item()
            contributions[name] = f_at_plus1

    for name, val in sorted(contributions.items(), key=lambda x: -x[1]):
        bar = "#" * int(abs(val) * 40 / max(abs(v) for v in contributions.values()))
        sign = "+" if val >= 0 else "-"
        label = name.replace("race_", "")
        print(f"  {label:20s}  {sign}{abs(val):.4f}  {bar}")

    print()
    top = max(contributions, key=contributions.__getitem__).replace("race_", "")
    bottom_two = sorted(contributions, key=contributions.__getitem__)[:2]
    bottom_two_labels = [n.replace("race_", "") for n in bottom_two]
    print(f"Highest contribution: {top}")
    print(f"Lowest contributions: {', '.join(bottom_two_labels)}")

    # ------------------------------------------------------ plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    axes = axes.flatten()

    with torch.no_grad():
        for ax, (name, k) in zip(axes, zip(race_names, race_indices)):
            f_vals = (model.feature_nns[k](grid) - offsets[k]).numpy()
            ax.plot(grid.numpy(), f_vals, lw=2, color="steelblue")
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.axvline(0, color="gray", lw=0.8, ls="--")
            ax.set_title(name.replace("race_", ""), fontsize=11)
            ax.set_xlabel("encoded input", fontsize=9)
            ax.set_ylabel("f̃(x)", fontsize=9)
            # Mark contribution at +1
            f_at_1 = contributions[name]
            ax.scatter([1.0], [f_at_1], color="crimson", zorder=5, s=40,
                       label=f"f(+1)={f_at_1:.3f}")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Centred race shape functions (NAM replication)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nPlot saved to {OUT_PNG}")


if __name__ == "__main__":
    main()
