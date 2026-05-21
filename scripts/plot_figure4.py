"""Replicate NAM paper Figure 4 — shape functions for COMPAS features.

Prefers results/nam_compas_cv.pt (shape functions averaged across all
100 CV models) when available; falls back to results/nam_compas_paper.pt
(single-split model) otherwise.

Layout:
  Row 1 — age, priors count, length of stay   (continuous line plots)
  Row 2 — race, sex, charge degree             (categorical bar charts)

Saved to results/figure4_replication.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

CV_CKPT     = _ROOT / "results" / "nam_compas_cv.pt"
SINGLE_CKPT = _ROOT / "results" / "nam_compas_paper.pt"
CLEAN_CSV   = _ROOT / "results" / "compas_clean_v1.csv"
OUT_PNG     = _ROOT / "results" / "figure4_replication.png"

C_POS   = "#2171b5"
C_NEG   = "#cb181d"
C_ZERO  = "#bdbdbd"
ALPHA_FILL = 0.18


# ── helpers ───────────────────────────────────────────────────────────────────

def _bar_colors(values):
    return [C_POS if v >= 0 else C_NEG for v in values]


def _inv_col(scaler, col_idx: int, scaled: np.ndarray) -> np.ndarray:
    d_min = scaler.data_min_[col_idx]
    d_max = scaler.data_max_[col_idx]
    return d_min + (scaled + 1.0) / 2.0 * (d_max - d_min)


def _style_ax(ax, title: str, xlabel: str, ylabel: str = "shape function output"):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.axhline(0, color=C_ZERO, lw=0.9, ls="--", zorder=0)
    ax.grid(True, axis="y", color=C_ZERO, lw=0.5, alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _plot_line(ax, x_orig, f_vals, title, xlabel):
    ax.plot(x_orig, f_vals, color=C_POS, lw=2, zorder=3)
    ax.fill_between(x_orig, f_vals, 0,
                    where=(f_vals >= 0), color=C_POS, alpha=ALPHA_FILL, zorder=2)
    ax.fill_between(x_orig, f_vals, 0,
                    where=(f_vals < 0),  color=C_NEG, alpha=ALPHA_FILL, zorder=2)
    _style_ax(ax, title, xlabel)


def _plot_bars(ax, labels, values, title, rotate=False):
    bars = ax.bar(
        labels, values,
        color=_bar_colors(values), alpha=0.85,
        edgecolor="white", linewidth=0.5, zorder=3,
    )
    _style_ax(ax, title, "")
    if rotate:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    for bar, val in zip(bars, values):
        va = "bottom" if val >= 0 else "top"
        off = 0.004 if val >= 0 else -0.004
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + off, f"{val:+.3f}",
            ha="center", va=va, fontsize=7, color="#333333",
        )


# ── load shape data ───────────────────────────────────────────────────────────

def load_from_cv(path: Path):
    """Load pre-averaged shape functions from run_full_cv checkpoint."""
    saved = torch.load(path, weights_only=False)
    assert "shape_fn_mean" in saved, (
        "CV checkpoint is missing shape_fn_mean — re-run with the updated run_full_cv."
    )
    grid      = saved["shape_fn_grid"]        # (N_GRID,) in [-1, 1]
    curves    = saved["shape_fn_mean"]        # (12, N_GRID)
    encoder   = saved["encoder"]
    feat_names = saved["feature_names"]
    n_models  = saved["shape_fn_n_models"]
    cfg       = saved["config"]
    print(f"Loaded CV checkpoint: averaged over {n_models} models")
    print(f"Ensemble AUC-ROC = {saved['ensemble_auc_roc']:.4f}  "
          f"mean fold AUC = {saved['mean_auc_roc']:.4f} +/- {saved['std_auc_roc']:.4f}")
    return grid, curves, encoder, feat_names, cfg, n_models


def load_from_single(path: Path):
    """Load and centre shape functions from a single-split checkpoint."""
    from src.nam.nam import NAM
    saved = torch.load(path, weights_only=False)
    encoder   = saved["encoder"]
    feat_names = saved["feature_names"]
    cfg       = saved["config"]

    model = NAM(
        n_features=12,
        dropout=float(cfg.get("dropout", 0.1)),
        feature_dropout=float(cfg.get("feature_dropout", 0.05)),
    )
    model.load_state_dict(saved["model"])
    model.eval()

    # Reconstruct training split to centre shape functions
    df = pd.read_csv(CLEAN_CSV)
    y_all = df["two_year_recid"].values.astype(np.float32)
    X_df = df.drop(columns=["two_year_recid"])

    sss_outer = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2,
        random_state=int(cfg.get("cv_seed", 42)),
    )
    trval_idx, _ = next(sss_outer.split(X_df, y_all))
    X_trval, y_trval = X_df.iloc[trval_idx], y_all[trval_idx]

    sss_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=float(cfg.get("val_size", 0.125)),
        random_state=int(cfg.get("val_seed", 1337)),
    )
    tr_idx, _ = next(sss_inner.split(X_trval, y_trval))
    X_train_enc = torch.tensor(encoder.transform(X_trval.iloc[tr_idx]), dtype=torch.float32)

    model.center_shape_functions(X_train_enc)
    offsets = model.shape_fn_offsets

    N_GRID = 300
    grid = torch.linspace(-1.0, 1.0, N_GRID).numpy()
    grid_t = torch.tensor(grid).unsqueeze(1)
    curves = np.zeros((12, N_GRID))
    with torch.no_grad():
        for k in range(12):
            curves[k] = (model.feature_nns[k](grid_t) - offsets[k]).numpy().flatten()

    print("Loaded single-split checkpoint (1 model)")
    print(f"Test AUC-ROC = {saved['test_auc_roc']:.4f}")
    return grid, curves, encoder, feat_names, cfg, 1


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if CV_CKPT.is_file():
        try:
            grid, curves, encoder, feat_names, cfg, n_models = load_from_cv(CV_CKPT)
            source_label = f"100-model CV ensemble ({n_models} models averaged)"
        except AssertionError as e:
            print(f"WARNING: {e}")
            print("Falling back to single-split model.")
            grid, curves, encoder, feat_names, cfg, n_models = load_from_single(SINGLE_CKPT)
            source_label = "single-split model (1 model)"
    else:
        grid, curves, encoder, feat_names, cfg, n_models = load_from_single(SINGLE_CKPT)
        source_label = "single-split model (1 model)"

    scaler = encoder._scaler

    # Column layout (Fix B):
    #   0-5  race OHE  (African-American, Asian, Caucasian, Hispanic, Native American, Other)
    #   6-7  sex OHE   (Female, Male)
    #   8    age  9  charge_degree  10  length_of_stay  11  priors_count

    race_labels   = [n.replace("race_", "") for n in feat_names[:6]]
    sex_labels    = [n.replace("sex_",  "") for n in feat_names[6:8]]

    # Categorical contributions: value of curve at x = +1.0 (the active OHE state)
    # grid[-1] == +1.0 exactly (linspace endpoint)
    race_vals     = [float(curves[k, -1]) for k in range(6)]
    sex_vals      = [float(curves[k, -1]) for k in range(6, 8)]
    # charge_degree: Felony encoded as -1.0 (grid[0]), Misdemeanor as +1.0 (grid[-1])
    charge_labels = ["Felony", "Misdemeanor"]
    charge_vals   = [float(curves[9, 0]), float(curves[9, -1])]

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 3, figsize=(14, 8),
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )

    # Row 0: continuous features
    for ax, col_idx, title, xlabel in [
        (axes[0, 0],  8, "Age",            "age (years)"),
        (axes[0, 1], 11, "Priors Count",   "number of prior charges"),
        (axes[0, 2], 10, "Length of Stay", "days in jail pre-trial"),
    ]:
        x_orig = _inv_col(scaler, col_idx, grid)
        _plot_line(ax, x_orig, curves[col_idx], title, xlabel)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))

    # Row 1: categorical features
    _plot_bars(axes[1, 0], race_labels,   race_vals,   "Race",          rotate=True)
    _plot_bars(axes[1, 1], sex_labels,    sex_vals,    "Sex")
    _plot_bars(axes[1, 2], charge_labels, charge_vals, "Charge Degree")

    fig.suptitle(
        "NAM shape functions — COMPAS recidivism (replication of Figure 4)\n"
        f"y-axis: mean-centred log-odds contribution  |  source: {source_label}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.02,
        "Positive values increase predicted recidivism risk; negative values decrease it.",
        ha="center", fontsize=8, color="#555555",
    )

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nSaved {OUT_PNG}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\nShape function values at active input (+1.0):")
    print(f"{'Feature':<26}  {'f(active)':>10}")
    print("-" * 40)
    for label, val in zip(race_labels, race_vals):
        print(f"  race: {label:<19}  {val:>+10.4f}")
    for label, val in zip(sex_labels, sex_vals):
        print(f"  sex:  {label:<19}  {val:>+10.4f}")
    for label, val in zip(charge_labels, charge_vals):
        print(f"  charge: {label:<17}  {val:>+10.4f}")


if __name__ == "__main__":
    main()
