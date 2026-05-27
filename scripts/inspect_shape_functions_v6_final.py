"""
Shape function inspection for NAM v6 Final (sweep-winning hyperparameters, 5 seeds).

Loads the best seed's checkpoint (highest test balanced accuracy) from
reports/nam/v6_final/ and visualises what the NAM learned per concept.

Architecture is read from reports/nam/v6_final/winning_config.csv so this
script does not need to hardcode the sweep winner.

Each plot shows:
  - 7 class curves: f_i(x_i) contribution to each class's log-odds
  - Rug plot along the x-axis showing empirical data density on training data
    (z-scored), so the reader can see where the bulk of the data actually lies

Outputs -> reports/nam/v6_final/shape_functions/

Run from project root after train_nam_v6_final.py has completed:
    python scripts/inspect_shape_functions_v6_final.py
"""

from __future__ import annotations

import ast
import os
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.models.nam_multiclass import NAMMulticlass

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
MODEL_DIR     = "reports/nam/v6_final_lambda1"
OUT_DIR       = "reports/nam/v6_final_lambda1/shape_functions"

# ── Fixed settings ─────────────────────────────────────────────────────────────
N_FEATURES  = 24
N_CLASSES   = 7
GRID_POINTS = 200

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_COLORS = {
    "akiec": "#E69F00", "bcc": "#56B4E9", "bkl": "#009E73",
    "df":    "#F0E442", "mel": "#D55E00", "nv":  "#0072B2", "vasc": "#CC79A7",
}

RUG_ALPHA  = 0.15
RUG_HEIGHT = 0.04
RUG_SAMPLE = 500


# ─────────────────────────────────────────────────────────────────────────────
# Read winning architecture from saved CSV
# ─────────────────────────────────────────────────────────────────────────────
config_csv = os.path.join(MODEL_DIR, "winning_config.csv")
if not os.path.exists(config_csv):
    raise FileNotFoundError(
        f"winning_config.csv not found at {config_csv}. "
        "Run train_nam_v6_final.py first."
    )

cfg      = pd.read_csv(config_csv).iloc[0]
HIDDEN_DIMS = tuple(ast.literal_eval(cfg["hidden_dims"]))
DROPOUT     = float(cfg["dropout"])

print(f"Winning architecture: hidden={list(HIDDEN_DIMS)}, dropout={DROPOUT}")


# ─────────────────────────────────────────────────────────────────────────────
# Load data, scaler, best checkpoint
# ─────────────────────────────────────────────────────────────────────────────
print("Loading features...")
feat        = np.load(FEATURES_PATH, allow_pickle=True)
scores_raw  = feat["scores"]
concept_ids = list(feat["concept_ids"])

print("Loading splits...")
train_idx   = np.load(SPLITS_PATH)["train_idx"]
X_train_raw = scores_raw[train_idx]

print("Loading scaler...")
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
X_train_sc = scaler.transform(X_train_raw).astype(np.float32)

print("Selecting best seed...")
agg = pd.read_csv(os.path.join(MODEL_DIR, "aggregated_metrics.csv"))
seed_rows = agg[~agg["seed"].isin(["mean", "std"])].copy()
seed_rows["balanced_accuracy"] = seed_rows["balanced_accuracy"].astype(float)
best_seed   = int(seed_rows.loc[seed_rows["balanced_accuracy"].idxmax(), "seed"])
best_balacc = float(
    seed_rows.loc[seed_rows["seed"].astype(str) == str(best_seed),
                  "balanced_accuracy"].values[0]
)
print(f"  Best seed: {best_seed}  (test bal_acc={best_balacc:.4f})")

ckpt_path = os.path.join(MODEL_DIR, f"seed_{best_seed}", "best_model.pt")
model = NAMMulticlass(
    n_features=N_FEATURES, num_classes=N_CLASSES,
    hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
model.eval()
print(f"  Checkpoint loaded: {ckpt_path}")

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shape function helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_shape_fn(col: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.percentile(X_train_sc[:, col], [1, 99])
    grid   = np.linspace(lo, hi, GRID_POINTS).astype(np.float32)
    fn     = model.concept_contributions(grid, col).numpy()
    return grid, fn


def range_score(fn: np.ndarray) -> float:
    return float(np.sum(np.abs(fn.max(axis=0) - fn.min(axis=0))))


def curvature_score(grid: np.ndarray, fn: np.ndarray) -> float:
    total = 0.0
    for c in range(fn.shape[1]):
        y     = fn[:, c]
        y_hat = np.polyval(np.polyfit(grid, y, 1), grid)
        total += float(np.sqrt(np.mean((y - y_hat) ** 2)))
    return total


def add_rug(ax: plt.Axes, col_data: np.ndarray, y_min: float, y_max: float) -> None:
    rng = np.random.default_rng(0)
    if len(col_data) > RUG_SAMPLE:
        idx = rng.choice(len(col_data), RUG_SAMPLE, replace=False)
        col_data = col_data[idx]
    rug_y = y_min - RUG_HEIGHT * (y_max - y_min)
    ax.plot(col_data, np.full_like(col_data, rug_y),
            "|", color="gray", alpha=RUG_ALPHA, markersize=4, markeredgewidth=0.6)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop — compute + plot 24 shape functions
# ─────────────────────────────────────────────────────────────────────────────
csv_rows  = []
influence = []

print(f"\nComputing shape functions for {len(concept_ids)} concepts...")

for col_idx, cid in enumerate(concept_ids):
    grid, fn = compute_shape_fn(col_idx)
    r_score  = range_score(fn)
    k_score  = curvature_score(grid, fn)
    influence.append({"concept_id": cid, "range": r_score, "curvature": k_score})

    for g_idx, x_val in enumerate(grid):
        for cls_idx, cls in enumerate(CLASS_NAMES):
            csv_rows.append({
                "concept_id":   cid,
                "x_value":      float(x_val),
                "class":        cls,
                "contribution": float(fn[g_idx, cls_idx]),
            })

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for cls_idx, cls in enumerate(CLASS_NAMES):
        ax.plot(grid, fn[:, cls_idx],
                label=cls, color=CLASS_COLORS[cls], linewidth=1.6)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")

    y_min, y_max = ax.get_ylim()
    add_rug(ax, X_train_sc[:, col_idx], y_min, y_max)
    ax.set_ylim(y_min - RUG_HEIGHT * (y_max - y_min) * 1.5, y_max)

    ax.set_xlabel("Concept score (z-scored)")
    ax.set_ylabel("Log-odds contribution")
    ax.set_title(f"{cid}\n(range={r_score:.3f}  curv={k_score:.4f}  "
                 f"seed {best_seed})")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"shape_{cid}.png"), dpi=120)
    plt.close(fig)

    if (col_idx + 1) % 6 == 0 or col_idx == len(concept_ids) - 1:
        print(f"  {col_idx + 1}/{len(concept_ids)} done")


# ── Save CSVs ─────────────────────────────────────────────────────────────────
sf_csv = pd.DataFrame(csv_rows)
sf_csv.to_csv(os.path.join(OUT_DIR, "shape_function_data.csv"), index=False)
print(f"\nShape function data: {len(sf_csv):,} rows "
      f"-> {OUT_DIR}/shape_function_data.csv")

infl_df = pd.DataFrame(influence).sort_values("range", ascending=False).reset_index(drop=True)
infl_df.to_csv(os.path.join(OUT_DIR, "influence_scores.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Summary grid (4x6, all 24 concepts)
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting 4x6 summary grid...")
fig, axes = plt.subplots(4, 6, figsize=(22, 13))
axes_flat = axes.ravel()

for col_idx, cid in enumerate(concept_ids):
    grid, fn = compute_shape_fn(col_idx)
    ax = axes_flat[col_idx]
    for cls_idx, cls in enumerate(CLASS_NAMES):
        ax.plot(grid, fn[:, cls_idx],
                color=CLASS_COLORS[cls], linewidth=0.9, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    y_lo, y_hi = ax.get_ylim()
    col_data = X_train_sc[:, col_idx]
    rng = np.random.default_rng(col_idx)
    idx = rng.choice(len(col_data), min(200, len(col_data)), replace=False)
    rug_y = y_lo - 0.05 * (y_hi - y_lo)
    ax.plot(col_data[idx], np.full(len(idx), rug_y),
            "|", color="gray", alpha=0.10, markersize=2.5, markeredgewidth=0.4)
    ax.set_ylim(rug_y - 0.01 * (y_hi - y_lo), y_hi)
    ax.set_title(cid, fontsize=7)
    ax.tick_params(labelsize=5)

handles = [
    plt.Line2D([0], [0], color=CLASS_COLORS[cls], linewidth=1.5, label=cls)
    for cls in CLASS_NAMES
]
fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle(
    f"NAM v6 Final — Shape Functions (seed {best_seed}, z-scored inputs)\n"
    f"hidden={list(HIDDEN_DIMS)}, dropout={DROPOUT}",
    fontsize=11, y=1.01,
)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "shape_functions_summary.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Summary grid -> {OUT_DIR}/shape_functions_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Top-10 summary
# ─────────────────────────────────────────────────────────────────────────────
top_range = infl_df.nlargest(10, "range")
top_curv  = infl_df.nlargest(10, "curvature")

SUMMARY = f"""
==== Shape Function Analysis (NAM v6 Final, seed {best_seed}) ====

Winning architecture: hidden={list(HIDDEN_DIMS)}, dropout={DROPOUT}
Test balanced accuracy (best seed): {best_balacc:.4f}

Top 10 concepts by total range (most influential):
{top_range[['concept_id', 'range', 'curvature']].to_string(index=False, float_format='%.4f')}

Top 10 concepts by curvature (most nonlinear — NAM adds value over LR here):
{top_curv[['concept_id', 'curvature', 'range']].to_string(index=False, float_format='%.4f')}

Interpretation:
  Range     — sum over classes of |max - min| contribution across empirical support;
              higher = concept strongly differentiates classes
  Curvature — L2 distance from best-fit line, summed over classes (root-mean
              squared residual per class, then summed); higher = genuinely
              nonlinear relationship the NAM learned that a linear model cannot
              capture. Concepts here are candidates for interaction terms in
              Phase 2.

Plots include rug marks showing training-data density in z-score space.
Dense rug regions are where shape function estimates are most reliable;
sparse tails may show unstable behaviour.

Outputs -> {OUT_DIR}/
"""
print(SUMMARY)

with open(os.path.join(OUT_DIR, "shape_analysis_summary.txt"), "w", encoding="utf-8") as f:
    f.write(SUMMARY.lstrip() + "\n")

print("Done.")
