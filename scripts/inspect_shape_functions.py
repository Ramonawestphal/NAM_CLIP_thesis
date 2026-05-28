"""
Shape function inspection for the base NAM (BiomedCLIP v5 features).

Loads the best seed's checkpoint (highest test balanced accuracy from
aggregated_metrics.csv) and visualises what the NAM learned per concept.

Template averaging rationale
─────────────────────────────
Each of the 24 concepts has 3 prompt templates, yielding 72 sub-networks.
We average the three sub-network outputs at each grid point rather than
plotting 72 separate shape functions.  This reduces prompt-specific noise
and focuses interpretation on the underlying concept — the signal we care
about — while still exposing the full per-template data in the CSV for
any downstream analysis.

Grid construction
──────────────────
For each concept, the grid spans [p1, p99] of the pooled (across 3 templates)
standardised training scores.  Using the pooled range avoids artificially
clipping any template that has a wider empirical support.

Outputs
────────
  reports/nam/base/shape_functions/
    shape_{concept_id}.png         — per-concept averaged shape function
    shape_functions_summary.png    — 4×6 grid of all 24 concepts
    shape_function_data.csv        — raw data: concept_id, template, x_value, class, contribution
    influence_scores.csv           — per-concept range and curvature scores + top-10 lists

Run from project root after train_nam_base.py has completed:
    python scripts/inspect_shape_functions.py
"""

from __future__ import annotations

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
from src.features.prompt_loader import load_prompts

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v5.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
PROMPTS_PATH  = "src/features/prompts/ham10000_prompts_v5_biomedclip.txt"
MODEL_DIR     = "reports/nam/base"
OUT_DIR       = "reports/nam/base/shape_functions"

# ── Model architecture (must match train_nam_base.py) ─────────────────────────
HIDDEN_DIMS = (64, 64, 32)
DROPOUT     = 0.1
N_FEATURES  = 72
N_CLASSES   = 7
GRID_POINTS = 200   # number of input values per concept grid

# ── Class ordering (sorted, matches training encoding) ────────────────────────
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Colour palette for the 7 classes (colour-blind friendly)
CLASS_COLORS = {
    "akiec": "#E69F00",
    "bcc":   "#56B4E9",
    "bkl":   "#009E73",
    "df":    "#F0E442",
    "mel":   "#D55E00",
    "nv":    "#0072B2",
    "vasc":  "#CC79A7",
}


# ─────────────────────────────────────────────────────────────────────────────
# Load data, scaler, and best checkpoint
# ─────────────────────────────────────────────────────────────────────────────
print("Loading features...")
feat       = np.load(FEATURES_PATH, allow_pickle=True)
scores_raw = feat["scores"]          # (10015, 72)
labels     = feat["labels"]

print("Loading splits...")
split     = np.load(SPLITS_PATH)
train_idx = split["train_idx"]
X_train_raw = scores_raw[train_idx]  # (8020, 72) — used for empirical ranges

print("Loading scaler...")
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
X_train_sc = scaler.transform(X_train_raw).astype(np.float32)

print("Loading prompts...")
prompts     = load_prompts(PROMPTS_PATH)
concept_ids = prompts["concept_ids"]   # list of 24 concept name strings
assert len(concept_ids) == 24

print("Selecting best seed...")
agg = pd.read_csv(os.path.join(MODEL_DIR, "aggregated_metrics.csv"))
seed_rows  = agg[~agg["seed"].isin(["mean", "std"])].copy()
seed_rows["balanced_accuracy"] = seed_rows["balanced_accuracy"].astype(float)
best_seed = int(seed_rows.loc[seed_rows["balanced_accuracy"].idxmax(), "seed"])
print(f"  Best seed: {best_seed} "
      f"(bal_acc={seed_rows.loc[seed_rows['seed'].astype(str)==str(best_seed), 'balanced_accuracy'].values[0]:.4f})")

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
# Shape function computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_shape_fn(concept_feat_cols: list[int], grid_pts: int = GRID_POINTS):
    """Compute the averaged shape function for a set of feature columns.

    Returns:
        grid (np.ndarray): shape (grid_pts,) — x values in standardised space
        avg_fn (np.ndarray): shape (grid_pts, C) — averaged contribution across templates
        per_template (list[np.ndarray]): each (grid_pts, C) — individual template outputs
    """
    # Grid spans [p1, p99] of pooled training distribution across all templates
    pooled = X_train_sc[:, concept_feat_cols].ravel()
    lo, hi = np.percentile(pooled, [1, 99])
    grid   = np.linspace(lo, hi, grid_pts).astype(np.float32)

    per_template = []
    for col in concept_feat_cols:
        out = model.concept_contributions(grid, col)   # (grid_pts, C)
        per_template.append(out.cpu().numpy())

    avg_fn = np.stack(per_template, axis=0).mean(axis=0)  # (grid_pts, C)
    return grid, avg_fn, per_template


def curvature_score(grid: np.ndarray, fn: np.ndarray) -> float:
    """Mean squared residual from the best-fit line, summed across classes.

    Measures how nonlinear the shape function is.  A purely linear shape
    function has curvature=0; large values indicate genuinely nonlinear
    learned relationships.
    """
    total = 0.0
    for c in range(fn.shape[1]):
        y = fn[:, c]
        # Fit degree-1 polynomial
        coeffs = np.polyfit(grid, y, 1)
        y_hat  = np.polyval(coeffs, grid)
        total  += float(np.mean((y - y_hat) ** 2))
    return total


def range_score(fn: np.ndarray) -> float:
    """Total range of the shape function summed across classes."""
    return float(np.sum(fn.max(axis=0) - fn.min(axis=0)))


# ─────────────────────────────────────────────────────────────────────────────
# Iterate over 24 concepts, plot, and collect CSV data
# ─────────────────────────────────────────────────────────────────────────────
csv_rows    = []
influence   = []

print("\nComputing shape functions for 24 concepts...")
for c_idx, cid in enumerate(concept_ids):
    cols   = [3 * c_idx, 3 * c_idx + 1, 3 * c_idx + 2]
    grid, avg_fn, per_tmpl = compute_shape_fn(cols)

    r_score  = range_score(avg_fn)
    k_score  = curvature_score(grid, avg_fn)
    influence.append({"concept_id": cid, "range": r_score, "curvature": k_score})

    # ── CSV: all template × grid × class rows ──
    for t_idx, tmpl_fn in enumerate(per_tmpl):
        for g_idx, x_val in enumerate(grid):
            for cls_idx, cls in enumerate(CLASS_NAMES):
                csv_rows.append({
                    "concept_id":   cid,
                    "template":     t_idx,
                    "x_value":      float(x_val),
                    "class":        cls,
                    "contribution": float(tmpl_fn[g_idx, cls_idx]),
                })

    # ── Per-concept plot (averaged across templates) ──
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for cls_idx, cls in enumerate(CLASS_NAMES):
        ax.plot(grid, avg_fn[:, cls_idx],
                label=cls, color=CLASS_COLORS[cls], linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Concept score (standardised)")
    ax.set_ylabel("Log-odds contribution")
    ax.set_title(f"{cid}\n(avg 3 templates | range={r_score:.3f} curv={k_score:.4f})")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"shape_{cid}.png"), dpi=120)
    plt.close(fig)

    if (c_idx + 1) % 6 == 0 or c_idx == 23:
        print(f"  {c_idx + 1}/24 done")

# ── Save CSV ──────────────────────────────────────────────────────────────────
sf_csv = pd.DataFrame(csv_rows)
sf_csv.to_csv(os.path.join(OUT_DIR, "shape_function_data.csv"), index=False)
print(f"\nShape function data → {OUT_DIR}/shape_function_data.csv  "
      f"({len(sf_csv):,} rows)")

# ── Save influence scores ──────────────────────────────────────────────────────
infl_df = pd.DataFrame(influence).sort_values("range", ascending=False)
infl_df.to_csv(os.path.join(OUT_DIR, "influence_scores.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Summary grid: 4×6 layout (all 24 concepts)
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting summary grid (4×6)...")
fig, axes = plt.subplots(4, 6, figsize=(22, 13))
axes_flat = axes.ravel()

for c_idx, cid in enumerate(concept_ids):
    cols = [3 * c_idx, 3 * c_idx + 1, 3 * c_idx + 2]
    grid, avg_fn, _ = compute_shape_fn(cols, grid_pts=100)  # fewer pts for speed

    ax = axes_flat[c_idx]
    for cls_idx, cls in enumerate(CLASS_NAMES):
        ax.plot(grid, avg_fn[:, cls_idx],
                color=CLASS_COLORS[cls], linewidth=0.9, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_title(cid, fontsize=7)
    ax.tick_params(labelsize=5)
    ax.set_xlabel("")
    ax.set_ylabel("")

# Shared legend outside the grid
handles = [
    plt.Line2D([0], [0], color=CLASS_COLORS[cls], linewidth=1.5, label=cls)
    for cls in CLASS_NAMES
]
fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle(
    f"NAM Base — Shape Functions (seed {best_seed}, avg 3 templates per concept)",
    fontsize=11, y=1.01,
)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "shape_functions_summary.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Summary grid → {OUT_DIR}/shape_functions_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Top-10 summary
# ─────────────────────────────────────────────────────────────────────────────
top_range = infl_df.nlargest(10, "range")
top_curv  = infl_df.nlargest(10, "curvature")

SUMMARY = f"""
==== Shape Function Analysis (NAM base, seed {best_seed}) ====

Top 10 concepts by total range (most influential):
{top_range[['concept_id','range','curvature']].to_string(index=False, float_format='%.4f')}

Top 10 concepts by curvature (most nonlinear):
{top_curv[['concept_id','curvature','range']].to_string(index=False, float_format='%.4f')}

Interpretation:
  Range    — max−min contribution summed across classes; higher = more discriminative
  Curvature — mean squared deviation from best-fit line, summed across classes;
              higher = the NAM is doing nonlinear work here beyond what LR could learn

Outputs → {OUT_DIR}/
"""
print(SUMMARY)

with open(os.path.join(OUT_DIR, "shape_analysis_summary.txt"), "w", encoding="utf-8") as f:
    f.write(SUMMARY.lstrip() + "\n")

print("Done.")
