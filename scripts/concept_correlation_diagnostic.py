"""
Concept correlation diagnostic for NAM v6 thesis methodology chapter.

Computes the 24x24 Pearson correlation matrix of BiomedCLIP concept scores
on the train_final split and tests whether residual concurvity
(R_perp ≈ 0.13 at lambda=1.0) is driven by intrinsic input correlation.

Feature loading replicates scripts/train_nam_v6_final.py lines 162-215 exactly.
Run from project root:
    python scripts/concept_correlation_diagnostic.py
"""

from __future__ import annotations

import os
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

# ── Paths (identical to train_nam_v6_final.py) ────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
OUT_DIR       = "reports/nam/diagnostics"
FINAL_R_PERP  = 0.13   # R_perp_val of regularized NAM at lambda=1.0

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load + split + standardise  (mirrors train_nam_v6_final.py lines 162-215)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading features...")
feat       = np.load(FEATURES_PATH, allow_pickle=True)
scores     = feat["scores"]        # (10015, 24)
labels     = feat["labels"]
lesion_ids = feat["lesion_ids"]
concept_names = feat["concept_ids"].tolist()   # list[str], length 24

assert scores.shape == (10015, 24), f"Unexpected shape: {scores.shape}"
assert len(concept_names) == 24

print("Loading splits...")
split     = np.load(SPLITS_PATH)
train_idx = split["train_idx"]
test_idx  = split["test_idx"]

X_all_train      = scores[train_idx]
y_all_train      = labels[train_idx]
lesion_ids_train = lesion_ids[train_idx]

# Val split — random_state=42, test_size=0.20, groups=lesion_id
# (identical to train_nam_v6_final.py lines 189-202)
print("Carving validation set (GroupShuffleSplit 80/20 by lesion_id, random_state=42)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_final_rel, val_rel = next(
    gss.split(X_all_train, y_all_train, groups=lesion_ids_train)
)
X_train_raw = X_all_train[train_final_rel]   # train_final split

print(f"  train_final : {len(train_final_rel):5d} images")
print(f"  val         : {len(val_rel):5d} images")
print(f"  test        : {len(test_idx):5d} images")

print("Standardising (z-score, fit on train_final)...")
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train_raw).astype(np.float32)

N_train = X_train_sc.shape[0]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Pearson correlation matrix (24x24)
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing 24x24 Pearson correlation matrix on train_final...")
corr_matrix = np.corrcoef(X_train_sc, rowvar=False)   # shape (24, 24)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
n = 24
# Indices of upper triangle (i < j)
i_idx, j_idx = np.triu_indices(n, k=1)
off_diag_r   = corr_matrix[i_idx, j_idx]
abs_r        = np.abs(off_diag_r)

mean_abs  = float(np.mean(abs_r))
med_abs   = float(np.median(abs_r))
max_abs   = float(np.max(abs_r))
max_pair  = (concept_names[i_idx[np.argmax(abs_r)]],
             concept_names[j_idx[np.argmax(abs_r)]])

n_pairs   = len(abs_r)
n_gt50    = int(np.sum(abs_r > 0.5))
n_gt70    = int(np.sum(abs_r > 0.7))
n_gt90    = int(np.sum(abs_r > 0.9))

print("\n" + "=" * 70)
print("CONCEPT CORRELATION SUMMARY (train_final, z-scored)")
print("=" * 70)
print(f"  N train_final                       : {N_train}")
print(f"  Number of concept pairs (24 choose 2): {n_pairs}")
print(f"  Mean  |r| off-diagonal               : {mean_abs:.4f}")
print(f"  Median |r| off-diagonal              : {med_abs:.4f}")
print(f"  Max   |r| off-diagonal               : {max_abs:.4f}")
print(f"  Pair achieving max |r|               : {max_pair[0]} vs {max_pair[1]}")
print(f"  Pairs with |r| > 0.5                 : {n_gt50:3d} / {n_pairs}  ({100*n_gt50/n_pairs:.1f}%)")
print(f"  Pairs with |r| > 0.7                 : {n_gt70:3d} / {n_pairs}  ({100*n_gt70/n_pairs:.1f}%)")
print(f"  Pairs with |r| > 0.9                 : {n_gt90:3d} / {n_pairs}  ({100*n_gt90/n_pairs:.1f}%)")

# Top 10 most correlated pairs
sort_order = np.argsort(-abs_r)
print("\nTop 10 most correlated pairs (by |r|):")
print(f"  {'Rank':<5} {'Concept A':<35} {'Concept B':<35} {'r':>8}")
print("  " + "-" * 83)
for rank, k in enumerate(sort_order[:10], 1):
    ci = concept_names[i_idx[k]]
    cj = concept_names[j_idx[k]]
    r  = off_diag_r[k]
    print(f"  {rank:<5} {ci:<35} {cj:<35} {r:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Secondary analysis: flagged concept pairs
# ─────────────────────────────────────────────────────────────────────────────
def pair_r(name_a: str, name_b: str) -> float:
    ia = concept_names.index(name_a)
    ib = concept_names.index(name_b)
    return float(corr_matrix[ia, ib])

flagged_pairs = [
    ("asymmetry",            "colour_variation"),
    ("asymmetry",            "irregular_pigmentation"),
    ("asymmetry",            "diameter_large"),
    ("colour_variation",     "irregular_pigmentation"),
    ("diameter_large",       "atypical_pigment_network"),
    ("border_irregularity",  "irregular_pigmentation"),   # high-impact pair from shape analysis
    ("asymmetry",            "border_irregularity"),      # canonical ABCD pair
]

print("\nFlagged concept pairs (input correlations for thesis text):")
print(f"  {'Concept A':<35} {'Concept B':<35} {'r':>8}")
print("  " + "-" * 78)
for ca, cb in flagged_pairs:
    try:
        r = pair_r(ca, cb)
        print(f"  {ca:<35} {cb:<35} {r:+.4f}")
    except ValueError as exc:
        print(f"  WARNING: {exc}")

# Also report any top-10 pair not already in flagged list
top10_pairs = set()
for k in sort_order[:10]:
    top10_pairs.add((concept_names[i_idx[k]], concept_names[j_idx[k]]))
flagged_set = {(a, b) for a, b in flagged_pairs}
extras = top10_pairs - flagged_set
if extras:
    print("\n  Additional top-10 pairs not in flagged list:")
    for ca, cb in sorted(extras):
        try:
            r = pair_r(ca, cb)
        except ValueError:
            r = corr_matrix[concept_names.index(ca), concept_names.index(cb)]
        print(f"  {ca:<35} {cb:<35} {r:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Heatmap figure
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 12))

im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r", fontsize=12)

# Tick labels
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(concept_names, fontsize=8)

# Cell annotations where |r| > 0.3
for row in range(n):
    for col in range(n):
        r_val = corr_matrix[row, col]
        if abs(r_val) > 0.3:
            text_color = "white" if abs(r_val) > 0.65 else "black"
            ax.text(col, row, f"{r_val:.2f}", ha="center", va="center",
                    fontsize=5.5, color=text_color, fontweight="normal")

ax.set_title(
    f"Pairwise correlation of 24 BiomedCLIP concept scores "
    f"(train split, N={N_train})",
    fontsize=13, pad=14
)

plt.tight_layout()

png_path = os.path.join(OUT_DIR, "concept_correlation.png")
pdf_path = os.path.join(OUT_DIR, "concept_correlation.pdf")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)
print(f"\nHeatmap saved to:\n  {png_path}\n  {pdf_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save correlation matrix as CSV
# ─────────────────────────────────────────────────────────────────────────────
df_corr = pd.DataFrame(corr_matrix, index=concept_names, columns=concept_names)
csv_path = os.path.join(OUT_DIR, "concept_correlation.csv")
df_corr.round(6).to_csv(csv_path)
print(f"Correlation matrix CSV saved to:\n  {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Comparison line
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"Mean off-diagonal |r| of input concept scores: {mean_abs:.3f}")
print(f"Final R_perp_val of regularized NAM (lambda=1.0): {FINAL_R_PERP:.2f}")
if mean_abs > FINAL_R_PERP * 1.5:
    interp = (
        f"X.XXX ({mean_abs:.3f}) >> 0.13: the regularizer has substantially reduced "
        "output concurvity below the level of input correlation, but cannot drop it "
        "further without distorting shape functions."
    )
else:
    interp = (
        f"X.XXX ({mean_abs:.3f}) ≈ 0.13: the regularizer has tracked input "
        "correlation closely."
    )
print(f"Interpretation: {interp}")
print("=" * 70)
