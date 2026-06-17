"""
Correlation diagnostic for chest X-ray BiomedCLIP concept scores — v4.

Mirror of correlation_diagnostic_v3.py pointed at the v4 (17-concept) feature
set. The pre-registered hypotheses H1-H5 were registered on v1; v4 results here
are a FOLLOW-UP check after dropping the hyperinflation concept, not a fresh
pre-registration.

BLOCKS are unchanged from v2/v3: normal_anchor_block has 4 members;
consolidation_block and interstitial_block are unchanged. hyperinflation was an
ungrouped concept, so dropping it only shrinks the ungrouped list (5 members).

Operates on train pool only (test indices never touched).
v1/v2/v3 artefacts are read-only; this script writes only to the _v4 output dir.

Outputs:
    results/chestxray/correlation_diagnostic_v4/correlation_matrix.npy
    results/chestxray/correlation_diagnostic_v4/correlation_heatmap.png
    results/chestxray/correlation_diagnostic_v4/summary.txt

Run from project root:
    python scripts/chestxray/correlation_diagnostic_v4.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # robust on cp1252 consoles
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
SCORES_NPZ  = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v4.npz"
OUTER_SPLIT = _ROOT / "data/splits/chestxray_outer_split.npz"
OUT_DIR     = _ROOT / "results/chestxray/correlation_diagnostic_v4"

# ── Pre-registered block structure (unchanged from v2) ────────────────────────
# normal_anchor_block has 4 members (no_focal_opacity was dropped in v2).
# consolidation_block and interstitial_block are unchanged.
BLOCKS = {
    "consolidation_block": [
        "consolidation",
        "focal_opacity",
        "lobar_consolidation",
        "dense_segmental_opacity",
        "air_bronchograms",
    ],
    "interstitial_block": [
        "bilateral_interstitial_pattern",
        "peribronchial_cuffing",
        "perihilar_infiltrates",
    ],
    "normal_anchor_block": [
        "clear_lung_fields",
        "sharp_costophrenic_angles",
        "normal_cardiac_silhouette",
        "symmetric_lung_aeration",
    ],
}

# Concepts not assigned to any block
UNGROUPED = [
    "pleural_effusion",
    "silhouette_sign",
    "patchy_infiltrate",
    "parapneumonic_effusion",
    "round_pneumonia",
]

# ── Pre-registered predictions (recorded before observing results) ─────────────
HYPOTHESES = {
    "H1": "Within consolidation_block, mean |r| > 0.7",
    "H2": "Within interstitial_block, mean |r| > 0.7",
    "H3": "Between consolidation_block and interstitial_block, mean |r| < within-block mean |r| of either block",
    "H4": "normal_anchor_block correlates negatively (mean signed r < 0) with both pneumonia-feature blocks",
    "H5": "Aggregate mean |r| over all 136 pairs falls in [0.3, 0.6], lower than HAM10000's 0.475",
}


def _block_indices(concept_names: list[str], block: list[str]) -> list[int]:
    idx = []
    for c in block:
        try:
            idx.append(concept_names.index(c))
        except ValueError:
            raise ValueError(f"Concept '{c}' not found in concept_names: {concept_names}")
    return idx


def _within_block_mean_abs_r(corr: np.ndarray, indices: list[int]) -> float:
    """Mean |r| over all unordered off-diagonal pairs within a block."""
    vals = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            vals.append(abs(corr[indices[i], indices[j]]))
    return float(np.mean(vals)) if vals else float("nan")


def _between_block_mean_signed_r(
    corr: np.ndarray,
    idx_a: list[int],
    idx_b: list[int],
) -> float:
    """Mean signed r for all pairs (i, j) with i in block_a and j in block_b."""
    vals = [corr[i, j] for i in idx_a for j in idx_b]
    return float(np.mean(vals)) if vals else float("nan")


def _between_block_mean_abs_r(
    corr: np.ndarray,
    idx_a: list[int],
    idx_b: list[int],
) -> float:
    vals = [abs(corr[i, j]) for i in idx_a for j in idx_b]
    return float(np.mean(vals)) if vals else float("nan")


def compute_correlation_matrix(scores: np.ndarray) -> np.ndarray:
    """Compute the K×K Pearson correlation matrix from z-scored features."""
    # Z-score (fit on train pool only — caller already subsets)
    mean = scores.mean(axis=0, keepdims=True)
    std  = scores.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-9, 1.0, std)
    z    = (scores - mean) / std
    corr = np.corrcoef(z.T)  # (K, K)
    return corr


def plot_heatmap(
    corr: np.ndarray,
    concept_names: list[str],
    out_path: pathlib.Path,
) -> None:
    n = len(concept_names)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(concept_names, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            color = "black" if abs(val) < 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    ax.set_title("Chest X-ray BiomedCLIP Concept Correlation Matrix — v4 (train pool)",
                 fontsize=11, pad=12)
    i_idx, j_idx = np.triu_indices(n, k=1)
    mean_abs_r = float(np.mean(np.abs(corr[i_idx, j_idx])))
    fig.text(
        0.5, 0.01,
        f"Mean |r| off-diagonal: {mean_abs_r:.3f}.",
        ha="center", fontsize=9, color="#444444",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved → {out_path.relative_to(_ROOT)}")


def build_summary(
    corr: np.ndarray,
    concept_names: list[str],
) -> str:
    n = len(concept_names)
    lines: list[str] = []

    # All off-diagonal pairs
    off_diag_vals = []
    off_diag_pairs: list[tuple[float, str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = abs(corr[i, j])
            off_diag_vals.append(v)
            off_diag_pairs.append((v, concept_names[i], concept_names[j]))

    off_diag_vals = np.array(off_diag_vals)
    mean_abs_r  = float(off_diag_vals.mean())
    median_abs_r = float(np.median(off_diag_vals))
    n_07 = int((off_diag_vals > 0.7).sum())
    n_08 = int((off_diag_vals > 0.8).sum())
    n_09 = int((off_diag_vals > 0.9).sum())

    lines.append("=" * 70)
    lines.append("CHEST X-RAY CONCEPT CORRELATION DIAGNOSTIC — v4")
    lines.append("(follow-up after dropping hyperinflation; H1-H5 pre-registered on v1)")
    lines.append("=" * 70)
    lines.append(f"Concepts: {n}  |  Off-diagonal pairs: {len(off_diag_vals)}")
    lines.append(f"Mean |r|   : {mean_abs_r:.4f}")
    lines.append(f"Median |r| : {median_abs_r:.4f}")
    lines.append(f"Pairs with |r| > 0.7: {n_07}")
    lines.append(f"Pairs with |r| > 0.8: {n_08}")
    lines.append(f"Pairs with |r| > 0.9: {n_09}")
    lines.append("")

    lines.append("Top-10 highest |r| pairs:")
    top10 = sorted(off_diag_pairs, reverse=True)[:10]
    for rank, (v, a, b) in enumerate(top10, 1):
        lines.append(f"  {rank:2d}. {a} × {b}  |r|={v:.4f}")
    lines.append("")

    # Block summaries
    lines.append("Block structure (pre-registered):")
    block_within: dict[str, float] = {}
    for block_name, members in BLOCKS.items():
        idx = _block_indices(concept_names, members)
        w   = _within_block_mean_abs_r(corr, idx)
        block_within[block_name] = w
        lines.append(f"  {block_name}")
        lines.append(f"    Members: {members}")
        lines.append(f"    Within-block mean |r|: {w:.4f}")

    lines.append("")
    lines.append("Between-block correlations:")
    block_names = list(BLOCKS.keys())
    block_indices_map = {
        bname: _block_indices(concept_names, members)
        for bname, members in BLOCKS.items()
    }
    for i in range(len(block_names)):
        for j in range(i + 1, len(block_names)):
            a, b = block_names[i], block_names[j]
            signed = _between_block_mean_signed_r(
                corr, block_indices_map[a], block_indices_map[b]
            )
            abs_r  = abs(signed)
            lines.append(f"  {a} × {b}")
            lines.append(f"    Mean signed r: {signed:.4f}   Mean |r|: {abs_r:.4f}")
    lines.append("")

    # Ungrouped concepts
    lines.append("Ungrouped concepts (mean |r| to all others):")
    all_idx = list(range(n))
    for uc in UNGROUPED:
        try:
            ui = concept_names.index(uc)
        except ValueError:
            lines.append(f"  {uc}: NOT IN CONCEPT SET")
            continue
        others = [k for k in all_idx if k != ui]
        mean_to_others = float(np.mean([abs(corr[ui, k]) for k in others]))
        lines.append(f"  {uc}: mean |r| to others = {mean_to_others:.4f}")
    lines.append("")

    # Hypothesis evaluation
    lines.append("=" * 70)
    lines.append("Pre-registered predictions (registered on v1; v4 = follow-up check):")
    lines.append("=" * 70)

    def pass_fail(condition: bool) -> str:
        return "PASS" if condition else "FAIL"

    # H1
    w_con = block_within["consolidation_block"]
    h1 = w_con > 0.7
    lines.append(f"H1: {HYPOTHESES['H1']}")
    lines.append(f"    Observed within-block mean |r| = {w_con:.4f}  → {pass_fail(h1)}")

    # H2
    w_int = block_within["interstitial_block"]
    h2 = w_int > 0.7
    lines.append(f"H2: {HYPOTHESES['H2']}")
    lines.append(f"    Observed within-block mean |r| = {w_int:.4f}  → {pass_fail(h2)}")

    # H3: between-block |r| < within-block |r| of either block
    con_idx = block_indices_map["consolidation_block"]
    int_idx = block_indices_map["interstitial_block"]
    between_con_int = _between_block_mean_abs_r(corr, con_idx, int_idx)
    h3 = between_con_int < w_con and between_con_int < w_int
    lines.append(f"H3: {HYPOTHESES['H3']}")
    lines.append(
        f"    Between-block mean |r| = {between_con_int:.4f}  "
        f"vs within-con={w_con:.4f}, within-int={w_int:.4f}  → {pass_fail(h3)}"
    )

    # H4: normal_anchor_block correlates negatively with both pneumonia blocks
    anc_idx = block_indices_map["normal_anchor_block"]
    anc_con = _between_block_mean_signed_r(corr, anc_idx, con_idx)
    anc_int = _between_block_mean_signed_r(corr, anc_idx, int_idx)
    h4 = anc_con < 0 and anc_int < 0
    lines.append(f"H4: {HYPOTHESES['H4']}")
    lines.append(
        f"    anchor×consolidation mean r = {anc_con:.4f}  "
        f"anchor×interstitial mean r = {anc_int:.4f}  → {pass_fail(h4)}"
    )

    # H5: aggregate mean |r| in [0.3, 0.6]
    h5 = 0.3 <= mean_abs_r <= 0.6
    lines.append(f"H5: {HYPOTHESES['H5']}")
    lines.append(f"    Observed mean |r| = {mean_abs_r:.4f}  → {pass_fail(h5)}")

    pass_count = sum([h1, h2, h3, h4, h5])
    lines.append("")
    lines.append(f"Hypothesis summary: {pass_count}/5 PASS")

    return "\n".join(lines)


def main() -> None:
    # Load data
    print(f"Loading scores from {SCORES_NPZ.relative_to(_ROOT)} ...")
    data = np.load(SCORES_NPZ, allow_pickle=True)
    scores_all    = data["scores"]               # (N, 18)
    concept_names = data["concept_names"].tolist()
    image_paths   = data["image_paths"]

    print(f"Loading outer split from {OUTER_SPLIT.relative_to(_ROOT)} ...")
    split = np.load(OUTER_SPLIT, allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]

    # Restrict to train pool only
    scores_train = scores_all[train_pool_idx]    # (N_train, 18)
    print(f"  Train pool: {len(train_pool_idx)} images  |  Concepts: {len(concept_names)}")

    # Drop any rows with NaN (failed images during encoding)
    valid_mask = ~np.isnan(scores_train).any(axis=1)
    n_dropped  = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"  Dropping {n_dropped} rows with NaN scores (encoding failures)")
    scores_clean = scores_train[valid_mask]

    # Compute correlation
    print("Computing Pearson correlation matrix...")
    corr = compute_correlation_matrix(scores_clean)

    # Basic integrity checks
    K = len(concept_names)
    assert corr.shape == (K, K), f"Unexpected corr shape: {corr.shape} (expected {K}×{K})"
    assert np.allclose(np.diag(corr), 1.0, atol=1e-5), "Diagonal != 1.0"
    assert np.allclose(corr, corr.T, atol=1e-6),        "Correlation matrix not symmetric"
    assert corr.min() >= -1.0 - 1e-6 and corr.max() <= 1.0 + 1e-6, \
        f"Correlation out of [-1, 1]"
    print("  Correlation matrix integrity checks passed ✓")

    # Verify all block/ungrouped concepts are in the concept set
    all_expected = (
        [c for members in BLOCKS.values() for c in members]
        + UNGROUPED
    )
    missing = [c for c in all_expected if c not in concept_names]
    if missing:
        print(f"  WARNING: concepts in BLOCKS/UNGROUPED but not in scores: {missing}")

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corr_path = OUT_DIR / "correlation_matrix.npy"
    np.save(corr_path, corr)
    print(f"  Saved correlation matrix → {corr_path.relative_to(_ROOT)}")

    heatmap_path = OUT_DIR / "correlation_heatmap.png"
    plot_heatmap(corr, concept_names, heatmap_path)

    summary_text = build_summary(corr, concept_names)
    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"  Saved summary           → {summary_path.relative_to(_ROOT)}")

    # Print to stdout as well
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
