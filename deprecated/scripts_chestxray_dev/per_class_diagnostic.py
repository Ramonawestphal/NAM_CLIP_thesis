"""
Per-class concept score diagnostic for chest X-ray BiomedCLIP features.

Investigates the H4 anomaly from the correlation diagnostic: the
normal_anchor_block correlated positively with the interstitial_block
(mean r = +0.50), contrary to the pre-registered prediction of negative
correlation.

Two competing explanations are evaluated:
  Explanation A: semantic-axis confound — BiomedCLIP encodes shared
      anatomic-location vocabulary as a shared latent axis across images,
      irrespective of the normal/abnormal distinction the prompts encode.
  Explanation B: prompt discrimination failure — the prompts themselves
      fail to activate on their intended target class, and cross-image
      variance is dominated by exposure/view/framing.

Operates on train pool only. Test indices are never touched.

Outputs:
    results/chestxray/per_class_diagnostic/per_class_means.csv
    results/chestxray/per_class_diagnostic/per_class_heatmap.png
    results/chestxray/per_class_diagnostic/discrimination_summary.txt
    results/chestxray/per_class_diagnostic/h4_investigation.txt

Run from project root:
    python scripts/chestxray/per_class_diagnostic.py
"""

from __future__ import annotations

import csv
import pathlib
import sys
from typing import Dict, List, Tuple

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
SCORES_NPZ  = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v1.npz"
OUTER_SPLIT = _ROOT / "data/splits/chestxray_outer_split.npz"
OUT_DIR     = _ROOT / "results/chestxray/per_class_diagnostic"

# ── Pre-registered block membership (mirrors correlation_diagnostic.py) ───────
NORMAL_ANCHOR_BLOCK = [
    "clear_lung_fields",
    "sharp_costophrenic_angles",
    "normal_cardiac_silhouette",
    "no_focal_opacity",
    "symmetric_lung_aeration",
]
INTERSTITIAL_BLOCK = [
    "bilateral_interstitial_pattern",
    "peribronchial_cuffing",
    "perihilar_infiltrates",
]

# ── Expected target classes per concept ──────────────────────────────────────
EXPECTED_TARGET: Dict[str, List[str]] = {
    # Tier 1: general pneumonia signs (higher than normal on either subtype)
    "consolidation":                    ["bacteria", "virus"],
    "focal_opacity":                    ["bacteria", "virus"],
    "air_bronchograms":                 ["bacteria", "virus"],
    "pleural_effusion":                 ["bacteria", "virus"],
    "silhouette_sign":                  ["bacteria", "virus"],
    "patchy_infiltrate":                ["bacteria", "virus"],
    # Tier 2A: bacterial-discriminative
    "lobar_consolidation":              ["bacteria"],
    "dense_segmental_opacity":          ["bacteria"],
    "parapneumonic_effusion":           ["bacteria"],
    "round_pneumonia":                  ["bacteria"],
    # Tier 2B: viral-discriminative
    "bilateral_interstitial_pattern":   ["virus"],
    "peribronchial_cuffing":            ["virus"],
    "perihilar_infiltrates":            ["virus"],
    "hyperinflation":                   ["virus"],
    # Tier 3: normal anchor
    "clear_lung_fields":                ["normal"],
    "sharp_costophrenic_angles":        ["normal"],
    "normal_cardiac_silhouette":        ["normal"],
    "no_focal_opacity":                 ["normal"],
    "symmetric_lung_aeration":          ["normal"],
}

TIER_LABELS: Dict[str, str] = {
    "consolidation":                    "T1-general",
    "focal_opacity":                    "T1-general",
    "air_bronchograms":                 "T1-general",
    "pleural_effusion":                 "T1-general",
    "silhouette_sign":                  "T1-general",
    "patchy_infiltrate":                "T1-general",
    "lobar_consolidation":              "T2A-bacteria",
    "dense_segmental_opacity":          "T2A-bacteria",
    "parapneumonic_effusion":           "T2A-bacteria",
    "round_pneumonia":                  "T2A-bacteria",
    "bilateral_interstitial_pattern":   "T2B-virus",
    "peribronchial_cuffing":            "T2B-virus",
    "perihilar_infiltrates":            "T2B-virus",
    "hyperinflation":                   "T2B-virus",
    "clear_lung_fields":                "T3-normal",
    "sharp_costophrenic_angles":        "T3-normal",
    "normal_cardiac_silhouette":        "T3-normal",
    "no_focal_opacity":                 "T3-normal",
    "symmetric_lung_aeration":          "T3-normal",
}

CLASSES = ["normal", "bacteria", "virus"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_per_class_stats(
    scores: np.ndarray,
    concept_names: List[str],
    labels_subtype: np.ndarray,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return nested dict: concept → class → {mean, std, n}."""
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i, cname in enumerate(concept_names):
        result[cname] = {}
        for cls in CLASSES:
            mask = labels_subtype == cls
            vals = scores[mask, i]
            vals = vals[~np.isnan(vals)]
            result[cname][cls] = {
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std":  float(vals.std())  if len(vals) else float("nan"),
                "n":    int(len(vals)),
            }
    return result


def write_csv(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    concept_names: List[str],
    path: pathlib.Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "class", "mean", "std", "n"])
        for cname in concept_names:
            for cls in CLASSES:
                s = stats[cname][cls]
                writer.writerow([
                    cname, cls,
                    f"{s['mean']:.6f}", f"{s['std']:.6f}", s["n"],
                ])
    print(f"  CSV saved → {path.relative_to(_ROOT)}")


def plot_heatmaps(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    concept_names: List[str],
    path: pathlib.Path,
) -> None:
    n_concepts = len(concept_names)
    n_classes  = len(CLASSES)

    # Build mean matrix (n_concepts, n_classes)
    means = np.array([
        [stats[c][cls]["mean"] for cls in CLASSES]
        for c in concept_names
    ])

    # Z-score across the 3 class means (per concept)
    row_mean = means.mean(axis=1, keepdims=True)
    row_std  = means.std(axis=1, keepdims=True)
    row_std  = np.where(row_std < 1e-9, 1.0, row_std)
    z_means  = (means - row_mean) / row_std

    fig, axes = plt.subplots(
        2, 1,
        figsize=(6, max(10, n_concepts * 0.55 * 2)),
        gridspec_kw={"hspace": 0.35},
    )

    for ax, data, cmap, title, vrange, fmt in [
        (axes[0], means,   "viridis",  "Absolute mean cosine similarity",
         (means.min(), means.max()), "{:.3f}"),
        (axes[1], z_means, "RdBu_r",   "Z-scored means (per-concept across classes)",
         (-2.5, 2.5), "{:.2f}"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap=cmap,
                       vmin=vrange[0], vmax=vrange[1])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(CLASSES, fontsize=9)
        ax.set_yticks(range(n_concepts))
        ax.set_yticklabels(concept_names, fontsize=7)
        ax.set_title(title, fontsize=9, pad=8)
        for r in range(n_concepts):
            for c in range(n_classes):
                val = data[r, c]
                contrast = "white" if abs(val) > (vrange[1] * 0.6) else "black"
                ax.text(c, r, fmt.format(val), ha="center", va="center",
                        fontsize=6, color=contrast)

    fig.suptitle(
        "Chest X-ray BiomedCLIP: per-class concept scores (train pool)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved → {path.relative_to(_ROOT)}")


def build_discrimination_summary(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    concept_names: List[str],
) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("PER-CONCEPT DISCRIMINATION SUMMARY (train pool)")
    lines.append("=" * 70)

    tier_match:    Dict[str, List[bool]] = {}
    tier_total:    Dict[str, int]        = {}
    match_count    = 0

    for cname in concept_names:
        cls_means = {cls: stats[cname][cls]["mean"] for cls in CLASSES}
        argmax = max(cls_means, key=lambda c: cls_means[c])
        argmin = min(cls_means, key=lambda c: cls_means[c])
        spread = cls_means[argmax] - cls_means[argmin]
        expected = EXPECTED_TARGET.get(cname, [])
        tier     = TIER_LABELS.get(cname, "unknown")

        is_match = argmax in expected
        match_count += int(is_match)
        verdict_str = "MATCH" if is_match else "MISMATCH"

        tier_match.setdefault(tier, []).append(is_match)
        tier_total[tier] = tier_total.get(tier, 0) + 1

        lines.append(
            f"{cname:<40s}  argmax={argmax:<10s}  expected={str(expected):<22s}"
            f"  spread={spread:.4f}  [{tier}]  {verdict_str}"
        )
        if not is_match:
            lines.append(
                f"    MISMATCH details: "
                + "  ".join(f"{c}={cls_means[c]:.4f}" for c in CLASSES)
            )

    lines.append("")
    lines.append(f"Concepts matching expected target: {match_count} / {len(concept_names)}")
    lines.append("")
    lines.append("Breakdown by tier:")
    for tier in sorted(tier_match):
        matches = sum(tier_match[tier])
        total   = len(tier_match[tier])
        lines.append(f"  {tier:<18s}: {matches}/{total}")

    return "\n".join(lines)


def build_h4_investigation(
    scores_train: np.ndarray,
    concept_names: List[str],
    labels_subtype: np.ndarray,
    stats: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("H4 INVESTIGATION: normal_anchor_block × interstitial_block")
    lines.append("=" * 70)
    lines.append(
        "H4 predicted: normal_anchor_block correlates negatively with both "
        "pneumonia-feature blocks.\n"
        "Observed: mean signed r = +0.50 (interstitial), anomalous positive.\n"
        "Two candidate explanations:\n"
        "  A: semantic-axis confound (shared latent axis across images)\n"
        "  B: prompt discrimination failure\n"
    )

    anc_idx  = [concept_names.index(c) for c in NORMAL_ANCHOR_BLOCK]
    int_idx  = [concept_names.index(c) for c in INTERSTITIAL_BLOCK]

    # ── Step 1 & 2: per-class means and discrimination check ──────────────────
    lines.append("─" * 60)
    lines.append("STEP 1-2: Per-class means and discrimination check")
    lines.append("─" * 60)

    lines.append("\nnormal_anchor_block (expected: normal > virus for all 4):")
    anchor_all_correct = True
    for cname in NORMAL_ANCHOR_BLOCK:
        m_normal = stats[cname]["normal"]["mean"]
        m_virus  = stats[cname]["virus"]["mean"]
        m_bact   = stats[cname]["bacteria"]["mean"]
        sign_ok  = m_normal > m_virus
        anchor_all_correct = anchor_all_correct and sign_ok
        direction = "normal > virus ✓" if sign_ok else "normal < virus ✗"
        lines.append(
            f"  {cname:<40s}  normal={m_normal:.4f}  virus={m_virus:.4f}"
            f"  bacteria={m_bact:.4f}  → {direction}"
        )

    lines.append("\ninterstitial_block (expected: virus > normal for all 3):")
    interstitial_all_correct = True
    for cname in INTERSTITIAL_BLOCK:
        m_normal = stats[cname]["normal"]["mean"]
        m_virus  = stats[cname]["virus"]["mean"]
        m_bact   = stats[cname]["bacteria"]["mean"]
        sign_ok  = m_virus > m_normal
        interstitial_all_correct = interstitial_all_correct and sign_ok
        direction = "virus > normal ✓" if sign_ok else "virus < normal ✗"
        lines.append(
            f"  {cname:<40s}  normal={m_normal:.4f}  virus={m_virus:.4f}"
            f"  bacteria={m_bact:.4f}  → {direction}"
        )

    # Decision from Step 2
    prompts_discriminate = anchor_all_correct and interstitial_all_correct
    lines.append(
        f"\nStep 2 decision: anchor all correct={anchor_all_correct}, "
        f"interstitial all correct={interstitial_all_correct}"
    )
    lines.append(
        "→ Prompts discriminate per-class correctly: "
        + ("YES" if prompts_discriminate else "NO")
    )

    # ── Step 3: cross-image correlations ──────────────────────────────────────
    lines.append("")
    lines.append("─" * 60)
    lines.append("STEP 3: Cross-image correlations (Explanation A confirmation)")
    lines.append("─" * 60)

    # Full train-pool 4×3 mean |r|
    full_cross_vals: List[float] = []
    for ai in anc_idx:
        for ii in int_idx:
            full_cross_vals.append(float(np.corrcoef(
                scores_train[:, ai], scores_train[:, ii]
            )[0, 1]))
    full_cross_mean = float(np.mean(np.abs(full_cross_vals)))
    lines.append(f"\nFull train-pool mean |r| (anchor × interstitial, 12 pairs): "
                 f"{full_cross_mean:.4f}")

    # Within-normal-class 4×3 correlations
    normal_mask = labels_subtype == "normal"
    scores_normal = scores_train[normal_mask]

    lines.append(f"\nWithin-normal-class correlations (n={normal_mask.sum()} images):")
    lines.append(
        f"{'':>42s}" + "".join(f"{c:<28s}" for c in INTERSTITIAL_BLOCK)
    )
    within_vals: List[float] = []
    for cname_a in NORMAL_ANCHOR_BLOCK:
        ai = concept_names.index(cname_a)
        row_str = f"  {cname_a:<40s}"
        for cname_b in INTERSTITIAL_BLOCK:
            ii = concept_names.index(cname_b)
            r  = float(np.corrcoef(scores_normal[:, ai], scores_normal[:, ii])[0, 1])
            within_vals.append(r)
            row_str += f"  {r:+.4f}              "
        lines.append(row_str)

    within_mean_abs = float(np.mean(np.abs(within_vals)))
    within_mean_signed = float(np.mean(within_vals))
    lines.append(
        f"\nWithin-normal mean |r| over 12 pairs: {within_mean_abs:.4f}"
    )
    lines.append(
        f"Within-normal mean signed r:          {within_mean_signed:+.4f}"
    )
    within_high = within_mean_abs > 0.4
    lines.append(
        f"Threshold (> 0.4 supports Explanation A): "
        + ("YES — within-class correlation is substantial" if within_high
           else "NO — within-class correlation is low")
    )

    # ── Step 4: Verdict ───────────────────────────────────────────────────────
    lines.append("")
    lines.append("─" * 60)
    lines.append("STEP 4: VERDICT")
    lines.append("─" * 60)
    lines.append("")

    if prompts_discriminate and within_high:
        verdict_label = "VERDICT: Explanation A (semantic-axis confound)"
        verdict_body = (
            "All 4 normal_anchor_block concepts score higher on the normal class "
            "than on the viral class, and all 3 interstitial_block concepts score "
            f"higher on viral than on normal — confirming that per-class discrimination "
            f"is working as intended (Step 2). However, the high within-normal-class "
            f"cross-block correlation ({within_mean_abs:.3f} mean |r|, Step 3) shows "
            "that both blocks co-vary strongly even when the class label is held constant. "
            "This is consistent with BiomedCLIP encoding shared anatomic-location "
            "vocabulary ('bilateral', 'symmetric', 'perihilar', 'normal cardiac') as a "
            "common semantic axis — images with larger/better-aerated lungs score higher "
            "on both blocks regardless of pathology. The H4 anomaly is therefore a property "
            "of the BiomedCLIP feature space, not a failure of the prompts. "
            "Implication: prompts are adequate for NAM training; the high cross-block "
            "correlation will appear as a sparsity challenge (the model must learn to "
            "down-weight the common axis), not as a semantic mismatch."
        )
    elif not prompts_discriminate and not within_high:
        verdict_label = "VERDICT: Explanation B (prompt discrimination failure)"
        verdict_body = (
            "One or more prompts in the normal_anchor_block or interstitial_block "
            "fail to discriminate their intended target class: the expected per-class "
            "ordering (normal > virus for anchor; virus > normal for interstitial) does "
            "not hold universally (Step 2). The within-normal-class cross-block correlation "
            f"is low ({within_mean_abs:.3f} mean |r|), suggesting the co-variance in the "
            "full train pool is driven by class composition rather than a shared latent "
            "axis. This is analogous to the HAM10000 v2 prompt leak. "
            "Implication: targeted prompt revision is needed for the failing concepts "
            "before proceeding to NAM training."
        )
    else:
        verdict_label = "VERDICT: Mixed / inconclusive"
        verdict_body = (
            f"The evidence is mixed. Per-class discrimination is "
            f"{'correct for all 7 concepts' if prompts_discriminate else 'incorrect for at least one concept'} "
            f"(Step 2), and the within-normal-class cross-block correlation is "
            f"{within_mean_abs:.3f} (Step 3), which "
            f"{'exceeds' if within_high else 'does not exceed'} the 0.4 threshold for "
            "Explanation A. Some prompts may discriminate correctly while others "
            "co-vary through a shared latent axis without correct per-class ordering. "
            "Implication: inspect the MISMATCH entries in discrimination_summary.txt "
            "and revise only the failing concepts before proceeding."
        )

    lines.append(verdict_label)
    lines.append("")
    # Wrap at ~80 chars for readability
    words = verdict_body.split()
    current_line: List[str] = []
    for word in words:
        if sum(len(w) + 1 for w in current_line) + len(word) > 78:
            lines.append("  " + " ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append("  " + " ".join(current_line))

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load artefacts
    print(f"Loading scores from {SCORES_NPZ.relative_to(_ROOT)} ...")
    data = np.load(SCORES_NPZ, allow_pickle=True)
    scores_all    = data["scores"]                    # (N, 19)
    concept_names = data["concept_names"].tolist()

    print(f"Loading outer split from {OUTER_SPLIT.relative_to(_ROOT)} ...")
    split          = np.load(OUTER_SPLIT, allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]
    labels_sub_all = split["labels_subtype"]

    # Restrict to train pool
    scores_train  = scores_all[train_pool_idx]        # (N_train, 19)
    labels_train  = labels_sub_all[train_pool_idx]

    # Drop NaN rows
    valid_mask = ~np.isnan(scores_train).any(axis=1)
    n_dropped  = (~valid_mask).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped} NaN rows from train pool")
    scores_clean = scores_train[valid_mask]
    labels_clean = labels_train[valid_mask]

    print(f"  Train pool: {len(scores_clean)} images  |  Concepts: {len(concept_names)}")
    for cls in CLASSES:
        print(f"    {cls}: {(labels_clean == cls).sum()} images")

    # 1. Per-class stats
    print("\nComputing per-class statistics...")
    stats = compute_per_class_stats(scores_clean, concept_names, labels_clean)

    # 2. CSV
    write_csv(stats, concept_names, OUT_DIR / "per_class_means.csv")

    # 3. Heatmap
    plot_heatmaps(stats, concept_names, OUT_DIR / "per_class_heatmap.png")

    # 4. Discrimination summary
    disc_text = build_discrimination_summary(stats, concept_names)
    disc_path = OUT_DIR / "discrimination_summary.txt"
    disc_path.write_text(disc_text, encoding="utf-8")
    print(f"  Discrimination summary → {disc_path.relative_to(_ROOT)}")

    # 5. H4 investigation
    h4_text  = build_h4_investigation(scores_clean, concept_names, labels_clean, stats)
    h4_path  = OUT_DIR / "h4_investigation.txt"
    h4_path.write_text(h4_text, encoding="utf-8")
    print(f"  H4 investigation       → {h4_path.relative_to(_ROOT)}")

    # ── Final stdout summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # Discrimination match count
    match_count = sum(
        max(stats[c][cls]["mean"] for cls in CLASSES) in
        [stats[c][t]["mean"] for t in EXPECTED_TARGET.get(c, [])]
        for c in concept_names
    )
    # recompute cleanly
    match_count = 0
    for cname in concept_names:
        cls_means = {cls: stats[cname][cls]["mean"] for cls in CLASSES}
        argmax = max(cls_means, key=lambda c: cls_means[c])
        if argmax in EXPECTED_TARGET.get(cname, []):
            match_count += 1
    print(f"Discrimination: {match_count}/{len(concept_names)} concepts match expected target class")

    # Extract verdict line from h4 text
    verdict_line = next(
        (l for l in h4_text.splitlines() if l.startswith("VERDICT:")), "VERDICT: not found"
    )
    print(f"H4 verdict: {verdict_line}")

    # Block means from discrimination
    for block_name, members in [
        ("normal_anchor_block", NORMAL_ANCHOR_BLOCK),
        ("interstitial_block",  INTERSTITIAL_BLOCK),
    ]:
        print(f"\n{block_name}:")
        for cname in members:
            row = "  " + cname.ljust(40)
            for cls in CLASSES:
                row += f"  {cls}={stats[cname][cls]['mean']:.4f}"
            print(row)

    print(f"\nAll outputs written to {OUT_DIR.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
