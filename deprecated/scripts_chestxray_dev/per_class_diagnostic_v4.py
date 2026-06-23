"""
Per-class concept score diagnostic for chest X-ray BiomedCLIP v4 features.

Mirror of per_class_diagnostic_v3.py pointed at v4 artefacts.
EXPECTED_TARGET mapping is identical to v2 (expected class does not change
with prompt wording — only discriminability should improve).

Operates on train pool only. Test indices are never touched.
v1/v2 artefacts are read-only; this script does not modify them.

Outputs:
    results/chestxray/per_class_diagnostic_v4/per_class_means.csv
    results/chestxray/per_class_diagnostic_v4/per_class_heatmap.png
    results/chestxray/per_class_diagnostic_v4/discrimination_summary.txt
    results/chestxray/per_class_diagnostic_v4/h4_investigation.txt

Run from project root:
    python scripts/chestxray/per_class_diagnostic_v4.py
"""

from __future__ import annotations

import csv
import pathlib
import sys
from typing import Dict, List

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
OUT_DIR     = _ROOT / "results/chestxray/per_class_diagnostic_v4"

# ── Block membership ──────────────────────────────────────────────────────────
# v4 has 17 concepts: v3 minus hyperinflation. Block membership
# is unchanged from v2; normal_anchor_block has 4 members (no_focal_opacity was
# dropped back in v2).
NORMAL_ANCHOR_BLOCK = [
    "clear_lung_fields",
    "sharp_costophrenic_angles",
    "normal_cardiac_silhouette",
    "symmetric_lung_aeration",
]
INTERSTITIAL_BLOCK = [
    "bilateral_interstitial_pattern",
    "peribronchial_cuffing",
    "perihilar_infiltrates",
]

# ── Expected target classes (same logic as v1; silhouette_sign restored) ──────
EXPECTED_TARGET: Dict[str, List[str]] = {
    "consolidation":                    ["bacteria", "virus"],
    "focal_opacity":                    ["bacteria", "virus"],
    "air_bronchograms":                 ["bacteria", "virus"],
    "pleural_effusion":                 ["bacteria", "virus"],
    "silhouette_sign":                  ["bacteria", "virus"],
    "patchy_infiltrate":                ["bacteria", "virus"],
    "lobar_consolidation":              ["bacteria"],
    "dense_segmental_opacity":          ["bacteria"],
    "parapneumonic_effusion":           ["bacteria"],
    "round_pneumonia":                  ["bacteria"],
    "bilateral_interstitial_pattern":   ["virus"],
    "peribronchial_cuffing":            ["virus"],
    "perihilar_infiltrates":            ["virus"],
    "clear_lung_fields":                ["normal"],
    "sharp_costophrenic_angles":        ["normal"],
    "normal_cardiac_silhouette":        ["normal"],
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
    "clear_lung_fields":                "T3-normal",
    "sharp_costophrenic_angles":        "T3-normal",
    "normal_cardiac_silhouette":        "T3-normal",
    "symmetric_lung_aeration":          "T3-normal",
}

CLASSES = ["normal", "bacteria", "virus"]


def compute_per_class_stats(
    scores: np.ndarray,
    concept_names: List[str],
    labels_subtype: np.ndarray,
) -> Dict[str, Dict[str, Dict[str, float]]]:
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
                writer.writerow([cname, cls,
                                  f"{s['mean']:.6f}", f"{s['std']:.6f}", s["n"]])
    print(f"  CSV saved → {path.relative_to(_ROOT)}")


def plot_heatmaps(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    concept_names: List[str],
    path: pathlib.Path,
) -> None:
    n_concepts = len(concept_names)
    means = np.array([[stats[c][cls]["mean"] for cls in CLASSES] for c in concept_names])
    row_mean = means.mean(axis=1, keepdims=True)
    row_std  = means.std(axis=1, keepdims=True)
    row_std  = np.where(row_std < 1e-9, 1.0, row_std)
    z_means  = (means - row_mean) / row_std

    fig, axes = plt.subplots(2, 1, figsize=(6, max(10, n_concepts * 0.55 * 2)),
                             gridspec_kw={"hspace": 0.35})

    for ax, data, cmap, title, vrange, fmt in [
        (axes[0], means,   "viridis", "Absolute mean cosine similarity (v4)",
         (means.min(), means.max()), "{:.3f}"),
        (axes[1], z_means, "RdBu_r",  "Z-scored means per-concept across classes (v4)",
         (-2.5, 2.5), "{:.2f}"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, fontsize=9)
        ax.set_yticks(range(n_concepts))
        ax.set_yticklabels(concept_names, fontsize=7)
        ax.set_title(title, fontsize=9, pad=8)
        for r in range(n_concepts):
            for c in range(len(CLASSES)):
                val = data[r, c]
                contrast = "white" if abs(val) > (vrange[1] * 0.6) else "black"
                ax.text(c, r, fmt.format(val), ha="center", va="center",
                        fontsize=6, color=contrast)

    fig.suptitle("Chest X-ray BiomedCLIP v4: per-class concept scores (train pool)",
                 fontsize=10, y=1.01)
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
    lines.append("PER-CONCEPT DISCRIMINATION SUMMARY v4 (train pool)")
    lines.append("=" * 70)

    tier_match: Dict[str, List[bool]] = {}
    match_count = 0

    for cname in concept_names:
        cls_means = {cls: stats[cname][cls]["mean"] for cls in CLASSES}
        argmax    = max(cls_means, key=lambda c: cls_means[c])
        argmin    = min(cls_means, key=lambda c: cls_means[c])
        spread    = cls_means[argmax] - cls_means[argmin]
        expected  = EXPECTED_TARGET.get(cname, [])
        tier      = TIER_LABELS.get(cname, "unknown")
        is_match  = argmax in expected
        match_count += int(is_match)
        tier_match.setdefault(tier, []).append(is_match)

        lines.append(
            f"{cname:<40s}  argmax={argmax:<10s}  expected={str(expected):<22s}"
            f"  spread={spread:.4f}  [{tier}]  {'MATCH' if is_match else 'MISMATCH'}"
        )
        if not is_match:
            lines.append(
                "    MISMATCH details: "
                + "  ".join(f"{c}={cls_means[c]:.4f}" for c in CLASSES)
            )

    lines.append(f"\nConcepts matching expected target: {match_count} / {len(concept_names)}")
    lines.append("\nBreakdown by tier:")
    for tier in sorted(tier_match):
        m = sum(tier_match[tier])
        t = len(tier_match[tier])
        lines.append(f"  {tier:<18s}: {m}/{t}")

    return "\n".join(lines)


def build_h4_investigation(
    scores_train: np.ndarray,
    concept_names: List[str],
    labels_subtype: np.ndarray,
    stats: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("H4 INVESTIGATION v4: normal_anchor_block × interstitial_block")
    lines.append("=" * 70)
    lines.append(
        "Repeating H4 investigation on v4 features to check whether prompt\n"
        "revision changed the between-block correlation pattern.\n"
        f"normal_anchor_block members (v4): {NORMAL_ANCHOR_BLOCK}\n"
        f"interstitial_block  members (v4): {INTERSTITIAL_BLOCK}\n"
    )

    anc_idx = [concept_names.index(c) for c in NORMAL_ANCHOR_BLOCK
               if c in concept_names]
    int_idx = [concept_names.index(c) for c in INTERSTITIAL_BLOCK
               if c in concept_names]

    # Per-class means and discrimination check
    lines.append("─" * 60)
    lines.append("STEP 1-2: Per-class means and discrimination check")
    lines.append("─" * 60)

    lines.append("\nnormal_anchor_block (expected: normal > virus for all):")
    anchor_all_correct = True
    for cname in NORMAL_ANCHOR_BLOCK:
        if cname not in concept_names:
            lines.append(f"  {cname:<40s}  NOT IN V4 CONCEPT SET")
            continue
        m_n = stats[cname]["normal"]["mean"]
        m_v = stats[cname]["virus"]["mean"]
        m_b = stats[cname]["bacteria"]["mean"]
        ok  = m_n > m_v
        anchor_all_correct = anchor_all_correct and ok
        lines.append(
            f"  {cname:<40s}  normal={m_n:.4f}  virus={m_v:.4f}"
            f"  bacteria={m_b:.4f}  → {'normal > virus ✓' if ok else 'normal < virus ✗'}"
        )

    lines.append("\ninterstitial_block (expected: virus > normal for all):")
    interstitial_all_correct = True
    for cname in INTERSTITIAL_BLOCK:
        if cname not in concept_names:
            lines.append(f"  {cname:<40s}  NOT IN V4 CONCEPT SET")
            continue
        m_n = stats[cname]["normal"]["mean"]
        m_v = stats[cname]["virus"]["mean"]
        m_b = stats[cname]["bacteria"]["mean"]
        ok  = m_v > m_n
        interstitial_all_correct = interstitial_all_correct and ok
        lines.append(
            f"  {cname:<40s}  normal={m_n:.4f}  virus={m_v:.4f}"
            f"  bacteria={m_b:.4f}  → {'virus > normal ✓' if ok else 'virus < normal ✗'}"
        )

    prompts_discriminate = anchor_all_correct and interstitial_all_correct
    lines.append(
        f"\nStep 2 decision: anchor all correct={anchor_all_correct}, "
        f"interstitial all correct={interstitial_all_correct}"
    )
    lines.append(
        "→ Prompts discriminate per-class correctly: "
        + ("YES" if prompts_discriminate else "NO")
    )

    # Within-normal-class correlations
    lines.append("")
    lines.append("─" * 60)
    lines.append("STEP 3: Cross-image correlations (within normal class)")
    lines.append("─" * 60)

    full_cross_vals = [
        float(np.corrcoef(scores_train[:, ai], scores_train[:, ii])[0, 1])
        for ai in anc_idx for ii in int_idx
    ]
    full_cross_mean = float(np.mean(np.abs(full_cross_vals)))
    lines.append(f"\nFull train-pool mean |r| (anchor × interstitial): {full_cross_mean:.4f}")

    normal_mask   = labels_subtype == "normal"
    scores_normal = scores_train[normal_mask]
    lines.append(f"Within-normal-class correlations (n={normal_mask.sum()} images):")
    anchor_names_present = [c for c in NORMAL_ANCHOR_BLOCK if c in concept_names]
    int_names_present    = [c for c in INTERSTITIAL_BLOCK   if c in concept_names]
    lines.append(
        f"{'':>42s}" + "".join(f"{c:<28s}" for c in int_names_present)
    )
    within_vals: List[float] = []
    for cname_a in anchor_names_present:
        ai      = concept_names.index(cname_a)
        row_str = f"  {cname_a:<40s}"
        for cname_b in int_names_present:
            ii  = concept_names.index(cname_b)
            r   = float(np.corrcoef(scores_normal[:, ai], scores_normal[:, ii])[0, 1])
            within_vals.append(r)
            row_str += f"  {r:+.4f}              "
        lines.append(row_str)

    within_mean_abs = float(np.mean(np.abs(within_vals)))
    lines.append(f"\nWithin-normal mean |r|: {within_mean_abs:.4f}")
    within_high = within_mean_abs > 0.4

    # Verdict
    lines.append("")
    lines.append("─" * 60)
    lines.append("VERDICT")
    lines.append("─" * 60)

    if prompts_discriminate and within_high:
        verdict = "VERDICT: Explanation A (semantic-axis confound) — consistent with v1 finding"
    elif not prompts_discriminate:
        verdict = "VERDICT: Explanation B (prompt discrimination failure) — revision insufficient"
    else:
        verdict = "VERDICT: Mixed / inconclusive"

    lines.append(verdict)
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Prompt version: v4")

    print(f"Loading scores from {SCORES_NPZ.relative_to(_ROOT)} ...")
    data          = np.load(SCORES_NPZ, allow_pickle=True)
    scores_all    = data["scores"]
    concept_names = data["concept_names"].tolist()

    print(f"Loading outer split from {OUTER_SPLIT.relative_to(_ROOT)} ...")
    split          = np.load(OUTER_SPLIT, allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]
    labels_sub_all = split["labels_subtype"]

    scores_train = scores_all[train_pool_idx]
    labels_train = labels_sub_all[train_pool_idx]

    valid_mask = ~np.isnan(scores_train).any(axis=1)
    if (~valid_mask).sum():
        print(f"  Dropping {(~valid_mask).sum()} NaN rows from train pool")
    scores_clean = scores_train[valid_mask]
    labels_clean = labels_train[valid_mask]

    print(f"  Train pool: {len(scores_clean)} images  |  Concepts: {len(concept_names)}")
    for cls in CLASSES:
        print(f"    {cls}: {(labels_clean == cls).sum()} images")

    print("\nComputing per-class statistics...")
    stats = compute_per_class_stats(scores_clean, concept_names, labels_clean)

    write_csv(stats, concept_names, OUT_DIR / "per_class_means.csv")
    plot_heatmaps(stats, concept_names, OUT_DIR / "per_class_heatmap.png")

    disc_text = build_discrimination_summary(stats, concept_names)
    (OUT_DIR / "discrimination_summary.txt").write_text(disc_text, encoding="utf-8")
    print(f"  Discrimination summary → {(OUT_DIR / 'discrimination_summary.txt').relative_to(_ROOT)}")

    h4_text = build_h4_investigation(scores_clean, concept_names, labels_clean, stats)
    (OUT_DIR / "h4_investigation.txt").write_text(h4_text, encoding="utf-8")
    print(f"  H4 investigation       → {(OUT_DIR / 'h4_investigation.txt').relative_to(_ROOT)}")

    # Console summary
    match_count = 0
    for cname in concept_names:
        cls_means = {cls: stats[cname][cls]["mean"] for cls in CLASSES}
        if max(cls_means, key=lambda c: cls_means[c]) in EXPECTED_TARGET.get(cname, []):
            match_count += 1

    verdict_line = next(
        (l for l in h4_text.splitlines() if l.startswith("VERDICT:")), "VERDICT: not found"
    )

    print("\n" + "=" * 70)
    print("V4 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Discrimination: {match_count}/{len(concept_names)} concepts match expected target class")
    print(f"H4 verdict: {verdict_line}")
    print(f"\nAll outputs written to {OUT_DIR.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
