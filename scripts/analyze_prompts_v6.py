"""
Diagnostic analysis of BiomedCLIP v6 concept score features (HAM10000).

Operates on the training partition only. Computes:
  1. Intended-class AUC per concept (one-vs-rest, using CONCEPT_TARGET_CLASS)
  2. Mean intended-class AUC by disease group, compared side-by-side with v5
  3. Per-concept best-class AUC (auc_max), with flags for values ≥ 0.85

Run from project root after extract_features_biomedclip_v6.py:
    python scripts/analyze_prompts_v6.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analysis.concept_targets import CONCEPT_TARGET_CLASS

# ── Paths ─────────────────────────────────────────────────────────────────────
SCORES_V6_NPZ = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLIT_NPZ     = _ROOT / "data/splits/train_test_lesion_split.npz"
REPORT_DIR    = _ROOT / "reports/prompt_analysis/biomedclip_v6"

# v5 group means for comparison (from reports/prompt_analysis/biomedclip_v5/)
V5_GROUP_MEANS: dict[str, float] = {
    "mel":   0.580,
    "bcc":   0.514,
    "akiec": 0.652,
    "bkl":   0.648,
    "df":    0.654,
    "nv":    0.706,
    "vasc":  0.814,
}

HIGH_AUC_FLAG = 0.85   # flag anything at or above this (except red_lacunae on vasc)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not SCORES_V6_NPZ.exists():
        raise FileNotFoundError(
            f"v6 scores not found: {SCORES_V6_NPZ}\n"
            "Run scripts/extract_features_biomedclip_v6.py first."
        )

    # ── Load ──────────────────────────────────────────────────────────────────
    data = np.load(SCORES_V6_NPZ, allow_pickle=True)
    split = np.load(SPLIT_NPZ)
    train_idx = split["train_idx"]

    scores      = data["scores"]        # (10015, 24)
    labels      = data["labels"]        # (10015,) str
    concept_ids = list(data["concept_ids"])   # 24 strings

    assert scores.shape == (10015, 24), f"Unexpected shape: {scores.shape}"
    assert len(concept_ids) == 24

    scores_tr = scores[train_idx]       # (8020, 24)
    labels_tr = labels[train_idx]

    print(f"v6 scores shape : {scores.shape}")
    print(f"Training split  : {len(train_idx):,} images")
    print(f"Concepts        : {len(concept_ids)}")

    all_classes = sorted(np.unique(labels_tr).tolist())

    # ── 1. Intended-class AUC per concept ────────────────────────────────────
    rows = []
    for col_idx, cid in enumerate(concept_ids):
        target_cls = CONCEPT_TARGET_CLASS.get(cid)
        if target_cls is None:
            print(f"  WARNING: {cid} not in CONCEPT_TARGET_CLASS — skipping")
            continue
        score_col = scores_tr[:, col_idx]
        y_bin = (labels_tr == target_cls).astype(int)
        intended_auc = roc_auc_score(y_bin, score_col)

        # Also compute auc_max (best single-class AUC across all 7 classes)
        per_class_aucs = {}
        for cls in all_classes:
            y_b = (labels_tr == cls).astype(int)
            per_class_aucs[cls] = roc_auc_score(y_b, score_col)
        best_cls  = max(per_class_aucs, key=per_class_aucs.__getitem__)
        auc_max   = per_class_aucs[best_cls]

        rows.append({
            "concept_id":      cid,
            "target_class":    target_cls,
            "intended_auc":    round(intended_auc, 4),
            "auc_max":         round(auc_max, 4),
            "auc_max_class":   best_cls,
            **{f"auc_{c}": round(per_class_aucs[c], 4) for c in all_classes},
        })

    df_intended = (
        pd.DataFrame(rows)
        .sort_values("intended_auc", ascending=False)
        .reset_index(drop=True)
    )
    df_intended.to_csv(REPORT_DIR / "intended_class_auc_v6.csv", index=False)

    # ── 2. Mean intended-class AUC by disease group ───────────────────────────
    group_rows = []
    for grp in sorted(set(CONCEPT_TARGET_CLASS.values())):
        subset = df_intended[df_intended["target_class"] == grp]
        v6_mean = float(subset["intended_auc"].mean()) if len(subset) else float("nan")
        v5_mean = V5_GROUP_MEANS.get(grp, float("nan"))
        group_rows.append({
            "disease_group": grp,
            "n_concepts":    len(subset),
            "v5_mean_auc":   round(v5_mean, 4),
            "v6_mean_auc":   round(v6_mean, 4),
            "delta":         round(v6_mean - v5_mean, 4) if not np.isnan(v5_mean) else float("nan"),
        })
    df_groups = pd.DataFrame(group_rows)
    df_groups.to_csv(REPORT_DIR / "disease_group_means_v6_vs_v5.csv", index=False)

    # ── 3. Prompt auc_max ─────────────────────────────────────────────────────
    df_auc_max = (
        df_intended[["concept_id", "target_class", "auc_max", "auc_max_class"]]
        .sort_values("auc_max", ascending=False)
        .reset_index(drop=True)
    )
    df_auc_max.to_csv(REPORT_DIR / "prompt_auc_max_v6.csv", index=False)

    # ── Flag high-AUC prompts ─────────────────────────────────────────────────
    flags = df_intended[df_intended["auc_max"] >= HIGH_AUC_FLAG].copy()
    # red_lacunae on vasc is expected; flag everything else
    unexpected_flags = flags[
        ~((flags["concept_id"] == "red_lacunae") & (flags["auc_max_class"] == "vasc"))
    ]

    # ── Build summary text ────────────────────────────────────────────────────
    mean_intended = float(df_intended["intended_auc"].mean())

    summary_lines = [
        "==== Prompt Analysis: BiomedCLIP v6 (24 concepts, single template) ====",
        "",
        "1. Intended-class AUC per concept (sorted descending):",
        df_intended[["concept_id", "target_class", "intended_auc",
                     "auc_max", "auc_max_class"]].to_string(index=False),
        "",
        f"   Mean intended-class AUC: {mean_intended:.4f}",
        "",
        "2. Mean intended-class AUC by disease group (v5 vs v6):",
        df_groups.to_string(index=False),
        "",
        "3. Top auc_max across 24 concepts:",
        df_auc_max.head(10).to_string(index=False),
        "",
    ]

    if len(flags):
        summary_lines += [
            f"   Concepts with auc_max ≥ {HIGH_AUC_FLAG}:",
        ]
        for _, r in flags.iterrows():
            tag = " [EXPECTED]" if (r["concept_id"] == "red_lacunae"
                                     and r["auc_max_class"] == "vasc") else " [FLAG]"
            summary_lines.append(
                f"     {r['concept_id']:30s} auc_max={r['auc_max']:.4f}"
                f"  best_class={r['auc_max_class']}{tag}"
            )
    else:
        summary_lines.append(f"   No concepts with auc_max ≥ {HIGH_AUC_FLAG}.")

    if len(unexpected_flags):
        summary_lines += [
            "",
            "   *** DIAGNOSTIC WARNING ***",
            f"   {len(unexpected_flags)} unexpected high-AUC concept(s) above {HIGH_AUC_FLAG}:",
        ]
        for _, r in unexpected_flags.iterrows():
            summary_lines.append(
                f"     {r['concept_id']} (best_class={r['auc_max_class']}, "
                f"auc_max={r['auc_max']:.4f}) — investigate prompt specificity"
            )

    summary_lines += ["", f"Outputs → {REPORT_DIR.relative_to(_ROOT)}"]
    summary = "\n".join(summary_lines)

    print("\n" + summary)
    (REPORT_DIR / "summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
