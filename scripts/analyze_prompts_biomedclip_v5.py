"""Prompt quality analysis for BiomedCLIP HAM10000 concept scores (v5 prompts).

Mirrors scripts/analyze_prompts_biomedclip.py but points at
data/features/biomedclip/ham10000_concept_scores_v5.npz and writes reports
to reports/prompt_analysis/biomedclip_v5/. Uses the same train/test split.

Outputs (all under reports/prompt_analysis/biomedclip_v5/):
    prompt_score_distributions.csv
    prompt_class_means.csv
    prompt_class_means_heatmap.png
    prompt_ovr_auc.csv
    top20_prompts_auc.png
    template_comparison.csv
    paired_contrasts.csv

Run from project root:
    python scripts/analyze_prompts_biomedclip_v5.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.analysis.prompt_quality import (
    analysis1_score_distribution,
    analysis2_class_means,
    analysis3_ovr_auc,
    analysis4_template_comparison,
    analysis5_paired_contrasts,
    build_prompt_meta_from_npz,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCORES_NPZ = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v5.npz"
SPLIT_NPZ  = _ROOT / "data/splits/train_test_lesion_split.npz"
REPORT_DIR = _ROOT / "reports/prompt_analysis/biomedclip_v5"
# ---------------------------------------------------------------------------


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not SCORES_NPZ.exists():
        raise FileNotFoundError(
            f"BiomedCLIP v5 scores not found: {SCORES_NPZ}\n"
            "Run scripts/iterate_prompts_biomedclip_v5.py first."
        )
    if not SPLIT_NPZ.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_NPZ}\n"
            "Run scripts/analyze_prompts.py once to create the lesion split."
        )

    print("Encoder : BiomedCLIP ViT-B/16 (PubMedBERT)")
    print("Prompts : v5_biomedclip (PubMed-caption style for mel + nv-anchor concepts)")
    print(f"Scores  : {SCORES_NPZ.relative_to(_ROOT)}")

    # ------------------------------------------------------------------
    # Load NPZ + split
    # ------------------------------------------------------------------
    data      = np.load(SCORES_NPZ, allow_pickle=True)
    split     = np.load(SPLIT_NPZ)
    train_idx = split["train_idx"]

    scores      = data["scores"]
    labels      = data["labels"]
    concept_ids = list(data["concept_ids"])
    tiers       = data["tiers"]

    scores_train = scores[train_idx]
    labels_train = labels[train_idx]

    print(f"Loaded BiomedCLIP v5 scores : {scores.shape}")
    print(f"Training split              : {len(train_idx):,} images")

    meta = build_prompt_meta_from_npz(data)

    # ------------------------------------------------------------------
    # Analysis 1
    # ------------------------------------------------------------------
    print("Analysis 1: score distributions…")
    df1 = analysis1_score_distribution(scores_train, meta)
    df1.to_csv(REPORT_DIR / "prompt_score_distributions.csv", index=False)

    # ------------------------------------------------------------------
    # Analysis 2
    # ------------------------------------------------------------------
    print("Analysis 2: per-class means…")
    df2 = analysis2_class_means(scores_train, labels_train, meta, REPORT_DIR)
    df2.to_csv(REPORT_DIR / "prompt_class_means.csv", index=False)

    # ------------------------------------------------------------------
    # Analysis 3
    # ------------------------------------------------------------------
    print("Analysis 3: OvR AUC (72 prompts × 7 classes)…")
    df3 = analysis3_ovr_auc(scores_train, labels_train, meta, REPORT_DIR)
    df3.to_csv(REPORT_DIR / "prompt_ovr_auc.csv", index=False)

    # ------------------------------------------------------------------
    # Analysis 4
    # ------------------------------------------------------------------
    print("Analysis 4: template comparison…")
    df4 = analysis4_template_comparison(df3, concept_ids, tiers)
    df4.to_csv(REPORT_DIR / "template_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # Analysis 5
    # ------------------------------------------------------------------
    print("Analysis 5: paired contrasts…")
    df5 = analysis5_paired_contrasts(scores_train, labels_train, meta)
    df5.to_csv(REPORT_DIR / "paired_contrasts.csv", index=False)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    n_dead        = int(df1["flagged_dead"].sum())
    n_pass        = int((df5["status"] == "PASS").sum())
    n_contrasts   = len(df5)
    mean_auc_max  = float(df3["auc_max"].mean())
    top5          = df3.nlargest(5, "auc_max")
    winner_counts = df4["best_template"].value_counts()

    print("\n========== Summary (BiomedCLIP v5) ==========")
    print(f"Dead prompts (p95-p05 < 0.02) : {n_dead} / 72")
    print(f"Paired contrasts passed        : {n_pass} / {n_contrasts}")
    print(f"Mean auc_max across 72 prompts : {mean_auc_max:.4f}")
    print(f"\nTemplate wins  — "
          f"t1: {winner_counts.get('t1', 0)}  "
          f"t2: {winner_counts.get('t2', 0)}  "
          f"t3: {winner_counts.get('t3', 0)}")
    print("\nTop 5 prompts by auc_max:")
    for _, r in top5.iterrows():
        print(f"  [{r['concept_id']} {r['template']}]  "
              f"auc_max={r['auc_max']:.4f}  best_class={r['auc_argmax_class']}")
        print(f"    \"{r['prompt']}\"")
    print("=============================================")
    print(f"\nOutputs → {REPORT_DIR}")


if __name__ == "__main__":
    main()
