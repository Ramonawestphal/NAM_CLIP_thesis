"""Prompt quality analysis for HAM10000 CLIP concept scores.

Outputs (all under reports/prompt_analysis/):
    prompt_score_distributions.csv
    prompt_class_means.csv
    prompt_class_means_heatmap.png
    prompt_ovr_auc.csv
    top20_prompts_auc.png
    template_comparison.csv
    paired_contrasts.csv

Split saved to:
    data/splits/train_test_lesion_split.npz

Run from project root:
    python scripts/analyze_prompts.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.analysis.prompt_quality import (
    analysis1_score_distribution,
    analysis2_class_means,
    analysis3_ovr_auc,
    analysis4_template_comparison,
    analysis5_paired_contrasts,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCORES_NPZ  = pathlib.Path("data/features/ham10000_concept_scores.npz")
SPLIT_NPZ   = pathlib.Path("data/splits/train_test_lesion_split.npz")
REPORT_DIR  = pathlib.Path("reports/prompt_analysis")
# ---------------------------------------------------------------------------


def build_lesion_split(
    lesion_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 80/20 split by lesion_id; seed fixed at 42."""
    # Map each unique lesion to its dx label (one lesion → one class)
    seen: dict[str, str] = {}
    for lid, lbl in zip(lesion_ids, labels):
        seen[lid] = lbl

    unique_lesions = list(seen.keys())
    lesion_labels  = [seen[l] for l in unique_lesions]

    np.random.seed(42)
    train_lesions, test_lesions = train_test_split(
        unique_lesions,
        test_size=0.2,
        random_state=42,
        stratify=lesion_labels,
    )

    train_set = set(train_lesions)
    test_set  = set(test_lesions)

    train_idx = np.array([i for i, lid in enumerate(lesion_ids) if lid in train_set])
    test_idx  = np.array([i for i, lid in enumerate(lesion_ids) if lid in test_set])
    return train_idx, test_idx


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_NPZ.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load NPZ
    # ------------------------------------------------------------------
    data = np.load(SCORES_NPZ, allow_pickle=True)
    scores              = data["scores"]              # (10015, 72)
    labels              = data["labels"]
    lesion_ids          = data["lesion_ids"]
    concept_ids         = list(data["concept_ids"])
    prompts             = list(data["prompts"])
    prompt_concept_idx  = data["prompt_concept_idx"]
    prompt_template_idx = data["prompt_template_idx"]
    tiers               = data["tiers"]

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    if SPLIT_NPZ.exists():
        split     = np.load(SPLIT_NPZ)
        train_idx = split["train_idx"]
        test_idx  = split["test_idx"]
        print(f"Loaded existing split: {len(train_idx):,} train / {len(test_idx):,} test images")
    else:
        train_idx, test_idx = build_lesion_split(lesion_ids, labels)
        np.savez(SPLIT_NPZ, train_idx=train_idx, test_idx=test_idx)
        print(f"Created split: {len(train_idx):,} train / {len(test_idx):,} test images")

    scores_train = scores[train_idx]
    labels_train = labels[train_idx]

    # ------------------------------------------------------------------
    # Prompt metadata DataFrame (72 rows)
    # ------------------------------------------------------------------
    tmpl_name = {0: "t1", 1: "t2", 2: "t3"}
    meta = pd.DataFrame({
        "prompt_idx":  np.arange(len(prompts)),
        "concept_id":  [concept_ids[i] for i in prompt_concept_idx],
        "template":    [tmpl_name[i]   for i in prompt_template_idx],
        "prompt":      prompts,
        "concept_idx": list(prompt_concept_idx),
        "tier":        [int(tiers[i])  for i in prompt_concept_idx],
    })

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

    print("\n========== Summary ==========")
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
    print("==============================")
    print(f"\nOutputs → {REPORT_DIR}")
    print(f"Split   → {SPLIT_NPZ}")


if __name__ == "__main__":
    main()
