"""Encoder comparison using designed target classes (methodologically correct version).

Unlike compare_encoders.py (which uses each concept's empirical argmax class),
this script evaluates every concept on the class it was *designed* to detect,
as defined in src/analysis/concept_targets.CONCEPT_TARGET_CLASS.

Inputs (loaded from NPZ; no dependency on pre-generated analysis CSVs):
    data/features/ham10000_concept_scores_v4.npz        (ViT-B/32)
    data/features/biomedclip/ham10000_concept_scores_v4.npz  (BiomedCLIP)
    data/splits/train_test_lesion_split.npz

Outputs (reports/encoder_comparison/designed_targets/):
    designed_target_auc_per_concept.csv
    designed_target_auc_per_disease_group.csv
    designed_target_auc_comparison_summary.txt

Run from project root:
    python scripts/compare_encoders_designed_targets.py
"""

from __future__ import annotations

import io
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analysis.concept_targets import CONCEPT_TARGET_CLASS, MALIGNANT_CLASSES
from src.analysis.prompt_quality import build_prompt_meta_from_npz

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VITB32_NPZ  = _ROOT / "data/features/ham10000_concept_scores_v4.npz"
BMC_NPZ     = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v4.npz"
SPLIT_NPZ   = _ROOT / "data/splits/train_test_lesion_split.npz"
OUTPUT_DIR  = _ROOT / "reports/encoder_comparison/designed_targets"

PER_CONCEPT_CSV   = OUTPUT_DIR / "designed_target_auc_per_concept.csv"
PER_GROUP_CSV     = OUTPUT_DIR / "designed_target_auc_per_disease_group.csv"
SUMMARY_TXT       = OUTPUT_DIR / "designed_target_auc_comparison_summary.txt"
# ---------------------------------------------------------------------------

# Display order for disease groups in the summary
GROUP_ORDER = ["mel", "bcc", "akiec", "bkl", "df", "nv", "vasc"]


def _max_auc_on_class(
    scores: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    concept_id: str,
    target_class: str,
) -> float:
    """Max OvR AUC across t1/t2/t3 for one concept on its designed target class."""
    y_bin = (labels == target_class).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return np.nan

    best = np.nan
    for _, row in meta[meta["concept_id"] == concept_id].iterrows():
        col = scores[:, row["prompt_idx"]]
        try:
            auc = float(roc_auc_score(y_bin, col))
            if np.isnan(best) or auc > best:
                best = auc
        except Exception:
            pass
    return best


def compute_designed_target_aucs(
    scores: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    concept_ids: list[str],
) -> dict[str, float]:
    """Return {concept_id: max_auc_on_designed_target} for all concepts."""
    return {
        cid: _max_auc_on_class(
            scores, labels, meta, cid, CONCEPT_TARGET_CLASS[cid]
        )
        for cid in concept_ids
    }


def load_encoder(
    npz_path: pathlib.Path,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Load scores NPZ and return (scores_train, labels_train, meta, concept_ids)."""
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Scores NPZ not found: {npz_path}\n"
            "Run the corresponding extract_features script first."
        )
    data = np.load(npz_path, allow_pickle=True)
    scores_train = data["scores"][train_idx]
    labels_train = data["labels"][train_idx]
    meta         = build_prompt_meta_from_npz(data)
    concept_ids  = list(data["concept_ids"])
    return scores_train, labels_train, meta, concept_ids


def build_per_concept_df(
    concept_ids: list[str],
    vitb32_aucs: dict[str, float],
    bmc_aucs: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for cid in concept_ids:
        v1  = vitb32_aucs[cid]
        bmc = bmc_aucs[cid]
        delta = (bmc - v1) if (not np.isnan(v1) and not np.isnan(bmc)) else np.nan
        rows.append({
            "concept_id":      cid,
            "designed_target": CONCEPT_TARGET_CLASS[cid],
            "vitb32_auc":      round(v1,    4) if not np.isnan(v1)    else np.nan,
            "biomedclip_auc":  round(bmc,   4) if not np.isnan(bmc)   else np.nan,
            "delta":           round(delta, 4) if not np.isnan(delta) else np.nan,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("delta", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def build_per_group_df(per_concept: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        sub = per_concept[per_concept["designed_target"] == group]
        if sub.empty:
            continue
        n = len(sub)
        v1_mean  = float(sub["vitb32_auc"].mean())
        bmc_mean = float(sub["biomedclip_auc"].mean())
        delta    = round(bmc_mean - v1_mean, 4)

        v1_best_row  = sub.loc[sub["vitb32_auc"].idxmax()]
        bmc_best_row = sub.loc[sub["biomedclip_auc"].idxmax()]

        rows.append({
            "designed_target":       group,
            "n_concepts":            n,
            "vitb32_mean_auc":       round(v1_mean,  4),
            "biomedclip_mean_auc":   round(bmc_mean, 4),
            "delta":                 delta,
            "vitb32_best_concept":   v1_best_row["concept_id"],
            "biomedclip_best_concept": bmc_best_row["concept_id"],
        })
    return (
        pd.DataFrame(rows)
        .sort_values("delta", ascending=False)
        .reset_index(drop=True)
    )


def format_summary(
    per_concept: pd.DataFrame,
    per_group: pd.DataFrame,
) -> str:
    buf = io.StringIO()

    def w(line: str = "") -> None:
        buf.write(line + "\n")

    w("==== Designed-target AUC comparison: ViT-B/32 vs BiomedCLIP (HAM10000 v4) ====")
    w()
    w("Per disease group (mean AUC across concepts designed for that target):")
    w()

    # Build a group lookup for ordered display
    grp_lookup = per_group.set_index("designed_target")
    for group in GROUP_ORDER:
        if group not in grp_lookup.index:
            continue
        r = grp_lookup.loc[group]
        delta_sign = "+" if r["delta"] >= 0 else ""
        w(
            f"   {group:<7}: ViT-B/32 {r['vitb32_mean_auc']:.3f}  |  "
            f"BiomedCLIP {r['biomedclip_mean_auc']:.3f}  |  "
            f"Δ {delta_sign}{r['delta']:.3f}  (n={int(r['n_concepts'])})"
        )

    w()
    w("Overall mean across all 24 concepts:")
    overall_v1  = float(per_concept["vitb32_auc"].mean())
    overall_bmc = float(per_concept["biomedclip_auc"].mean())
    overall_d   = overall_bmc - overall_v1
    w(f"   ViT-B/32   : {overall_v1:.3f}")
    w(f"   BiomedCLIP : {overall_bmc:.3f}")
    w(f"   Δ          : {'+' if overall_d >= 0 else ''}{overall_d:.3f}")

    w()
    malignant_concepts = per_concept[
        per_concept["designed_target"].isin(MALIGNANT_CLASSES)
    ]
    n_mal    = len(malignant_concepts)
    mal_v1   = float(malignant_concepts["vitb32_auc"].mean())
    mal_bmc  = float(malignant_concepts["biomedclip_auc"].mean())
    mal_d    = mal_bmc - mal_v1
    mal_sign = "+" if mal_d >= 0 else ""
    w(f"Malignant-only mean (mel + bcc + akiec; {n_mal} concepts):")
    w(f"   ViT-B/32   : {mal_v1:.3f}")
    w(f"   BiomedCLIP : {mal_bmc:.3f}")
    w(f"   Δ          : {mal_sign}{mal_d:.3f}")

    w()
    top5_improved = per_concept.head(5)
    w("Top 5 concepts by BiomedCLIP improvement (Δ):")
    for _, r in top5_improved.iterrows():
        d_sign = "+" if r["delta"] >= 0 else ""
        w(
            f"   [{r['concept_id']}]  designed={r['designed_target']}  "
            f"ViT-B/32={r['vitb32_auc']:.3f}  BiomedCLIP={r['biomedclip_auc']:.3f}  "
            f"Δ={d_sign}{r['delta']:.3f}"
        )

    w()
    regressed = per_concept[per_concept["delta"] < 0].tail(5).iloc[::-1]
    w("Top 5 concepts by BiomedCLIP regression (Δ):")
    if regressed.empty:
        w("   (none)")
    else:
        for _, r in regressed.iterrows():
            w(
                f"   [{r['concept_id']}]  designed={r['designed_target']}  "
                f"ViT-B/32={r['vitb32_auc']:.3f}  BiomedCLIP={r['biomedclip_auc']:.3f}  "
                f"Δ={r['delta']:.3f}"
            )

    w()
    n_improved  = int((per_concept["delta"] >= 0.02).sum())
    n_regressed = int((per_concept["delta"] <= -0.02).sum())
    w(
        f"Concepts where BiomedCLIP improves by ≥0.02 AUC on its designed target: "
        f"{n_improved} / {len(per_concept)}"
    )
    w(
        f"Concepts where BiomedCLIP regresses by ≥0.02 AUC on its designed target: "
        f"{n_regressed} / {len(per_concept)}"
    )

    return buf.getvalue()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLIT_NPZ.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_NPZ}\n"
            "Run scripts/analyze_prompts.py once to create the lesion split."
        )
    train_idx = np.load(SPLIT_NPZ)["train_idx"]

    # ------------------------------------------------------------------
    # Load both encoders
    # ------------------------------------------------------------------
    print("Loading ViT-B/32 scores…")
    v1_scores, v1_labels, v1_meta, v1_cids = load_encoder(VITB32_NPZ, train_idx)

    print("Loading BiomedCLIP scores…")
    bmc_scores, bmc_labels, bmc_meta, bmc_cids = load_encoder(BMC_NPZ, train_idx)

    # ------------------------------------------------------------------
    # Validate concept coverage
    # ------------------------------------------------------------------
    for cid in v1_cids:
        if cid not in CONCEPT_TARGET_CLASS:
            raise KeyError(
                f"Concept '{cid}' in ViT-B/32 NPZ has no entry in CONCEPT_TARGET_CLASS.\n"
                "Update src/analysis/concept_targets.py before proceeding."
            )
    for cid in bmc_cids:
        if cid not in CONCEPT_TARGET_CLASS:
            raise KeyError(
                f"Concept '{cid}' in BiomedCLIP NPZ has no entry in CONCEPT_TARGET_CLASS.\n"
                "Update src/analysis/concept_targets.py before proceeding."
            )

    assert v1_cids == bmc_cids, (
        "ViT-B/32 and BiomedCLIP NPZ files contain different concept_id lists. "
        "Both must be built from the same prompt set."
    )
    concept_ids = v1_cids
    print(f"Concepts: {len(concept_ids)}  |  Training images: {len(train_idx):,}")

    # ------------------------------------------------------------------
    # Compute designed-target AUCs
    # ------------------------------------------------------------------
    print("Computing designed-target AUC for ViT-B/32…")
    v1_aucs  = compute_designed_target_aucs(v1_scores,  v1_labels,  v1_meta,  concept_ids)

    print("Computing designed-target AUC for BiomedCLIP…")
    bmc_aucs = compute_designed_target_aucs(bmc_scores, bmc_labels, bmc_meta, concept_ids)

    # ------------------------------------------------------------------
    # Build output DataFrames
    # ------------------------------------------------------------------
    per_concept = build_per_concept_df(concept_ids, v1_aucs, bmc_aucs)
    per_group   = build_per_group_df(per_concept)

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------
    summary = format_summary(per_concept, per_group)
    print()
    print(summary)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    per_concept.to_csv(PER_CONCEPT_CSV, index=False)
    per_group.to_csv(PER_GROUP_CSV,     index=False)
    SUMMARY_TXT.write_text(summary, encoding="utf-8")

    print(f"Saved: {PER_CONCEPT_CSV}")
    print(f"Saved: {PER_GROUP_CSV}")
    print(f"Saved: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
