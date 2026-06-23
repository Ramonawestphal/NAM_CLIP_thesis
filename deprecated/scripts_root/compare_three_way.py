"""Three-way comparison: ViT-B/32 v4 vs BiomedCLIP v4 vs BiomedCLIP v5.

Evaluates every concept on its designed target class (from concept_targets.py),
taking the max AUC across templates. Includes a per-template breakdown for
melanoma-targeted concepts under BiomedCLIP v5, which reveals which of the
three new PubMed-caption template styles works best.

Outputs (reports/encoder_comparison/three_way/):
    three_way_auc_per_concept.csv
    three_way_auc_per_disease_group.csv
    three_way_summary.txt

Run from project root:
    python scripts/compare_three_way.py
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
VITB32_NPZ = _ROOT / "data/features/ham10000_concept_scores_v4.npz"
BMCV4_NPZ  = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v4.npz"
BMCV5_NPZ  = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v5.npz"
SPLIT_NPZ  = _ROOT / "data/splits/train_test_lesion_split.npz"
OUTPUT_DIR = _ROOT / "reports/encoder_comparison/three_way"

PER_CONCEPT_CSV = OUTPUT_DIR / "three_way_auc_per_concept.csv"
PER_GROUP_CSV   = OUTPUT_DIR / "three_way_auc_per_disease_group.csv"
SUMMARY_TXT     = OUTPUT_DIR / "three_way_summary.txt"
# ---------------------------------------------------------------------------

GROUP_ORDER = ["mel", "bcc", "akiec", "bkl", "df", "nv", "vasc"]

# v5 template style labels for the per-template breakdown section
TEMPLATE_STYLES = {
    "t1": "Dermoscopy of melanoma showing ...",
    "t2": "Dermoscopic image of melanoma demonstrating ...",
    "t3": "[feature] in dermoscopy of melanoma",
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _load_encoder(
    npz_path: pathlib.Path,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], np.ndarray]:
    """Load NPZ; return (scores_train, labels_train, meta, concept_ids, image_ids)."""
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Scores NPZ not found: {npz_path}\n"
            "Run the corresponding extract/iterate script first."
        )
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["scores"][train_idx],
        data["labels"][train_idx],
        build_prompt_meta_from_npz(data),
        list(data["concept_ids"]),
        data["image_ids"],
    )


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


def _compute_aucs(
    scores: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    concept_ids: list[str],
) -> dict[str, float]:
    return {
        cid: _max_auc_on_class(scores, labels, meta, cid, CONCEPT_TARGET_CLASS[cid])
        for cid in concept_ids
    }


def _per_template_mel_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
) -> dict[str, float]:
    """Mean OvR AUC on mel per template, averaged across all mel-targeted concepts."""
    mel_concepts = [
        cid for cid, tgt in CONCEPT_TARGET_CLASS.items() if tgt == "mel"
    ]
    y_bin = (labels == "mel").astype(int)
    if y_bin.sum() == 0:
        return {"t1": np.nan, "t2": np.nan, "t3": np.nan}

    result: dict[str, float] = {}
    for tmpl in ("t1", "t2", "t3"):
        aucs = []
        for cid in mel_concepts:
            rows = meta[(meta["concept_id"] == cid) & (meta["template"] == tmpl)]
            if rows.empty:
                continue
            col = scores[:, rows.iloc[0]["prompt_idx"]]
            try:
                aucs.append(float(roc_auc_score(y_bin, col)))
            except Exception:
                pass
        result[tmpl] = float(np.mean(aucs)) if aucs else np.nan
    return result


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def _build_per_concept(
    concept_ids: list[str],
    v1_aucs: dict[str, float],
    bmc4_aucs: dict[str, float],
    bmc5_aucs: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for cid in concept_ids:
        v1   = v1_aucs[cid]
        bmc4 = bmc4_aucs[cid]
        bmc5 = bmc5_aucs[cid]
        d_v4v5 = round(bmc5 - bmc4, 4) if not (np.isnan(bmc4) or np.isnan(bmc5)) else np.nan
        d_v1v5 = round(bmc5 - v1,   4) if not (np.isnan(v1)   or np.isnan(bmc5)) else np.nan
        rows.append({
            "concept_id":                    cid,
            "designed_target":               CONCEPT_TARGET_CLASS[cid],
            "vitb32_v4_auc":                 round(v1,   4) if not np.isnan(v1)   else np.nan,
            "biomedclip_v4_auc":             round(bmc4, 4) if not np.isnan(bmc4) else np.nan,
            "biomedclip_v5_auc":             round(bmc5, 4) if not np.isnan(bmc5) else np.nan,
            "delta_v4_to_v5_biomedclip":     d_v4v5,
            "delta_vitb32_to_v5_biomedclip": d_v1v5,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("delta_v4_to_v5_biomedclip", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def _build_per_group(per_concept: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        sub = per_concept[per_concept["designed_target"] == group]
        if sub.empty:
            continue
        rows.append({
            "designed_target":           group,
            "n_concepts":                len(sub),
            "vitb32_v4_mean":            round(float(sub["vitb32_v4_auc"].mean()),     4),
            "biomedclip_v4_mean":        round(float(sub["biomedclip_v4_auc"].mean()), 4),
            "biomedclip_v5_mean":        round(float(sub["biomedclip_v5_auc"].mean()), 4),
            "delta_v4_to_v5_biomedclip": round(
                float(sub["biomedclip_v5_auc"].mean()) - float(sub["biomedclip_v4_auc"].mean()), 4
            ),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("delta_v4_to_v5_biomedclip", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Summary formatter
# ---------------------------------------------------------------------------

def _fmt(per_concept: pd.DataFrame, per_group: pd.DataFrame, tmpl_mel: dict) -> str:
    buf = io.StringIO()

    def w(line: str = "") -> None:
        buf.write(line + "\n")

    def sgn(v: float) -> str:
        return "+" if v >= 0 else ""

    w("==== Three-way comparison: ViT-B/32 v4 vs BiomedCLIP v4 vs BiomedCLIP v5_biomedclip ====")
    w()
    w("Per disease group (mean AUC across designed-target concepts):")
    w()
    grp = per_group.set_index("designed_target")
    for group in GROUP_ORDER:
        if group not in grp.index:
            continue
        r = grp.loc[group]
        d = r["delta_v4_to_v5_biomedclip"]
        w(
            f"   {group:<7}: ViT32_v4 {r['vitb32_v4_mean']:.3f}  |  "
            f"BMC_v4 {r['biomedclip_v4_mean']:.3f}  |  "
            f"BMC_v5 {r['biomedclip_v5_mean']:.3f}  |  "
            f"Δ v4→v5: {sgn(d)}{d:.3f}  (n={int(r['n_concepts'])})"
        )

    w()
    w("Overall mean across all 24 concepts:")
    ov1   = float(per_concept["vitb32_v4_auc"].mean())
    obmc4 = float(per_concept["biomedclip_v4_auc"].mean())
    obmc5 = float(per_concept["biomedclip_v5_auc"].mean())
    od    = obmc5 - obmc4
    w(f"   ViT-B/32 v4         : {ov1:.3f}")
    w(f"   BiomedCLIP v4       : {obmc4:.3f}")
    w(f"   BiomedCLIP v5       : {obmc5:.3f}")
    w(f"   Δ v4→v5 (BiomedCLIP): {sgn(od)}{od:.3f}")

    w()
    mal = per_concept[per_concept["designed_target"].isin(MALIGNANT_CLASSES)]
    mv1   = float(mal["vitb32_v4_auc"].mean())
    mbmc4 = float(mal["biomedclip_v4_auc"].mean())
    mbmc5 = float(mal["biomedclip_v5_auc"].mean())
    md    = mbmc5 - mbmc4
    w(f"Malignant-only mean (mel + bcc + akiec; {len(mal)} concepts):")
    w(f"   ViT-B/32 v4         : {mv1:.3f}")
    w(f"   BiomedCLIP v4       : {mbmc4:.3f}")
    w(f"   BiomedCLIP v5       : {mbmc5:.3f}")
    w(f"   Δ v4→v5 (BiomedCLIP): {sgn(md)}{md:.3f}")

    w()
    mel_only = per_concept[per_concept["designed_target"] == "mel"]
    mmv1   = float(mel_only["vitb32_v4_auc"].mean())
    mmbmc4 = float(mel_only["biomedclip_v4_auc"].mean())
    mmbmc5 = float(mel_only["biomedclip_v5_auc"].mean())
    mmd    = mmbmc5 - mmbmc4
    w(f"[KEY METRIC] Melanoma-only mean (mel; {len(mel_only)} concepts):")
    w(f"   ViT-B/32 v4         : {mmv1:.3f}")
    w(f"   BiomedCLIP v4       : {mmbmc4:.3f}")
    w(f"   BiomedCLIP v5       : {mmbmc5:.3f}")
    w(f"   Δ v4→v5 (BiomedCLIP): {sgn(mmd)}{mmd:.3f}")

    w()
    w("Top 5 concepts by BiomedCLIP v4→v5 improvement:")
    for _, r in per_concept.head(5).iterrows():
        d = r["delta_v4_to_v5_biomedclip"]
        w(
            f"   [{r['concept_id']}]  designed={r['designed_target']}  "
            f"BMC_v4={r['biomedclip_v4_auc']:.3f}  BMC_v5={r['biomedclip_v5_auc']:.3f}  "
            f"Δ={sgn(d)}{d:.3f}"
        )

    w()
    regressed = per_concept[per_concept["delta_v4_to_v5_biomedclip"] < 0].tail(5).iloc[::-1]
    w("Top 5 concepts by BiomedCLIP v4→v5 regression (if any):")
    if regressed.empty:
        w("   (none)")
    else:
        for _, r in regressed.iterrows():
            d = r["delta_v4_to_v5_biomedclip"]
            w(
                f"   [{r['concept_id']}]  designed={r['designed_target']}  "
                f"BMC_v4={r['biomedclip_v4_auc']:.3f}  BMC_v5={r['biomedclip_v5_auc']:.3f}  "
                f"Δ={d:.3f}"
            )

    w()
    n_imp = int((per_concept["delta_v4_to_v5_biomedclip"] >= 0.02).sum())
    n_reg = int((per_concept["delta_v4_to_v5_biomedclip"] <= -0.02).sum())
    w(f"Concepts where BMC_v5 improves over BMC_v4 by ≥0.02 AUC: {n_imp} / {len(per_concept)}")
    w(f"Concepts where BMC_v5 regresses against BMC_v4 by ≥0.02 AUC: {n_reg} / {len(per_concept)}")

    w()
    w("Per-template results for melanoma-targeted concepts (BMC_v5 only):")
    for tmpl, style in TEMPLATE_STYLES.items():
        auc = tmpl_mel.get(tmpl, np.nan)
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else " n/a"
        w(f"   {tmpl} (\"{style}\"):  mean AUC on mel = {auc_str}")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLIT_NPZ.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_NPZ}\n"
            "Run scripts/analyze_prompts.py once to create the lesion split."
        )
    train_idx = np.load(SPLIT_NPZ)["train_idx"]

    # ------------------------------------------------------------------
    # Load all three encoders
    # ------------------------------------------------------------------
    print("Loading ViT-B/32 v4…")
    v1_scores, v1_labels, v1_meta, v1_cids, v1_ids = _load_encoder(VITB32_NPZ, train_idx)

    print("Loading BiomedCLIP v4…")
    bmc4_scores, bmc4_labels, bmc4_meta, bmc4_cids, bmc4_ids = _load_encoder(BMCV4_NPZ, train_idx)

    print("Loading BiomedCLIP v5…")
    bmc5_scores, bmc5_labels, bmc5_meta, bmc5_cids, bmc5_ids = _load_encoder(BMCV5_NPZ, train_idx)

    # ------------------------------------------------------------------
    # Assert identical image ordering across all three feature matrices
    # ------------------------------------------------------------------
    assert np.array_equal(v1_ids, bmc4_ids), (
        "image_ids mismatch: ViT-B/32 v4 vs BiomedCLIP v4\n"
        "The two NPZ files were not built from the same metadata CSV."
    )
    assert np.array_equal(v1_ids, bmc5_ids), (
        "image_ids mismatch: ViT-B/32 v4 vs BiomedCLIP v5\n"
        "The two NPZ files were not built from the same metadata CSV."
    )
    print(f"image_ids ordering verified across all three feature matrices ({len(v1_ids)} images)")

    # ------------------------------------------------------------------
    # Validate concept coverage
    # ------------------------------------------------------------------
    for tag, cids in [("ViT-B/32 v4", v1_cids), ("BiomedCLIP v4", bmc4_cids), ("BiomedCLIP v5", bmc5_cids)]:
        for cid in cids:
            if cid not in CONCEPT_TARGET_CLASS:
                raise KeyError(
                    f"Concept '{cid}' in {tag} NPZ has no entry in CONCEPT_TARGET_CLASS.\n"
                    "Update src/analysis/concept_targets.py before proceeding."
                )

    assert v1_cids == bmc4_cids == bmc5_cids, (
        "concept_id lists differ across NPZ files. All three must use the same prompt set."
    )
    concept_ids = v1_cids
    print(f"Concepts: {len(concept_ids)}  |  Training images: {len(train_idx):,}")

    # ------------------------------------------------------------------
    # Compute designed-target AUCs
    # ------------------------------------------------------------------
    print("Computing designed-target AUC for ViT-B/32 v4…")
    v1_aucs   = _compute_aucs(v1_scores,   v1_labels,   v1_meta,   concept_ids)

    print("Computing designed-target AUC for BiomedCLIP v4…")
    bmc4_aucs = _compute_aucs(bmc4_scores, bmc4_labels, bmc4_meta, concept_ids)

    print("Computing designed-target AUC for BiomedCLIP v5…")
    bmc5_aucs = _compute_aucs(bmc5_scores, bmc5_labels, bmc5_meta, concept_ids)

    # Per-template mel AUC breakdown for v5
    print("Computing per-template mel AUC for BiomedCLIP v5…")
    tmpl_mel = _per_template_mel_auc(bmc5_scores, bmc5_labels, bmc5_meta)

    # ------------------------------------------------------------------
    # Build DataFrames + summary
    # ------------------------------------------------------------------
    per_concept = _build_per_concept(concept_ids, v1_aucs, bmc4_aucs, bmc5_aucs)
    per_group   = _build_per_group(per_concept)
    summary     = _fmt(per_concept, per_group, tmpl_mel)

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
