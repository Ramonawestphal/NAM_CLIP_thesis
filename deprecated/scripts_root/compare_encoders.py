"""Side-by-side comparison of ViT-B/32 (CLIP) vs BiomedCLIP diagnostic outputs.

Requires that both analysis scripts have been run first (v4 prompts):
    python scripts/analyze_prompts.py            # ViT-B/32, v4 scores
    python scripts/analyze_prompts_biomedclip.py # BiomedCLIP, v4 scores

Intended-class definition
--------------------------
For each concept, "intended class" is the class where ViT-B/32's best
template peaks (auc_argmax_class from the ViT-B/32 OvR AUC CSV). BiomedCLIP
is then evaluated on that *same* fixed class so the delta is a fair
head-to-head comparison, not a comparison of each encoder's own best class.

Outputs (all under reports/encoder_comparison/):
    paired_contrasts_vitb32_vs_biomedclip.csv
    intended_class_auc_vitb32_vs_biomedclip.csv

Run from project root:
    python scripts/compare_encoders.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
V1_REPORT_DIR  = _ROOT / "reports/prompt_analysis"
BMC_REPORT_DIR = _ROOT / "reports/prompt_analysis/biomedclip"
OUTPUT_DIR     = _ROOT / "reports/encoder_comparison"

PAIRED_CONTRASTS_CSV   = OUTPUT_DIR / "paired_contrasts_vitb32_vs_biomedclip.csv"
INTENDED_CLASS_AUC_CSV = OUTPUT_DIR / "intended_class_auc_vitb32_vs_biomedclip.csv"
# ---------------------------------------------------------------------------


def _require(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the corresponding analysis script first."
        )
    return pd.read_csv(path)


def compare_paired_contrasts(
    v1_dir: pathlib.Path,
    bmc_dir: pathlib.Path,
) -> pd.DataFrame:
    """Merge ViT-B/32 and BiomedCLIP paired contrast results side-by-side."""
    df_v1  = _require(v1_dir  / "paired_contrasts.csv")
    df_bmc = _require(bmc_dir / "paired_contrasts.csv")

    key_cols = ["positive_concept", "negative_concept", "target_class"]
    merged = df_v1[key_cols + ["diff", "status"]].merge(
        df_bmc[key_cols + ["diff", "status"]],
        on=key_cols,
        suffixes=("_vitb32", "_biomedclip"),
    )
    merged["delta"] = (merged["diff_biomedclip"] - merged["diff_vitb32"]).round(5)
    return merged


def compare_intended_class_auc(
    v1_dir: pathlib.Path,
    bmc_dir: pathlib.Path,
) -> pd.DataFrame:
    """Per-concept intended-class AUC comparison.

    Intended class is fixed to ViT-B/32's argmax class for the best template
    of each concept. BiomedCLIP AUC is reported on that same class.
    """
    tc_v1  = _require(v1_dir  / "template_comparison.csv")  # best_auc, best_template per concept
    tc_bmc = _require(bmc_dir / "template_comparison.csv")
    auc_v1  = _require(v1_dir  / "prompt_ovr_auc.csv")
    auc_bmc = _require(bmc_dir / "prompt_ovr_auc.csv")

    rows = []
    for _, v1_row in tc_v1.iterrows():
        concept_id   = v1_row["concept_id"]
        best_tmpl_v1 = v1_row["best_template"]
        vitb32_auc   = float(v1_row["best_auc"])

        # Intended class: the class where ViT-B/32's best template peaks
        match = auc_v1[
            (auc_v1["concept_id"] == concept_id) &
            (auc_v1["template"]   == best_tmpl_v1)
        ]
        if match.empty or pd.isna(match["auc_argmax_class"].iloc[0]):
            intended_class = None
            bmc_auc_on_class = np.nan
        else:
            intended_class = match["auc_argmax_class"].iloc[0]
            # BiomedCLIP AUC on that same class, best template
            bmc_concept = auc_bmc[auc_bmc["concept_id"] == concept_id]
            col = f"auc_{intended_class}"
            if col in bmc_concept.columns:
                bmc_auc_on_class = float(bmc_concept[col].max())
            else:
                bmc_auc_on_class = np.nan

        # BiomedCLIP's own best AUC for reference
        bmc_row = tc_bmc[tc_bmc["concept_id"] == concept_id]
        bmc_best_auc = float(bmc_row["best_auc"].iloc[0]) if len(bmc_row) else np.nan

        delta = bmc_auc_on_class - vitb32_auc if not np.isnan(bmc_auc_on_class) else np.nan

        rows.append({
            "concept_id":        concept_id,
            "intended_class":    intended_class,
            "vitb32_auc":        round(vitb32_auc,        4),
            "biomedclip_auc":    round(bmc_auc_on_class,  4) if not np.isnan(bmc_auc_on_class) else np.nan,
            "biomedclip_auc_max": round(bmc_best_auc,     4) if not np.isnan(bmc_best_auc) else np.nan,
            "delta":             round(delta, 4)              if not np.isnan(delta) else np.nan,
        })

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paired contrasts
    # ------------------------------------------------------------------
    df_contrasts = compare_paired_contrasts(V1_REPORT_DIR, BMC_REPORT_DIR)
    df_contrasts.to_csv(PAIRED_CONTRASTS_CSV, index=False)

    # ------------------------------------------------------------------
    # Intended-class AUC
    # ------------------------------------------------------------------
    df_auc = compare_intended_class_auc(V1_REPORT_DIR, BMC_REPORT_DIR)
    df_auc.to_csv(INTENDED_CLASS_AUC_CSV, index=False)

    # ------------------------------------------------------------------
    # Print side-by-side tables
    # ------------------------------------------------------------------
    col_w = 52
    print("\n─── Paired contrasts ───────────────────────────────────────────────────")
    header = (
        f"{'Contrast':<{col_w}}"
        f"{'vitb32':>10}  {'':>6}  {'biomedclip':>10}  {'':>6}  {'Δ':>9}"
    )
    print(header)
    print("─" * len(header))
    for _, r in df_contrasts.iterrows():
        contrast = (
            f"{r['positive_concept']} > {r['negative_concept']} on {r['target_class']}"
        )
        print(
            f"{contrast:<{col_w}}"
            f"{r['diff_vitb32']:>+10.4f}  {r['status_vitb32']:>6}  "
            f"{r['diff_biomedclip']:>+10.4f}  {r['status_biomedclip']:>6}  "
            f"{r['delta']:>+9.4f}"
        )

    print("\n─── Intended-class AUC per concept ─────────────────────────────────────")
    hdr2 = f"{'concept_id':<28} {'intended_class':<14} {'vitb32':>8} {'bmc_fixed':>10} {'bmc_best':>9} {'Δ':>8}"
    print(hdr2)
    print("─" * len(hdr2))
    for _, r in df_auc.sort_values("delta", ascending=False, na_position="last").iterrows():
        delta_str = f"{r['delta']:>+8.4f}" if not pd.isna(r["delta"]) else "     n/a"
        print(
            f"{r['concept_id']:<28} {str(r['intended_class']):<14} "
            f"{r['vitb32_auc']:>8.4f} {r['biomedclip_auc']:>10.4f} "
            f"{r['biomedclip_auc_max']:>9.4f} {delta_str}"
        )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    n_pass_v1  = int((df_contrasts["status_vitb32"]    == "PASS").sum())
    n_pass_bmc = int((df_contrasts["status_biomedclip"] == "PASS").sum())
    n_total    = len(df_contrasts)

    valid_delta = df_auc["delta"].dropna()
    mean_v1  = float(df_auc["vitb32_auc"].mean())
    mean_bmc = float(df_auc["biomedclip_auc"].mean())
    n_improved = int((valid_delta >= 0.02).sum())
    n_regressed = int((valid_delta <= -0.02).sum())

    top5_improved  = df_auc.nlargest(5, "delta")[["concept_id", "intended_class", "delta"]]
    top5_regressed = df_auc.nsmallest(5, "delta")[["concept_id", "intended_class", "delta"]]

    print("\n─── Summary ─────────────────────────────────────────────────────────────")
    print(f"Paired contrasts passed  : ViT-B/32 {n_pass_v1}/{n_total}  |  BiomedCLIP {n_pass_bmc}/{n_total}")
    print(f"Mean intended-class AUC  : ViT-B/32 {mean_v1:.4f}  |  BiomedCLIP {mean_bmc:.4f}")
    print(f"BiomedCLIP improves by ≥0.02 AUC : {n_improved} / {len(df_auc)} concepts")
    print(f"BiomedCLIP regresses by ≥0.02 AUC: {n_regressed} / {len(df_auc)} concepts")

    print("\nTop 5 concepts — BiomedCLIP improvement:")
    for _, r in top5_improved.iterrows():
        print(f"  {r['concept_id']:<28} {str(r['intended_class']):<10}  Δ={r['delta']:>+.4f}")

    if n_regressed > 0:
        print("\nTop 5 concepts — BiomedCLIP regression:")
        for _, r in top5_regressed[top5_regressed["delta"] < 0].iterrows():
            print(f"  {r['concept_id']:<28} {str(r['intended_class']):<10}  Δ={r['delta']:>+.4f}")

    print("\n─────────────────────────────────────────────────────────────────────────")
    print(f"Saved: {PAIRED_CONTRASTS_CSV}")
    print(f"Saved: {INTENDED_CLASS_AUC_CSV}")


if __name__ == "__main__":
    main()
