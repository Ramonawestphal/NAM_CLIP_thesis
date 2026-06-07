"""
Generate thesis tables — v7 corrected pipeline (STEP 8).

Aggregates final results across all four conditions and writes:
  1. results/v7/thesis_tables/main_results.csv
       One row per condition: mean ± std for bal_acc, macro_f1, AUC, R_perp.
  2. results/v7/thesis_tables/comparison_vs_deprecated.csv
       Side-by-side of v7 results vs. pre-audit deprecated numbers (if available).
  3. results/v7/thesis_tables/feature_selection_summary.csv
       For sparsity conditions: n_active features, selected lambda, which features
       survived (mean group-norm across seeds).
  4. results/v7/thesis_tables/per_class_metrics.csv
       Per-class F1, recall, precision, AUC for the sparsity_conc condition.
  5. results/v7/methodology.md
       Methodology document for the v7 corrected pipeline.

Usage (from project root)
──────────────────────────
  python scripts/v7/generate_thesis_tables.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime, timezone

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

from scripts.v7._common import write_step_flag

# ── Constants ─────────────────────────────────────────────────────────────────
SEEDS      = [42, 43, 44, 45, 46]
CONDITIONS = ["plain_nam", "concurvity_only", "sparsity_only", "sparsity_conc"]

RESULTS_V7      = "results/v7"
TABLE_DIR       = "results/v7/thesis_tables"
OP_POINT_JSON   = "results/v7/operating_point.json"
DEPRECATED_DIR  = "results/DEPRECATED_pre_audit_fix"
STEP_N          = 8


def _fmt(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def load_condition_results(condition: str) -> pd.DataFrame | None:
    """Load aggregated_metrics.csv for a condition.  Returns None if missing."""
    path = os.path.join(RESULTS_V7, condition, "aggregated_metrics.csv")
    if not os.path.exists(path):
        print(f"  [WARN] Missing: {path}")
        return None
    df = pd.read_csv(path)
    return df


def load_feature_norms(condition: str, seeds: list[int]) -> pd.DataFrame | None:
    """Load and average feature_group_norms.csv across seeds for a condition."""
    dfs = []
    for seed in seeds:
        path = os.path.join(
            RESULTS_V7, condition, f"seed_{seed}", "feature_group_norms.csv"
        )
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["seed"] = seed
            dfs.append(df)
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    mean_norms = (
        combined.groupby("concept_name")["group_norm"]
        .mean()
        .reset_index()
        .rename(columns={"group_norm": "mean_group_norm"})
        .sort_values("mean_group_norm", ascending=False)
    )
    mean_norms["n_seeds_active"] = (
        combined[combined["group_norm"] >= 1e-4]
        .groupby("concept_name")["seed"].nunique()
        .reindex(mean_norms["concept_name"])
        .fillna(0)
        .astype(int)
        .values
    )
    return mean_norms


def main() -> None:
    os.makedirs(TABLE_DIR,  exist_ok=True)
    os.makedirs(RESULTS_V7, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"NAM v7 — Generate thesis tables (STEP {STEP_N})")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Output: {TABLE_DIR}/")
    print(f"{'='*65}\n")

    # ── 1. Main results table ─────────────────────────────────────────────────
    METRIC_KEYS = [
        "balanced_accuracy", "macro_f1", "weighted_f1",
        "top1_accuracy", "auc_ovr_weighted",
        "r_perp_val_at_best",
    ]
    main_rows = []

    for cond in CONDITIONS:
        df = load_condition_results(cond)
        if df is None:
            main_rows.append({"condition": cond, "status": "MISSING"})
            continue

        seed_df = df[df["seed"].apply(
            lambda x: str(x).isdigit() or (isinstance(x, float) and not pd.isna(x))
        )]
        # Filter to seed rows (not mean/std rows)
        try:
            seed_df = df[pd.to_numeric(df["seed"], errors="coerce").notna()]
        except Exception:
            seed_df = df

        row = {"condition": cond, "status": "OK", "n_seeds": len(seed_df)}
        for key in METRIC_KEYS:
            if key in df.columns:
                mean_row = df[df["seed"] == "mean"]
                std_row  = df[df["seed"] == "std"]
                if not mean_row.empty and not std_row.empty:
                    m = float(mean_row[key].iloc[0])
                    s = float(std_row[key].iloc[0])
                else:
                    m = float(seed_df[key].mean())
                    s = float(seed_df[key].std())   # ddof=1
                row[f"{key}_mean"] = round(m, 4)
                row[f"{key}_std"]  = round(s, 4)
                row[f"{key}_fmt"]  = _fmt(m, s)

        # n_selected (sparsity conditions)
        if "n_selected" in df.columns:
            try:
                ns_vals = pd.to_numeric(
                    seed_df["n_selected"] if "n_selected" in seed_df.columns
                    else seed_df["n_selected"],
                    errors="coerce"
                ).dropna()
                row["n_selected_mean"] = round(ns_vals.mean(), 1)
            except Exception:
                pass

        main_rows.append(row)

    main_df = pd.DataFrame(main_rows)
    main_path = os.path.join(TABLE_DIR, "main_results.csv")
    main_df.to_csv(main_path, index=False)
    print(f"  [OK] Main results → {main_path}")

    # Pretty print
    display_cols = ["condition"] + [
        f"{k}_fmt" for k in METRIC_KEYS if f"{k}_fmt" in main_df.columns
    ]
    available = [c for c in display_cols if c in main_df.columns]
    print("\n  Main results (mean ± std across 5 seeds):")
    print(main_df[available].to_string(index=False))

    # ── 2. Operating-point info ───────────────────────────────────────────────
    op_info = {}
    if os.path.exists(OP_POINT_JSON):
        with open(OP_POINT_JSON) as f:
            op_info = json.load(f)
        print(f"\n  Operating point (STEP 6):")
        print(f"    lambda        = {op_info.get('selected_lambda', 'N/A')}")
        print(f"    n_active      = {op_info.get('selected_n_active', 'N/A')}")
        print(f"    val_balacc    = {op_info.get('selected_val_balacc', 'N/A'):.4f}")
    else:
        print(f"  [WARN] operating_point.json not found at {OP_POINT_JSON}")

    # ── 3. Feature selection summary ──────────────────────────────────────────
    feature_rows = []
    for cond in ["sparsity_only", "sparsity_conc"]:
        norms_df = load_feature_norms(cond, SEEDS)
        if norms_df is None:
            continue
        op_lambda = op_info.get("selected_lambda", None) if cond == "sparsity_conc" else None
        norms_df["condition"] = cond
        norms_df["operating_lambda"] = op_lambda
        feature_rows.append(norms_df)

    if feature_rows:
        feat_df   = pd.concat(feature_rows, ignore_index=True)
        feat_path = os.path.join(TABLE_DIR, "feature_selection_summary.csv")
        feat_df.to_csv(feat_path, index=False)
        print(f"\n  [OK] Feature selection → {feat_path}")

        for cond in ["sparsity_only", "sparsity_conc"]:
            sub = feat_df[feat_df["condition"] == cond]
            if sub.empty:
                continue
            n_consistently_active = (sub["n_seeds_active"] == len(SEEDS)).sum()
            print(f"  {cond}: {n_consistently_active}/{len(sub)} features "
                  f"active in all {len(SEEDS)} seeds")
    else:
        print("  [WARN] No feature norm data found for sparsity conditions.")

    # ── 4. Per-class metrics for sparsity_conc ───────────────────────────────
    per_cls_path_src = os.path.join(RESULTS_V7, "sparsity_conc", "per_class_metrics.csv")
    if os.path.exists(per_cls_path_src):
        per_cls_df   = pd.read_csv(per_cls_path_src, index_col=0)
        per_cls_dest = os.path.join(TABLE_DIR, "per_class_metrics.csv")
        per_cls_df.to_csv(per_cls_dest)
        print(f"\n  [OK] Per-class metrics → {per_cls_dest}")
    else:
        print(f"\n  [WARN] per_class_metrics.csv not found for sparsity_conc")

    # ── 5. Comparison vs deprecated ──────────────────────────────────────────
    depr_tables = os.path.join(DEPRECATED_DIR, "thesis_tables")
    depr_csv    = os.path.join(depr_tables, "nam_v6_final_metrics.csv")
    if os.path.exists(depr_csv):
        depr_df = pd.read_csv(depr_csv)
        print(f"\n  [INFO] Found deprecated results at {depr_csv}")

        comp_rows = []
        for cond in CONDITIONS:
            v7_row_mask = main_df["condition"] == cond
            if not v7_row_mask.any():
                continue
            v7_row = main_df[v7_row_mask].iloc[0]

            comp_row = {"condition": cond}
            for key in ["balanced_accuracy", "macro_f1", "auc_ovr_weighted"]:
                v7_key = f"{key}_mean"
                if v7_key in v7_row:
                    comp_row[f"v7_{key}"] = v7_row[v7_key]

            # Try to match deprecated row
            depr_match = depr_df[depr_df.get("condition", depr_df.columns[0]) == cond]
            for key in ["balanced_accuracy", "macro_f1", "auc_ovr_weighted"]:
                for col in depr_df.columns:
                    if key in col.lower() and "mean" in col.lower():
                        if not depr_match.empty:
                            comp_row[f"depr_{key}"] = float(depr_match[col].iloc[0])
                        break

            comp_rows.append(comp_row)

        if comp_rows:
            comp_df   = pd.DataFrame(comp_rows)
            comp_path = os.path.join(TABLE_DIR, "comparison_vs_deprecated.csv")
            comp_df.to_csv(comp_path, index=False)
            print(f"  [OK] Comparison table → {comp_path}")
    else:
        print(f"\n  [INFO] No deprecated thesis tables found; skipping comparison.")
        # Write a placeholder
        comp_path = os.path.join(TABLE_DIR, "comparison_vs_deprecated.csv")
        placeholder = pd.DataFrame(
            main_rows, columns=["condition", "status"]
            + [f"{k}_mean" for k in METRIC_KEYS if f"{k}_mean" in main_df.columns]
        )
        placeholder.to_csv(comp_path, index=False)
        print(f"  [INFO] Placeholder comparison → {comp_path}")

    # ── 6. Methodology document ───────────────────────────────────────────────
    _write_methodology_doc(main_rows, op_info)

    print(f"\n{'='*65}")
    print(f"All tables written to {TABLE_DIR}/")
    print(f"{'='*65}")

    write_step_flag(RESULTS_V7, STEP_N)


def _write_methodology_doc(main_rows: list, op_info: dict) -> None:
    """Write results/v7/methodology.md."""
    meth_path = os.path.join(RESULTS_V7, "methodology.md")
    now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cond_table = "| Condition | Bal. Acc (mean±std) | Macro F1 | AUC (OvR wtd) |\n"
    cond_table += "|-----------|-------------------|----------|---------------|\n"
    for row in main_rows:
        if row.get("status") != "OK":
            cond_table += f"| {row['condition']} | MISSING | – | – |\n"
            continue
        ba = row.get("balanced_accuracy_fmt", "–")
        f1 = row.get("macro_f1_fmt", "–")
        au = row.get("auc_ovr_weighted_fmt", "–")
        cond_table += f"| {row['condition']} | {ba} | {f1} | {au} |\n"

    op_text = ""
    if op_info:
        op_text = (
            f"\n### Operating point (Issue 11 fix)\n\n"
            f"Selected by `select_operating_point.py` using:\n"
            f"- Acceptable if `val_balacc >= dense_val_balacc - 0.02`\n"
            f"- Selected: minimum `n_active` among acceptable; tie-break: largest lambda\n\n"
            f"| Parameter | Value |\n"
            f"|-----------|-------|\n"
            f"| Selected lambda | {op_info.get('selected_lambda', 'N/A'):.4e} |\n"
            f"| n_active features | {op_info.get('selected_n_active', 'N/A')} |\n"
            f"| val_balacc at selection | {op_info.get('selected_val_balacc', 0):.4f} |\n"
            f"| dense val_balacc | {op_info.get('dense_val_balacc', 0):.4f} |\n"
            f"| delta | {op_info.get('selected_val_balacc', 0) - op_info.get('dense_val_balacc', 0):+.4f} |\n"
        )

    doc = f"""# NAM v7 — Corrected Pipeline Methodology

Generated: {now}

## Overview

This document describes the corrected NAM training pipeline for the thesis.
All results under `results/v7/` were produced by scripts in `scripts/v7/`,
which fix the protocol issues identified in `results/methodology_audit.md`.

## Issues Fixed

| Issue | Severity | Description | Fix |
|-------|----------|-------------|-----|
| 1 | Critical | Architecture selection by test balanced accuracy | 5-fold GroupKFold CV (`architecture_search_cv.py`) |
| 2 | Medium | Stage 1 sweep stored test metrics | CV-only metrics recorded |
| 3 | Low | Missing `random.seed()` call | `set_all_seeds()` in `_common.py` |
| 4 | Medium | Warm-start checkpoint = end-of-patience state | Best-within-step saved and restored |
| 5 | Low | Inconsistent std denominator | pandas `.std()` (ddof=1) throughout |
| 7 | Low | Single global scaler, path inconsistency | Per-seed scaler saved to `seed_N/scaler.pkl` |
| 8 | Low | Missing CUDA determinism | `cudnn.deterministic=True`, `manual_seed_all` |
| 9 | Medium | No concurvity warm-up | `warmup_epochs = max(1, int(0.05 * MAX_EPOCHS))` |
| 10 | Low | weight_decay potentially zeroed | Kept at config value throughout |
| 11 | Medium | Informal operating point selection | Codified 0.02 tolerance rule |

## Execution Order

| Step | Script | Description |
|------|--------|-------------|
| 1 | `architecture_search_cv.py` | 5-fold CV architecture search |
| 2 | `train_final.py --condition plain_nam` | 5-seed plain NAM |
| 3 | `run_concurvity_sweep.py` | 10-value lambda_c sweep (seed=42) |
| 4 | `train_final.py --condition concurvity_only` | 5-seed concurvity NAM |
| 5 | `run_sparsity_sweep.py` | Warm-start sparsity path (seed=42) |
| 6 | `select_operating_point.py` | Codified operating-point selection |
| 7 | `train_final.py --condition sparsity_conc` | 5-seed sparsity+concurvity NAM |
| 8 | `generate_thesis_tables.py` | Aggregate and write tables |

## Architecture Selection (Issue 1 fix)

Architecture selected by `architecture_search_cv.py` using 5-fold GroupKFold
cross-validation on `train_idx` only.  The test set was NOT touched at any
point during selection.  Winner recorded in `results/v7/architecture_search_cv/winner.json`.

## Concurvity Warm-up (Issue 9 fix)

Following Siems et al. (2023) Appendix C.1, the effective concurvity penalty
is set to zero for the first `warmup_epochs = max(1, int(0.05 * MAX_EPOCHS))`
epochs (= 5 for MAX_EPOCHS=100).

## Warm-start Regularization Path (Issue 4 fix)

Within each lambda step of the regularization path, the checkpoint with the
**lowest validation loss** (not the end-of-patience state) is saved in memory
and restored before advancing to lambda+1.  This ensures that the warm-start
initialization for each step is the best-performing state of the previous step.

{op_text}
## Final Results

{cond_table}

_All metrics are mean ± std (ddof=1) across 5 seeds: 42, 43, 44, 45, 46._
_Test set evaluated once per seed after training; results are test-set metrics._

## Data

- Features: `data/features/biomedclip/ham10000_concept_scores_v6.npz` (10015 × 24)
- Splits: `data/splits/train_test_lesion_split.npz` (train=8020, test=1995)
- Splits are fixed and lesion-disjoint (GroupShuffleSplit, random_state=42).

## Reproducibility

All scripts call `set_all_seeds(seed)` which seeds:
- `torch.manual_seed`
- `torch.cuda.manual_seed_all`
- `numpy.random.seed`
- `random.seed`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
"""

    with open(meth_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"\n  [OK] Methodology doc → {meth_path}")


if __name__ == "__main__":
    main()
