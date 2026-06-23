"""
Generate comparison.md for the warm-up diagnostic experiment.

Reads results from:
  results/HAM10000/diagnostic_warmup/warmup_0/   (SETTING A — no warm-up)
  results/HAM10000/diagnostic_warmup/warmup_5/   (SETTING B — 5-epoch warm-up, v7 default)
  results/HAM10000/diagnostic_warmup/warmup_2/   (SETTING C — 2-epoch warm-up)

Writes:
  results/HAM10000/diagnostic_warmup/comparison.md

Usage (from project root):
    python scripts/HAM10000/diagnostic_warmup_report.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

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

SETTINGS = {
    "A (warmup=0)": "results/HAM10000/diagnostic_warmup/warmup_0",
    "B (warmup=5)": "results/HAM10000/diagnostic_warmup/warmup_5",
    "C (warmup=2)": "results/HAM10000/diagnostic_warmup/warmup_2",
}
SEEDS     = [42, 43, 44, 45, 46]
OUT_DIR   = "results/HAM10000/diagnostic_warmup"
OUT_MD    = "results/HAM10000/diagnostic_warmup/comparison.md"
R_PERP_TOL = 0.02   # tolerance for verdict: R_perp "within 0.02"


def load_setting(out_dir: str) -> dict | None:
    """Load aggregated_metrics.csv + per-seed training logs for one setting."""
    agg_path = os.path.join(out_dir, "aggregated_metrics.csv")
    cfg_path = os.path.join(out_dir, "run_config.json")
    if not os.path.exists(agg_path):
        return None

    agg_df = pd.read_csv(agg_path)
    # Extract mean/std rows
    mean_row = agg_df[agg_df["seed"] == "mean"]
    std_row  = agg_df[agg_df["seed"] == "std"]
    seed_df  = agg_df[agg_df["seed"].apply(
        lambda x: str(x).lstrip("-").replace(".", "").isdigit()
    )]

    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)

    # Per-seed training logs (for best_epoch / total_epochs)
    seed_logs: dict[int, pd.DataFrame] = {}
    for s in SEEDS:
        log_p = os.path.join(out_dir, f"seed_{s}", "training_log.csv")
        if os.path.exists(log_p):
            seed_logs[s] = pd.read_csv(log_p)

    return {
        "agg_df":    agg_df,
        "mean_row":  mean_row,
        "std_row":   std_row,
        "seed_df":   seed_df,
        "seed_logs": seed_logs,
        "cfg":       cfg,
        "out_dir":   out_dir,
    }


def get_metric(data: dict, key: str, stat: str = "mean") -> float:
    row = data["mean_row"] if stat == "mean" else data["std_row"]
    if key in row.columns and not row.empty:
        return float(row[key].iloc[0])
    # Fallback: compute from seed_df
    if key in data["seed_df"].columns:
        vals = pd.to_numeric(data["seed_df"][key], errors="coerce").dropna()
        return float(vals.mean() if stat == "mean" else vals.std())
    return float("nan")


def per_seed_best_epoch(data: dict) -> dict[int, int]:
    """Return {seed: best_epoch} from agg_df or training logs."""
    result = {}
    seed_df = data["seed_df"]
    if "best_epoch" in seed_df.columns:
        for _, row in seed_df.iterrows():
            try:
                s = int(float(row["seed"]))
                result[s] = int(row["best_epoch"])
            except (ValueError, TypeError):
                pass
    # Fill missing from logs
    for s, log_df in data["seed_logs"].items():
        if s not in result:
            result[s] = int(log_df["val_balanced_acc"].idxmax()) + 1
    return result


def per_seed_total_epochs(data: dict) -> dict[int, int]:
    result = {}
    seed_df = data["seed_df"]
    if "total_epochs" in seed_df.columns:
        for _, row in seed_df.iterrows():
            try:
                s = int(float(row["seed"]))
                result[s] = int(row["total_epochs"])
            except (ValueError, TypeError):
                pass
    for s, log_df in data["seed_logs"].items():
        if s not in result:
            result[s] = len(log_df)
    return result


def fmt(mean: float, std: float, decimals: int = 4) -> str:
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def compute_verdict(
    data_a: dict, data_b: dict, data_c: dict
) -> tuple[str, str]:
    """Return (verdict_string, recommended_setting)."""
    std_a = get_metric(data_a, "balanced_accuracy", "std")
    std_b = get_metric(data_b, "balanced_accuracy", "std")
    std_c = get_metric(data_c, "balanced_accuracy", "std")

    rperp_a = get_metric(data_a, "r_perp_val_at_best", "mean")
    rperp_b = get_metric(data_b, "r_perp_val_at_best", "mean")
    rperp_c = get_metric(data_c, "r_perp_val_at_best", "mean")

    balacc_a = get_metric(data_a, "balanced_accuracy", "mean")
    balacc_b = get_metric(data_b, "balanced_accuracy", "mean")
    balacc_c = get_metric(data_c, "balanced_accuracy", "mean")

    lines   = []
    verdict = "Warm-up neutral"

    # Primary comparison: A vs B
    b_lower_std  = std_b  < std_a
    a_lower_std  = std_a  < std_b
    ab_rperp_sim = abs(rperp_a - rperp_b) <= R_PERP_TOL

    if b_lower_std and ab_rperp_sim:
        verdict = "Warm-up helping"
        lines.append(
            f"SETTING B (warmup=5) has lower test bal_acc std ({std_b:.4f} < {std_a:.4f}) "
            f"and R_perp within {R_PERP_TOL} of SETTING A "
            f"(|{rperp_b:.4f} - {rperp_a:.4f}| = {abs(rperp_b - rperp_a):.4f})."
        )
    elif a_lower_std and ab_rperp_sim:
        verdict = "Warm-up hurting"
        lines.append(
            f"SETTING A (warmup=0) has lower test bal_acc std ({std_a:.4f} < {std_b:.4f}) "
            f"and R_perp within {R_PERP_TOL} of SETTING B "
            f"(|{rperp_a:.4f} - {rperp_b:.4f}| = {abs(rperp_a - rperp_b):.4f})."
        )
    else:
        lines.append(
            f"No clear advantage: std_A={std_a:.4f} vs std_B={std_b:.4f}, "
            f"|R_perp_A - R_perp_B| = {abs(rperp_a - rperp_b):.4f} "
            f"(tolerance = {R_PERP_TOL})."
        )

    # Check SETTING C
    c_better_balacc = (balacc_c > balacc_a + 0.001) and (balacc_c > balacc_b + 0.001)
    c_lower_std     = (std_c < std_a - 0.001) or (std_c < std_b - 0.001)
    if c_better_balacc or c_lower_std:
        lines.append(
            f"SETTING C (warmup=2) shows improvement: "
            f"mean bal_acc={balacc_c:.4f} vs A={balacc_a:.4f}, B={balacc_b:.4f}; "
            f"std={std_c:.4f} vs A={std_a:.4f}, B={std_b:.4f}. "
            f"Worth considering as an alternative."
        )

    # Recommended setting
    if verdict == "Warm-up helping":
        rec = "SETTING B (warmup_epochs=5, current v7 default)"
    elif verdict == "Warm-up hurting":
        rec = "SETTING A (warmup_epochs=0, no warm-up)"
    else:
        rec = "SETTING B (warmup_epochs=5, current v7 default) — no clear evidence to change"

    if c_better_balacc or c_lower_std:
        rec += "; SETTING C (warmup_epochs=2) is a secondary candidate"

    return verdict, rec, "\n".join(f"- {l}" for l in lines)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading diagnostic results ...")
    loaded = {}
    for label, out_dir in SETTINGS.items():
        d = load_setting(out_dir)
        if d is None:
            print(f"  [WARN] {label}: no aggregated_metrics.csv found at {out_dir}")
        else:
            cfg = d["cfg"]
            print(f"  {label}: warmup_epochs={cfg.get('warmup_epochs', '?')}, "
                  f"seeds={cfg.get('seeds', '?')}, lambda_c={cfg.get('concurvity_lambda', '?')}")
        loaded[label] = d

    missing = [k for k, v in loaded.items() if v is None]
    if missing:
        print(f"\nERROR: Missing results for: {missing}")
        print("Run all three settings before generating the report.")
        sys.exit(1)

    data_a = loaded["A (warmup=0)"]
    data_b = loaded["B (warmup=5)"]
    data_c = loaded["C (warmup=2)"]

    # ── Build metric summary ───────────────────────────────────────────────────
    METRICS = [
        ("balanced_accuracy",     "Test bal. acc"),
        ("macro_f1",              "Test macro F1"),
        ("auc_ovr_weighted",      "Test AUC (OvR wtd)"),
        ("best_val_balacc",       "Val bal. acc (at best ckpt)"),
        ("r_perp_val_at_best",    "Val R_perp (at best ckpt)"),
    ]

    rows_main = []
    for key, label in METRICS:
        row = {"Metric": label}
        for lbl, data in loaded.items():
            m = get_metric(data, key, "mean")
            s = get_metric(data, key, "std")
            row[lbl] = fmt(m, s)
        rows_main.append(row)

    # best_epoch and total_epochs from seed_df
    for key, label in [("best_epoch", "Mean best_epoch"), ("total_epochs", "Mean total_epochs")]:
        row = {"Metric": label}
        for lbl, data in loaded.items():
            m = get_metric(data, key, "mean")
            s = get_metric(data, key, "std")
            if np.isnan(m):
                # compute from logs
                if key == "best_epoch":
                    vals = list(per_seed_best_epoch(data).values())
                else:
                    vals = list(per_seed_total_epochs(data).values())
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[lbl] = fmt(m, s, decimals=1)
        rows_main.append(row)

    main_df = pd.DataFrame(rows_main)

    # ── Per-seed best_epoch table ──────────────────────────────────────────────
    best_epoch_rows = []
    for s in SEEDS:
        row = {"Seed": s}
        for lbl, data in loaded.items():
            be = per_seed_best_epoch(data).get(s, "?")
            row[lbl] = be
        best_epoch_rows.append(row)
    best_epoch_df = pd.DataFrame(best_epoch_rows)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict, rec, verdict_detail = compute_verdict(data_a, data_b, data_c)

    # ── Write comparison.md ───────────────────────────────────────────────────
    def df_to_md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep    = "| " + " | ".join("---" for _ in cols) + " |"
        rows   = []
        for _, r in df.iterrows():
            rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join([header, sep] + rows)

    cfg_b = data_b["cfg"]
    doc = f"""# Concurvity Warm-up Diagnostic — Comparison Report

**Date generated:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

## Experiment design

All three settings share identical hyperparameters; only `warmup_epochs` varies.

| Parameter | Value |
|-----------|-------|
| Condition | concurvity_only |
| lambda_c | {cfg_b.get('concurvity_lambda', 3.0)} |
| Architecture | hidden={cfg_b.get('hidden_dims', [64,32])}, dropout={cfg_b.get('dropout', 0.1)}, wd={cfg_b.get('weight_decay', 1e-4):.0e} |
| Seeds | {SEEDS} |
| Max epochs | {cfg_b.get('max_epochs', 100)} |
| Patience | {cfg_b.get('patience', 15)} |
| Fix A2 | post-warmup checkpoint reset active for warmup_epochs > 0 |

## Setting descriptions

| Setting | warmup_epochs | Description |
|---------|--------------|-------------|
| A | 0 | Concurvity active from epoch 1; no reset. Matches deprecated v6 protocol. |
| B | 5 | Current v7 default (5% of max_epochs). Matches Siems et al. 2023 App. C.1. |
| C | 2 | Short warm-up: stabilise initialisation without dominating early training. |

## 1. Main results (mean +/- std across 5 seeds)

{df_to_md(main_df)}

## 2. Per-seed best epoch

{df_to_md(best_epoch_df)}

## 3. Diagnostic verdict

**Verdict: {verdict}**

{verdict_detail}

Criteria applied:
- "Warm-up helping": SETTING B has lower test bal_acc std AND R_perp within {R_PERP_TOL} of SETTING A
- "Warm-up hurting": SETTING A has lower test bal_acc std AND R_perp within {R_PERP_TOL} of SETTING B
- "Warm-up neutral": neither criterion met

## 4. Recommended setting

**{rec}**

> This recommendation is based solely on val/test metrics from this diagnostic run.
> The user decides whether to adopt this recommendation for the final pipeline.
> Do NOT treat the test metrics above as final thesis numbers.
"""

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"\ncomparison.md written to {OUT_MD}")
    print(f"Verdict: {verdict}")
    print(f"Recommended setting: {rec}")


if __name__ == "__main__":
    main()
