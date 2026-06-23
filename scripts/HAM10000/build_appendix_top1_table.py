"""
Build appendix top-1 accuracy table for thesis.

NAM — Plain NAM and Concurvity only: train_final aggregated_metrics.csv.
NAM — Sparsity conditions: ANEC K=10 sweep operating point (train_final STEP 7 not run).
ML baselines: aggregate_summary.csv + per-seed CSVs from run_ml_baselines.py.

Usage (from project root):
    python scripts/HAM10000/build_appendix_top1_table.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

OUT = _ROOT / "results" / "v7" / "thesis_tables"

DISPLAY = {
    "plain_nam": "Plain NAM",
    "concurvity_only": "Concurvity only",
    "sparsity_only": "Sparsity only",
    "sparsity_conc": "Sparsity + concurvity",
    "sparsity_concurvity": "Sparsity + concurvity",
}

BASELINE_DISPLAY = {
    "logreg": "Logistic regression",
    "xgboost": "XGBoost",
    "random_forest": "Random forest",
}

BASELINE_PER_SEED = {
    ("HAM10000", "xgboost"): "results/baselines_ml/xgboost_per_seed.csv",
    ("HAM10000", "random_forest"): "results/baselines_ml/rf_per_seed.csv",
    ("Chest X-ray", "xgboost"): "results/chestxray/baselines_ml/xgboost_per_seed.csv",
    ("Chest X-ray", "random_forest"): "results/chestxray/baselines_ml/rf_per_seed.csv",
    ("Chest X-ray", "logreg"): "results/chestxray/baselines_ml/lr/per_seed_metrics.csv",
}


def load_train_final(rel_path: str, dataset: str, condition_key: str) -> dict:
    df = pd.read_csv(_ROOT / rel_path)
    seed_rows = df[pd.to_numeric(df["seed"], errors="coerce").notna()].copy()
    mean = float(df.loc[df["seed"] == "mean", "top1_accuracy"].iloc[0])
    std = float(df.loc[df["seed"] == "std", "top1_accuracy"].iloc[0])
    return {
        "model_type": "NAM",
        "dataset": dataset,
        "condition": DISPLAY[condition_key],
        "condition_key": condition_key,
        "source": "train_final (5-seed protocol)",
        "top1_mean": mean,
        "top1_std": std,
        "top1_fmt": f"{mean:.3f} ± {std:.3f}",
        "seeds": {int(r.seed): float(r.top1_accuracy) for _, r in seed_rows.iterrows()},
        "notes": "",
    }


def load_anec_k10(
    by_seed_path: str,
    agg_path: str,
    dataset: str,
    condition_key: str,
    anec_name: str | None = None,
    K: int = 10,
) -> dict:
    anec_name = anec_name or condition_key
    df = pd.read_csv(_ROOT / by_seed_path)
    k_mask = pd.to_numeric(df["target_K"], errors="coerce") == K
    sub = df[(df["condition"] == anec_name) & k_mask]
    if sub.empty:
        raise RuntimeError(f"No ANEC K={K} rows for {anec_name}")
    agg = pd.read_csv(_ROOT / agg_path)
    agg_k = pd.to_numeric(agg["target_K"], errors="coerce") == K
    arow = agg[(agg["condition"] == anec_name) & agg_k].iloc[0]
    mean = float(arow["mean_test_top1_acc"])
    std = float(arow["std_test_top1_acc"])
    return {
        "model_type": "NAM",
        "dataset": dataset,
        "condition": DISPLAY[condition_key],
        "condition_key": condition_key,
        "source": "ANEC sweep, K=10 operating point",
        "top1_mean": mean,
        "top1_std": std,
        "top1_fmt": f"{mean:.3f} ± {std:.3f}",
        "seeds": {int(r.seed): float(r.test_top1_acc) for _, r in sub.iterrows()},
        "notes": "",
    }


def load_baseline(dataset: str, model_key: str, agg_rel: str) -> dict:
    agg = pd.read_csv(_ROOT / agg_rel)
    row = agg[agg["model"] == model_key].iloc[0]
    mean = float(row["top1_acc_mean"])
    std = float(row["top1_acc_std"])
    notes = str(row.get("notes", "") or "").strip()
    if notes.lower() == "nan":
        notes = ""

    seeds: dict[int, float] = {}
    per_path = BASELINE_PER_SEED.get((dataset, model_key))
    if per_path and (_ROOT / per_path).exists():
        ps = pd.read_csv(_ROOT / per_path)
        col = "top1_acc" if "top1_acc" in ps.columns else "top1_accuracy"
        for _, r in ps.iterrows():
            seeds[int(r["seed"])] = float(r[col])
    elif model_key == "logreg" and dataset == "HAM10000":
        seeds = {42: mean}

    n_seeds = len(seeds) if seeds else (1 if "single seed" in notes.lower() else 5)

    return {
        "model_type": "ML baseline",
        "dataset": dataset,
        "condition": BASELINE_DISPLAY[model_key],
        "condition_key": model_key,
        "source": "run_ml_baselines.py (test set)",
        "top1_mean": mean,
        "top1_std": std,
        "top1_fmt": f"{mean:.3f} ± {std:.3f}",
        "seeds": seeds,
        "notes": notes,
        "n_seeds": n_seeds,
    }


def _seed_cell(seeds: dict[int, float], seed: int) -> str:
    if seed in seeds:
        return f"{seeds[seed]:.4f}"
    return "—"


def _write_markdown(nam_rows: list, baseline_rows: list) -> None:
    lines = [
        "# Appendix — Top-1 accuracy (test set)",
        "",
        "Mean ± sample standard deviation over seeds 42–46 (where applicable). "
        "Fixed train/validation/test partition.",
        "",
        "## NAM models",
        "",
        "| Dataset | Condition | Top-1 accuracy | Source |",
        "|---------|-----------|----------------|--------|",
    ]
    for r in nam_rows:
        lines.append(
            f"| {r['dataset']} | {r['condition']} | {r['top1_fmt']} | {r['source']} |"
        )

    lines += [
        "",
        "## ML baselines",
        "",
        "| Dataset | Model | Top-1 accuracy | Source | Notes |",
        "|---------|-------|----------------|--------|-------|",
    ]
    for r in baseline_rows:
        note = r["notes"] or ("5 seeds" if r.get("n_seeds", 5) == 5 else "")
        lines.append(
            f"| {r['dataset']} | {r['condition']} | {r['top1_fmt']} | "
            f"{r['source']} | {note} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- **NAM — Plain NAM** and **Concurvity only**: from `train_final` "
        "(`aggregated_metrics.csv`, column `top1_accuracy`).",
        "- **NAM — Sparsity only** and **Sparsity + concurvity**: warm-start "
        "sparsity sweep at **K=10** (`anec_evaluation/by_seed.csv`, `test_top1_acc`).",
        "- **ML baselines**: `results/baselines_ml/` (HAM10000) and "
        "`results/chestxray/baselines_ml/` (`aggregate_summary.csv`).",
        "- HAM10000 logistic regression: single seed (42). HAM10000 XGBoost: "
        "identical predictions across seeds (std = 0). Chest X-ray logistic "
        "regression: five seeds, deterministic `lbfgs` solver (std = 0).",
        "- Per-seed values: `appendix_top1_accuracy_by_seed.csv`.",
        "",
        "## Per-seed detail — NAM",
        "",
        "### HAM10000",
        "",
        "| Condition | 42 | 43 | 44 | 45 | 46 |",
        "|-----------|-----|-----|-----|-----|-----|",
    ]
    ham_nam = [r for r in nam_rows if r["dataset"] == "HAM10000"]
    for r in ham_nam:
        s = r["seeds"]
        lines.append(
            f"| {r['condition']} | {_seed_cell(s, 42)} | {_seed_cell(s, 43)} | "
            f"{_seed_cell(s, 44)} | {_seed_cell(s, 45)} | {_seed_cell(s, 46)} |"
        )

    lines += [
        "",
        "### Chest X-ray",
        "",
        "| Condition | 42 | 43 | 44 | 45 | 46 |",
        "|-----------|-----|-----|-----|-----|-----|",
    ]
    cxr_nam = [r for r in nam_rows if r["dataset"] == "Chest X-ray"]
    for r in cxr_nam:
        s = r["seeds"]
        lines.append(
            f"| {r['condition']} | {_seed_cell(s, 42)} | {_seed_cell(s, 43)} | "
            f"{_seed_cell(s, 44)} | {_seed_cell(s, 45)} | {_seed_cell(s, 46)} |"
        )

    lines += [
        "",
        "## Per-seed detail — ML baselines",
        "",
        "### HAM10000",
        "",
        "| Model | 42 | 43 | 44 | 45 | 46 |",
        "|-------|-----|-----|-----|-----|-----|",
    ]
    ham_bl = [r for r in baseline_rows if r["dataset"] == "HAM10000"]
    for r in ham_bl:
        s = r["seeds"]
        lines.append(
            f"| {r['condition']} | {_seed_cell(s, 42)} | {_seed_cell(s, 43)} | "
            f"{_seed_cell(s, 44)} | {_seed_cell(s, 45)} | {_seed_cell(s, 46)} |"
        )

    lines += [
        "",
        "### Chest X-ray",
        "",
        "| Model | 42 | 43 | 44 | 45 | 46 |",
        "|-------|-----|-----|-----|-----|-----|",
    ]
    cxr_bl = [r for r in baseline_rows if r["dataset"] == "Chest X-ray"]
    for r in cxr_bl:
        s = r["seeds"]
        lines.append(
            f"| {r['condition']} | {_seed_cell(s, 42)} | {_seed_cell(s, 43)} | "
            f"{_seed_cell(s, 44)} | {_seed_cell(s, 45)} | {_seed_cell(s, 46)} |"
        )

    (OUT / "appendix_top1_accuracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    nam_rows = [
        load_train_final(
            "results/HAM10000/plain_nam/aggregated_metrics.csv", "HAM10000", "plain_nam"
        ),
        load_train_final(
            "results/HAM10000/concurvity_only/aggregated_metrics.csv",
            "HAM10000",
            "concurvity_only",
        ),
        load_anec_k10(
            "results/HAM10000/anec_evaluation/by_seed.csv",
            "results/HAM10000/anec_evaluation/aggregated.csv",
            "HAM10000",
            "sparsity_only",
        ),
        load_anec_k10(
            "results/HAM10000/anec_evaluation/by_seed.csv",
            "results/HAM10000/anec_evaluation/aggregated.csv",
            "HAM10000",
            "sparsity_conc",
            anec_name="sparsity_concurvity",
        ),
        load_train_final(
            "results/chestxray/plain_nam/aggregated_metrics.csv",
            "Chest X-ray",
            "plain_nam",
        ),
        load_train_final(
            "results/chestxray/concurvity_only/aggregated_metrics.csv",
            "Chest X-ray",
            "concurvity_only",
        ),
        load_anec_k10(
            "results/chestxray/anec_evaluation/by_seed.csv",
            "results/chestxray/anec_evaluation/aggregated.csv",
            "Chest X-ray",
            "sparsity_only",
        ),
        load_anec_k10(
            "results/chestxray/anec_evaluation/by_seed.csv",
            "results/chestxray/anec_evaluation/aggregated.csv",
            "Chest X-ray",
            "sparsity_conc",
            anec_name="sparsity_conc",
        ),
    ]

    baseline_rows = [
        load_baseline("HAM10000", "logreg", "results/baselines_ml/aggregate_summary.csv"),
        load_baseline("HAM10000", "xgboost", "results/baselines_ml/aggregate_summary.csv"),
        load_baseline(
            "HAM10000", "random_forest", "results/baselines_ml/aggregate_summary.csv"
        ),
        load_baseline(
            "Chest X-ray", "logreg", "results/chestxray/baselines_ml/aggregate_summary.csv"
        ),
        load_baseline(
            "Chest X-ray", "xgboost", "results/chestxray/baselines_ml/aggregate_summary.csv"
        ),
        load_baseline(
            "Chest X-ray",
            "random_forest",
            "results/chestxray/baselines_ml/aggregate_summary.csv",
        ),
    ]

    all_rows = nam_rows + baseline_rows

    summary = pd.DataFrame(
        [
            {
                "model_type": r["model_type"],
                "dataset": r["dataset"],
                "condition": r["condition"],
                "top1_accuracy_mean": round(r["top1_mean"], 6),
                "top1_accuracy_std": round(r["top1_std"], 6),
                "top1_accuracy_fmt": r["top1_fmt"],
                "source": r["source"],
                "n_seeds": r.get("n_seeds", 5),
                "notes": (
                    r.get("notes", "")
                    or ("5 seeds" if r["model_type"] == "ML baseline" and r.get("n_seeds", 5) == 5 else "")
                ),
            }
            for r in all_rows
        ]
    )
    summary.to_csv(OUT / "appendix_top1_accuracy.csv", index=False)

    per_seed = []
    for r in all_rows:
        for seed, val in sorted(r["seeds"].items()):
            per_seed.append(
                {
                    "model_type": r["model_type"],
                    "dataset": r["dataset"],
                    "condition": r["condition"],
                    "seed": seed,
                    "top1_accuracy": val,
                    "source": r["source"],
                }
            )
    pd.DataFrame(per_seed).to_csv(OUT / "appendix_top1_accuracy_by_seed.csv", index=False)

    _write_markdown(nam_rows, baseline_rows)

    print(summary.to_string(index=False))
    print(f"\nWrote {OUT / 'appendix_top1_accuracy.csv'}")
    print(f"Wrote {OUT / 'appendix_top1_accuracy.md'}")


if __name__ == "__main__":
    main()
