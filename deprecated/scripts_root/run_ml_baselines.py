"""
XGBoost and Random Forest baselines on 24-dim BiomedCLIP v6 concept scores.

Same train/val/test protocol and metrics as the NAM pipeline.

Run from project root:
    python scripts/run_ml_baselines.py
    python scripts/run_ml_baselines.py --model xgboost --seeds 42 43
    python scripts/run_ml_baselines.py --skip-tuning   # quick sanity check

Outputs → results/baselines_ml/
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import textwrap
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, PredefinedSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# ── Project root & paths ───────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FEATURES_PATH  = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH    = _ROOT / "data/splits/train_test_lesion_split.npz"
OUT_DIR        = _ROOT / "results/baselines_ml"
LR_METRICS_CSV = _ROOT / "reports/baselines/logreg_biomedclip_v6/metrics_summary.csv"
LR_PERCLASS_CSV = _ROOT / "reports/baselines/logreg_biomedclip_v6/classification_report.csv"
NAM_AGG_CSV    = _ROOT / "results/thesis_tables/aggregate_metrics.csv"
NAM_SEED_CSV   = _ROOT / "results/thesis_tables/per_seed_full.csv"

# Class order matching the thesis tables
CLASS_ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

# ── Hyperparameter grids ───────────────────────────────────────────────────────
XGBOOST_GRID = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.05, 0.1, 0.2],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

RF_GRID = {
    "n_estimators":      [200, 500, 1000],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
}

# Defaults used when --skip-tuning is set
XGBOOST_DEFAULTS = dict(n_estimators=300, max_depth=5, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8)
RF_DEFAULTS = dict(n_estimators=500, max_depth=None, min_samples_split=2,
                   min_samples_leaf=1, max_features="sqrt")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    print("Loading features ...")
    feat = np.load(FEATURES_PATH, allow_pickle=True)
    X          = feat["scores"].astype(np.float32)   # (10015, 24)
    y_str      = feat["labels"]                       # string labels
    lesion_ids = feat["lesion_ids"]
    concept_ids = feat["concept_ids"]

    assert X.shape == (10015, 24), f"Unexpected feature shape: {X.shape}"
    assert len(y_str) == len(lesion_ids) == X.shape[0]

    print("Loading splits ...")
    split     = np.load(SPLITS_PATH)
    train_idx = split["train_idx"]   # 8020
    test_idx  = split["test_idx"]    # 1995

    assert len(np.intersect1d(train_idx, test_idx)) == 0, "Train/test index overlap!"
    assert len(np.union1d(train_idx, test_idx)) == X.shape[0], "Missing indices in split!"

    # Encode string labels → integers
    le = LabelEncoder()
    le.fit(sorted(np.unique(y_str)))          # alphabetical: akiec bcc bkl df mel nv vasc
    y_int = le.transform(y_str)

    print(f"Feature shape : {X.shape}")
    print(f"Classes       : {list(le.classes_)}")
    print(f"Train pool    : {len(train_idx)}, Test: {len(test_idx)}")
    return X, y_int, y_str, lesion_ids, concept_ids, train_idx, test_idx, le


def carve_val_split(X_pool, y_pool, lesion_pool):
    """Fixed GroupShuffleSplit val carve — random_state=42 always."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_rel, val_rel = next(gss.split(X_pool, y_pool, groups=lesion_pool))
    return train_rel, val_rel


def verify_no_lesion_overlap(lid_a, lid_b, label_a="A", label_b="B"):
    overlap = set(lid_a) & set(lid_b)
    assert len(overlap) == 0, (
        f"Lesion overlap between {label_a} and {label_b}: {len(overlap)} shared lesions"
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true_int, y_pred_int, y_proba, le: LabelEncoder) -> dict:
    """Compute all required aggregate and per-class metrics."""
    classes  = le.classes_
    y_true_str = le.inverse_transform(y_true_int)

    bal_acc      = balanced_accuracy_score(y_true_int, y_pred_int)
    macro_f1     = f1_score(y_true_int, y_pred_int, average="macro", zero_division=0)
    top1_acc     = accuracy_score(y_true_int, y_pred_int)
    auc_weighted = roc_auc_score(
        y_true_str, y_proba, multi_class="ovr", average="weighted", labels=classes
    )

    per_class: Dict[str, dict] = {}
    for i, cls in enumerate(classes):
        y_bin  = (y_true_int == i).astype(int)
        p_bin  = (y_pred_int == i).astype(int)
        per_class[cls] = dict(
            auc  = roc_auc_score(y_bin, y_proba[:, i]),
            f1   = f1_score(y_bin, p_bin, zero_division=0),
            prec = precision_score(y_bin, p_bin, zero_division=0),
            rec  = recall_score(y_bin, p_bin, zero_division=0),
        )

    row = dict(balanced_acc=bal_acc, macro_f1=macro_f1,
               auc_weighted=auc_weighted, top1_acc=top1_acc)
    for cls in CLASS_ORDER:
        m = per_class.get(cls, {})
        row[f"auc_{cls}"]  = m.get("auc",  np.nan)
        row[f"f1_{cls}"]   = m.get("f1",   np.nan)
        row[f"prec_{cls}"] = m.get("prec", np.nan)
        row[f"rec_{cls}"]  = m.get("rec",  np.nan)
    return row


# ── XGBoost ───────────────────────────────────────────────────────────────────

def run_xgboost_seed(X_pool, y_pool, lid_pool, X_test, y_test, le, seed, skip_tuning):
    from xgboost import XGBClassifier

    n_classes = len(le.classes_)
    train_rel, val_rel = carve_val_split(X_pool, y_pool, lid_pool)

    X_tf, y_tf   = X_pool[train_rel], y_pool[train_rel]
    X_val, y_val = X_pool[val_rel],   y_pool[val_rel]
    lid_tf, lid_val = lid_pool[train_rel], lid_pool[val_rel]

    verify_no_lesion_overlap(lid_tf, lid_val, "train_final", "val")
    print(f"  Split sizes -> train_final: {len(y_tf)}, val: {len(y_val)}, test: {len(y_test)}")

    xgb_common = dict(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    if skip_tuning:
        best_params = XGBOOST_DEFAULTS.copy()
        print(f"  [skip-tuning] {best_params}")
    else:
        # Grid search — no early stopping; PredefinedSplit on [train_final | val]
        fold_labels = np.concatenate([
            -np.ones(len(train_rel), dtype=int),
             np.zeros(len(val_rel),   dtype=int),
        ])
        ps   = PredefinedSplit(fold_labels)
        X_tv = np.vstack([X_tf, X_val])
        y_tv = np.concatenate([y_tf, y_val])
        sw_tv = compute_sample_weight("balanced", y_tv)

        gs = GridSearchCV(
            XGBClassifier(**xgb_common), XGBOOST_GRID,
            scoring="balanced_accuracy", cv=ps,
            refit=False, verbose=2, n_jobs=1,
        )
        t0 = time.time()
        gs.fit(X_tv, y_tv, sample_weight=sw_tv)
        print(f"  Grid search done in {time.time()-t0:.0f}s")
        print(f"  Best val balanced_accuracy : {gs.best_score_:.4f}")
        print(f"  Best params : {gs.best_params_}")
        best_params = gs.best_params_

    # Final refit on the full train pool (train_final + val) with the winning config
    # n_estimators is taken from the grid winner; no early stopping for reproducibility
    sw_full = compute_sample_weight("balanced", y_pool)
    model = XGBClassifier(**xgb_common, **best_params)
    model.fit(X_pool, y_pool, sample_weight=sw_full)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba, le)
    metrics["seed"] = seed
    return metrics, {"seed": seed, **best_params}


# ── Random Forest ─────────────────────────────────────────────────────────────

def run_rf_seed(X_pool, y_pool, lid_pool, X_test, y_test, le, seed, skip_tuning):
    train_rel, val_rel = carve_val_split(X_pool, y_pool, lid_pool)

    X_tf, y_tf   = X_pool[train_rel], y_pool[train_rel]
    X_val, y_val = X_pool[val_rel],   y_pool[val_rel]
    lid_tf, lid_val = lid_pool[train_rel], lid_pool[val_rel]

    verify_no_lesion_overlap(lid_tf, lid_val, "train_final", "val")
    print(f"  Split sizes -> train_final: {len(y_tf)}, val: {len(y_val)}, test: {len(y_test)}")

    rf_common = dict(class_weight="balanced", random_state=seed, n_jobs=-1)

    if skip_tuning:
        best_params = RF_DEFAULTS.copy()
        print(f"  [skip-tuning] {best_params}")
    else:
        fold_labels = np.concatenate([
            -np.ones(len(train_rel), dtype=int),
             np.zeros(len(val_rel),   dtype=int),
        ])
        ps   = PredefinedSplit(fold_labels)
        X_tv = np.vstack([X_tf, X_val])
        y_tv = np.concatenate([y_tf, y_val])

        gs = GridSearchCV(
            RandomForestClassifier(**rf_common), RF_GRID,
            scoring="balanced_accuracy", cv=ps,
            refit=False, verbose=2, n_jobs=1,
        )
        t0 = time.time()
        gs.fit(X_tv, y_tv)
        print(f"  Grid search done in {time.time()-t0:.0f}s")
        print(f"  Best val balanced_accuracy : {gs.best_score_:.4f}")
        print(f"  Best params : {gs.best_params_}")
        best_params = gs.best_params_

    model = RandomForestClassifier(**rf_common, **best_params)
    model.fit(X_pool, y_pool)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba, le)
    metrics["seed"] = seed
    return metrics, {"seed": seed, **best_params}


# ── Aggregation helpers ───────────────────────────────────────────────────────

TOP_METRICS = ["balanced_acc", "macro_f1", "auc_weighted", "top1_acc"]


def aggregate(rows: List[dict]) -> dict:
    df = pd.DataFrame(rows)
    agg = {}
    for col in df.columns:
        if col == "seed":
            continue
        agg[f"{col}_mean"] = df[col].mean()
        agg[f"{col}_std"]  = df[col].std(ddof=1)
    return agg


def per_class_rows(seed_rows: List[dict], model_name: str) -> pd.DataFrame:
    records = []
    for row in seed_rows:
        seed = row["seed"]
        for cls in CLASS_ORDER:
            records.append(dict(
                model=model_name, seed=seed, cls=cls,
                auc  = row.get(f"auc_{cls}",  np.nan),
                f1   = row.get(f"f1_{cls}",   np.nan),
                prec = row.get(f"prec_{cls}", np.nan),
                rec  = row.get(f"rec_{cls}",  np.nan),
            ))
    return pd.DataFrame(records)


# ── NAM reference numbers ─────────────────────────────────────────────────────

def load_nam_numbers() -> dict:
    """Pull NAM metrics from thesis_tables if available."""
    result = {}
    if NAM_AGG_CSV.exists():
        df = pd.read_csv(NAM_AGG_CSV)
        for _, r in df.iterrows():
            cond = r["condition"]
            result[cond] = {
                "auc_weighted_mean": r.get("auc_weighted_mean"),
                "auc_weighted_std":  r.get("auc_weighted_std"),
                "balanced_acc_mean": r.get("balanced_acc_mean"),
                "balanced_acc_std":  r.get("balanced_acc_std"),
            }

    if NAM_SEED_CSV.exists():
        df = pd.read_csv(NAM_SEED_CSV)
        f1_cols = [c for c in df.columns if c.startswith("test_f1_")]
        for cond, grp in df.groupby("condition"):
            macro_f1_per_seed = grp[f1_cols].mean(axis=1)
            if cond not in result:
                result[cond] = {}
            result[cond]["macro_f1_mean"] = macro_f1_per_seed.mean()
            result[cond]["macro_f1_std"]  = macro_f1_per_seed.std(ddof=1)
            result[cond]["top1_acc_mean"] = None   # not stored in NAM results
            result[cond]["top1_acc_std"]  = None

    return result


def load_lr_numbers() -> dict:
    if not LR_METRICS_CSV.exists():
        return {}
    df = pd.read_csv(LR_METRICS_CSV)
    r  = df.iloc[0]
    return dict(
        balanced_acc_mean = r.get("balanced_accuracy"),
        balanced_acc_std  = 0.0,
        macro_f1_mean     = r.get("macro_f1"),
        macro_f1_std      = 0.0,
        auc_weighted_mean = r.get("auc_ovr_weighted"),
        auc_weighted_std  = 0.0,
        top1_acc_mean     = r.get("top1_accuracy"),
        top1_acc_std      = 0.0,
    )


# ── Summary writers ───────────────────────────────────────────────────────────

def fmt(mean, std, decimals=4):
    if mean is None:
        return "N/A"
    if std is None or std == 0.0 or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def write_aggregate_summary(xgb_agg, rf_agg, out_dir):
    lr = load_lr_numbers()
    rows = []

    def add_row(name, d):
        rows.append(dict(
            model              = name,
            balanced_acc_mean  = d.get("balanced_acc_mean"),
            balanced_acc_std   = d.get("balanced_acc_std"),
            macro_f1_mean      = d.get("macro_f1_mean"),
            macro_f1_std       = d.get("macro_f1_std"),
            auc_weighted_mean  = d.get("auc_weighted_mean"),
            auc_weighted_std   = d.get("auc_weighted_std"),
            top1_acc_mean      = d.get("top1_acc_mean"),
            top1_acc_std       = d.get("top1_acc_std"),
            notes              = d.get("notes", ""),
        ))

    if lr:
        add_row("logreg", {**lr, "notes": "single seed (seed=42)"})
    else:
        add_row("logreg", {"notes": "TODO — LR metrics not found"})

    add_row("xgboost", xgb_agg)
    add_row("random_forest", rf_agg)

    pd.DataFrame(rows).to_csv(out_dir / "aggregate_summary.csv", index=False)


def write_summary_txt(xgb_rows, rf_rows, xgb_agg, rf_agg, out_dir):
    lr = load_lr_numbers()
    nam = load_nam_numbers()

    lines = [
        "=" * 72,
        "ML Baselines on 24-dim BiomedCLIP v6 features - HAM10000",
        "=" * 72,
        "",
        "Protocol: 5 seeds (42-46), fixed val split (GroupShuffleSplit random_state=42)",
        "Val used for hparam tuning; test touched once per (model, seed).",
        "",
    ]

    for model_name, rows, agg in [("XGBoost", xgb_rows, xgb_agg),
                                   ("Random Forest", rf_rows, rf_agg)]:
        lines += [f"-- {model_name} " + "-"*50]
        lines += [f"  {'Metric':<30} {'Mean':>10} {'Std':>10}"]
        lines += [f"  {'-'*50}"]
        for m in TOP_METRICS:
            lines.append(f"  {m:<30} {agg.get(m+'_mean', np.nan):>10.4f}"
                         f" {agg.get(m+'_std',  np.nan):>10.4f}")
        lines.append("")
        lines.append(f"  Per-seed top-1 balanced accuracy:")
        for r in rows:
            lines.append(f"    seed={r['seed']}  bal_acc={r['balanced_acc']:.4f}"
                         f"  macro_f1={r['macro_f1']:.4f}"
                         f"  auc={r['auc_weighted']:.4f}")
        lines.append("")

    lines += [
        "-- Comparison Table " + "-"*50,
        f"  {'Model':<30} {'Bal.Acc':>10} {'Macro F1':>10} {'AUC (wt)':>10} {'Top-1':>10}",
        f"  {'-'*62}",
    ]

    def cmp_row(name, d):
        return (f"  {name:<30}"
                f" {fmt(d.get('balanced_acc_mean'), d.get('balanced_acc_std'), 4):>10}"
                f" {fmt(d.get('macro_f1_mean'),    d.get('macro_f1_std'),    4):>10}"
                f" {fmt(d.get('auc_weighted_mean'), d.get('auc_weighted_std'), 4):>10}"
                f" {fmt(d.get('top1_acc_mean'),    d.get('top1_acc_std'),    4):>10}")

    if lr:
        lines.append(cmp_row("LR (single seed)", lr))
    else:
        lines.append(f"  {'LR':<30} {'TODO':>10}")

    lines.append(cmp_row("XGBoost (5 seeds)", xgb_agg))
    lines.append(cmp_row("Random Forest (5 seeds)", rf_agg))

    for cond_key, label in [
        ("plain_nam",              "NAM plain (5 seeds)"),
        ("sparsity_conc_lc1_warmft", "NAM sparse+conc (5 seeds)"),
    ]:
        nd = nam.get(cond_key, {})
        if nd:
            lines.append(cmp_row(label, nd))
        else:
            lines.append(f"  {label:<30} {'TODO':>10}")

    lines.append("")
    txt = "\n".join(lines)
    print("\n" + txt)
    (out_dir / "summary.txt").write_text(txt, encoding="utf-8")


def write_comparison_md(xgb_agg, rf_agg, out_dir):
    lr  = load_lr_numbers()
    nam = load_nam_numbers()

    def mfmt(d, key_mean, key_std, dec=4):
        m = d.get(key_mean)
        s = d.get(key_std)
        if m is None:
            return "—"
        if s is None or s == 0.0:
            return f"{m:.{dec}f}"
        return f"{m:.{dec}f} ± {s:.{dec}f}"

    def make_row(label, d, note=""):
        return (f"| {label:<38} "
                f"| {mfmt(d,'balanced_acc_mean','balanced_acc_std'):>16} "
                f"| {mfmt(d,'macro_f1_mean','macro_f1_std'):>10} "
                f"| {mfmt(d,'auc_weighted_mean','auc_weighted_std'):>10} "
                f"| {mfmt(d,'top1_acc_mean','top1_acc_std'):>8} "
                f"| {note} |")

    header = (
        "| Model                                  "
        "| Balanced Accuracy "
        "| Macro F1   "
        "| AUC (wt)   "
        "| Top-1    "
        "| Notes |"
    )
    sep = ("|" + "-"*40 + "|" + "-"*18 + "|" + "-"*12 + "|" + "-"*12 + "|"
           + "-"*10 + "|" + "-"*20 + "|")

    rows = [header, sep]

    if lr:
        rows.append(make_row("Logistic Regression", lr, "single seed"))
    else:
        rows.append(f"| Logistic Regression                    | TODO | | | | |")

    rows.append(make_row("XGBoost", xgb_agg, "5 seeds, grid search"))
    rows.append(make_row("Random Forest", rf_agg, "5 seeds, grid search"))

    for cond_key, label, note in [
        ("plain_nam",              "NAM (plain)",                    "λs=0, λc=0"),
        ("concurvity_only_lc1",    "NAM (concurvity only)",          "λs=0, λc=1"),
        ("sparsity_only_lc0_warmft","NAM (sparsity only)",           "λs=23.7, λc=0"),
        ("sparsity_conc_lc1_warmft","NAM (sparsity + concurvity)",   "λs=12, λc=1 ★"),
    ]:
        nd = nam.get(cond_key, {})
        rows.append(make_row(label, nd, note) if nd else
                    f"| {label:<38} | TODO | | | | {note} |")

    md = textwrap.dedent(f"""\
        # Baseline Comparison — HAM10000 (24-dim BiomedCLIP v6)

        All models trained on the same 24 BiomedCLIP concept-score features.
        Train/val/test split: lesion-disjoint 80/16/20 (GroupShuffleSplit, seed=42).
        XGBoost and Random Forest: 5 seeds (42–46), hyperparameters tuned on the val set
        using PredefinedSplit + GridSearchCV (balanced accuracy criterion).
        NAM results from thesis sweeps (5 seeds each, same protocol).

        ★ = primary NAM condition reported in thesis.

        | Metric definition: AUC = OvR weighted, Macro F1 = unweighted class average.

        """) + "\n".join(rows) + textwrap.dedent("""

        ## Notes

        - LR run with a single seed (random_state=42); no seed variance reported.
        - NAM top-1 accuracy not stored in thesis result files — marked as `—`.
        - XGBoost final refit uses the winning `n_estimators` from grid search
          (no early stopping in final refit, for reproducibility).
        - Random Forest final refit on the full train pool (train_final + val combined).
        """)

    (out_dir / "baseline_comparison.md").write_text(md, encoding="utf-8")
    print(f"\nComparison saved -> {out_dir / 'baseline_comparison.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _append_to_csv(path: pathlib.Path, new_df: pd.DataFrame) -> None:
    """Append rows to a CSV, creating it if it doesn't exist yet."""
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def _done_seeds(csv_path: pathlib.Path) -> set:
    """Return the set of seed values already written to a per-seed CSV."""
    if not csv_path.exists():
        return set()
    try:
        return set(pd.read_csv(csv_path)["seed"].tolist())
    except Exception:
        return set()


def _flush_seed(model_tag: str, metrics: dict, hparams: dict, out_dir: pathlib.Path) -> None:
    """Write one seed's results immediately to disk (append mode)."""
    top_cols = ["seed"] + TOP_METRICS
    seed_row  = {k: metrics[k] for k in top_cols}

    _append_to_csv(out_dir / f"{model_tag}_per_seed.csv",
                   pd.DataFrame([seed_row]))
    _append_to_csv(out_dir / f"{model_tag}_per_class.csv",
                   per_class_rows([metrics], model_tag))
    _append_to_csv(out_dir / f"{model_tag}_best_hparams.csv",
                   pd.DataFrame([hparams]))
    print(f"  [checkpointed seed={metrics['seed']} -> {model_tag}_per_seed.csv]",
          flush=True)


def _rebuild_summaries(out_dir: pathlib.Path) -> None:
    """Regenerate aggregate CSVs and text reports by reading completed per-seed files."""
    xgb_path = out_dir / "xgboost_per_seed.csv"
    rf_path  = out_dir / "rf_per_seed.csv"

    xgb_seed_rows: List[dict] = []
    rf_seed_rows:  List[dict] = []

    if xgb_path.exists():
        xgb_seed_rows = pd.read_csv(xgb_path).to_dict("records")
    if rf_path.exists():
        rf_seed_rows = pd.read_csv(rf_path).to_dict("records")

    xgb_agg = aggregate(xgb_seed_rows) if xgb_seed_rows else {}
    rf_agg  = aggregate(rf_seed_rows)  if rf_seed_rows  else {}

    if xgb_agg or rf_agg:
        write_aggregate_summary(xgb_agg, rf_agg, out_dir)
        write_summary_txt(xgb_seed_rows, rf_seed_rows, xgb_agg, rf_agg, out_dir)
        write_comparison_md(xgb_agg, rf_agg, out_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",  choices=["xgboost", "rf", "both"], default="both")
    p.add_argument("--seeds",  type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--skip-tuning", action="store_true",
                   help="Skip grid search and use default hyperparameters")
    p.add_argument("--overwrite", action="store_true",
                   help="Delete existing per-seed CSVs and start fresh (no resume)")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Optional clean slate ───────────────────────────────────────────────────
    if args.overwrite:
        for stem in ["xgboost_per_seed", "xgboost_per_class", "xgboost_best_hparams",
                     "rf_per_seed", "rf_per_class", "rf_best_hparams"]:
            p = OUT_DIR / f"{stem}.csv"
            if p.exists():
                p.unlink()
                print(f"  Deleted {p.name}")

    X, y_int, y_str, lesion_ids, concept_ids, train_idx, test_idx, le = load_data()

    X_pool   = X[train_idx]
    y_pool   = y_int[train_idx]
    lid_pool = lesion_ids[train_idx]
    X_test   = X[test_idx]
    y_test   = y_int[test_idx]

    verify_no_lesion_overlap(lid_pool, lesion_ids[test_idx], "train_pool", "test")
    print("No lesion overlap between train pool and test - OK")

    run_xgb = args.model in ("xgboost", "both")
    run_rf  = args.model in ("rf", "both")

    # ── Detect already-completed seeds (resume support) ────────────────────────
    xgb_done = _done_seeds(OUT_DIR / "xgboost_per_seed.csv") if run_xgb else set()
    rf_done  = _done_seeds(OUT_DIR / "rf_per_seed.csv")      if run_rf  else set()

    if xgb_done:
        print(f"Resuming XGBoost — seeds already done: {sorted(xgb_done)}")
    if rf_done:
        print(f"Resuming RF — seeds already done: {sorted(rf_done)}")

    total_t0 = time.time()

    for seed in args.seeds:
        # ── XGBoost ───────────────────────────────────────────────────────────
        if run_xgb:
            if seed in xgb_done:
                print(f"\n[skip] XGBoost seed={seed} already in CSV.")
            else:
                print(f"\n{'='*60}", flush=True)
                print(f"=== Starting XGBoost seed={seed} ===", flush=True)
                print(f"{'='*60}", flush=True)
                t0 = time.time()
                metrics, hparams = run_xgboost_seed(
                    X_pool, y_pool, lid_pool, X_test, y_test, le,
                    seed, args.skip_tuning
                )
                elapsed = time.time() - t0
                print(f"  Test results  bal_acc={metrics['balanced_acc']:.4f}"
                      f"  macro_f1={metrics['macro_f1']:.4f}"
                      f"  auc={metrics['auc_weighted']:.4f}"
                      f"  top1={metrics['top1_acc']:.4f}  [{elapsed:.0f}s]",
                      flush=True)
                # Write this seed immediately — survives a mid-run kill
                _flush_seed("xgboost", metrics, hparams, OUT_DIR)

        # ── Random Forest ─────────────────────────────────────────────────────
        if run_rf:
            if seed in rf_done:
                print(f"\n[skip] Random Forest seed={seed} already in CSV.")
            else:
                print(f"\n{'='*60}", flush=True)
                print(f"=== Starting Random Forest seed={seed} ===", flush=True)
                print(f"{'='*60}", flush=True)
                t0 = time.time()
                metrics, hparams = run_rf_seed(
                    X_pool, y_pool, lid_pool, X_test, y_test, le,
                    seed, args.skip_tuning
                )
                elapsed = time.time() - t0
                print(f"  Test results  bal_acc={metrics['balanced_acc']:.4f}"
                      f"  macro_f1={metrics['macro_f1']:.4f}"
                      f"  auc={metrics['auc_weighted']:.4f}"
                      f"  top1={metrics['top1_acc']:.4f}  [{elapsed:.0f}s]",
                      flush=True)
                _flush_seed("rf", metrics, hparams, OUT_DIR)

    print(f"\nTotal wall time: {(time.time()-total_t0)/60:.1f} min")

    # ── Regenerate aggregate summaries from the (now complete) per-seed CSVs ──
    _rebuild_summaries(OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
