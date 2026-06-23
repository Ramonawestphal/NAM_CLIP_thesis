"""
XGBoost and Random Forest baselines on 17-dim BiomedCLIP v4 concept scores.
Chest X-ray three-way classification (Normal / Bacteria / Virus).

Same train/val/test protocol and metrics as the chest X-ray NAM pipeline:
  - Patient-disjoint outer split  (chestxray_outer_split.npz)
  - Fixed val split carved from train_pool via GroupShuffleSplit(random_state=42)
    where groups = patient_ids (patient-level disjointness)
  - Test set loaded once, touched only in per-seed evaluation
  - 5 seeds (42–46), GridSearchCV(refit=False) + manual refit on full train pool

Hard isolation rules
────────────────────
- Do NOT modify scripts/HAM10000/, scripts/run_ml_baselines.py, src/, or any
  HAM10000 artefact.
- Do NOT modify any existing chest X-ray artefact (splits, features, prior
  NAM results).
- Do NOT write to results/baselines_ml/ (HAM10000's directory).
- Test set is loaded only in the per-seed evaluation block — never used for
  hyperparameter tuning or fitting.

Outputs → results/chestxray/baselines_ml/
    xgboost_per_seed.csv        one row per seed (top metrics)
    xgboost_per_class.csv       per-class AUC/F1/prec/rec for each seed
    xgboost_best_hparams.csv    winning hyperparameters per seed
    rf_per_seed.csv
    rf_per_class.csv
    rf_best_hparams.csv
    aggregate_summary.csv       mean ± std across seeds for each model
    summary.txt                 human-readable comparison table
    baseline_comparison.md      Markdown table including NAM reference numbers

Run from project root:
    python scripts/chestxray/run_ml_baselines.py
    python scripts/chestxray/run_ml_baselines.py --model xgboost --seeds 42 43
    python scripts/chestxray/run_ml_baselines.py --skip-tuning
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
import time
from typing import Dict, List

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

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError as _xgb_err:
    raise ModuleNotFoundError(
        "xgboost is required.  Install with:  pip install xgboost"
    ) from _xgb_err

# ── Project root & paths ───────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FEATURES_PATH = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v4.npz"
SPLIT_PATH    = _ROOT / "data/splits/chestxray_outer_split.npz"
OUT_DIR       = _ROOT / "results/chestxray/baselines_ml"

# LR reference — not yet run; gracefully absent
LR_METRICS_CSV = _ROOT / "results/chestxray/baselines_ml/logreg_metrics_summary.csv"

# NAM reference CSVs (existing chest X-ray train_final results)
NAM_PLAIN_CSV = _ROOT / "results/chestxray/plain_nam/aggregated_metrics.csv"
NAM_CONC_CSV  = _ROOT / "results/chestxray/concurvity_only/aggregated_metrics.csv"
# sparsity_conc and sparsity_only: not yet implemented in train_final.py
NAM_SPARSITY_CONC_CSV  = _ROOT / "results/chestxray/sparsity_conc/aggregated_metrics.csv"
NAM_SPARSITY_ONLY_CSV  = _ROOT / "results/chestxray/sparsity_only/aggregated_metrics.csv"

# ── Constants ──────────────────────────────────────────────────────────────────
N_FEATURES  = 17
N_CLASSES   = 3
# Matches SUBTYPE_TO_INT in train_final.py — used for display ordering only
CLASS_ORDER = ["normal", "bacteria", "virus"]

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
    """Load chest X-ray features, splits, and labels.

    Test indices are returned but must NOT be used before the per-seed
    evaluation block — only X_test / y_test are constructed in main().
    """
    print("Loading features ...")
    feat          = np.load(FEATURES_PATH, allow_pickle=True)
    X             = feat["scores"].astype(np.float32)       # (5856, 17)
    concept_names = feat["concept_names"].tolist()

    assert X.shape == (5856, N_FEATURES), \
        f"Unexpected feature shape: {X.shape}, expected (5856, {N_FEATURES})"
    assert X.dtype == np.float32, f"Unexpected dtype: {X.dtype}"

    print("Loading splits ...")
    split          = np.load(SPLIT_PATH, allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]   # 4658
    test_idx       = split["test_idx"]         # 1198
    labels_subtype = split["labels_subtype"]   # string: "normal"/"bacteria"/"virus"
    patient_ids    = split["patient_ids"]      # patient-level grouping variable

    assert len(np.intersect1d(train_pool_idx, test_idx)) == 0, \
        "Train/test index overlap!"
    assert len(train_pool_idx) == 4658, \
        f"train_pool len={len(train_pool_idx)}, expected 4658"
    assert len(test_idx) == 1198, \
        f"test len={len(test_idx)}, expected 1198"

    # Patient-level non-overlap
    train_patients = set(patient_ids[train_pool_idx].tolist())
    test_patients  = set(patient_ids[test_idx].tolist())
    pat_overlap    = train_patients & test_patients
    assert len(pat_overlap) == 0, \
        f"{len(pat_overlap)} patients shared between train pool and test"

    # Encode string labels → integers via LabelEncoder (alphabetical):
    # bacteria=0, normal=1, virus=2
    le = LabelEncoder()
    le.fit(sorted(np.unique(labels_subtype)))
    y_int = le.transform(labels_subtype)

    print(f"Feature shape : {X.shape}")
    print(f"Concept names : {concept_names}")
    print(f"Classes (LE)  : {list(le.classes_)}")
    print(f"Train pool    : {len(train_pool_idx)}, Test: {len(test_idx)}")
    print(f"Patient overlap (train/test): 0  ✓")

    return X, y_int, labels_subtype, patient_ids, train_pool_idx, test_idx, le


def carve_val_split(X_pool, y_pool, pid_pool):
    """Fixed GroupShuffleSplit val carve — random_state=42 always.

    groups = patient_ids, matching the NAM train_final.py convention.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_rel, val_rel = next(gss.split(X_pool, y_pool, groups=pid_pool))
    return train_rel, val_rel


def verify_no_patient_overlap(pid_a, pid_b, label_a="A", label_b="B"):
    overlap = set(pid_a.tolist()) & set(pid_b.tolist())
    assert len(overlap) == 0, (
        f"Patient overlap between {label_a} and {label_b}: "
        f"{len(overlap)} shared patients"
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true_int, y_pred_int, y_proba, le: LabelEncoder) -> dict:
    """Compute aggregate and per-class metrics.

    AUC uses OvR weighted (matches NAM pipeline's weighted_auc_ovr).
    Per-class metrics keyed by le.classes_ string names.
    """
    classes    = le.classes_                        # ["bacteria", "normal", "virus"]
    y_true_str = le.inverse_transform(y_true_int)

    bal_acc      = balanced_accuracy_score(y_true_int, y_pred_int)
    macro_f1     = f1_score(y_true_int, y_pred_int, average="macro",    zero_division=0)
    top1_acc     = accuracy_score(y_true_int, y_pred_int)
    auc_weighted = roc_auc_score(
        y_true_str, y_proba,
        multi_class="ovr", average="weighted", labels=classes,
    )

    per_class: Dict[str, dict] = {}
    for i, cls in enumerate(classes):
        y_bin = (y_true_int == i).astype(int)
        p_bin = (y_pred_int == i).astype(int)
        per_class[cls] = dict(
            auc  = roc_auc_score(y_bin, y_proba[:, i]),
            f1   = f1_score(y_bin, p_bin,   zero_division=0),
            prec = precision_score(y_bin, p_bin, zero_division=0),
            rec  = recall_score(y_bin, p_bin,  zero_division=0),
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

def run_xgboost_seed(X_pool, y_pool, pid_pool, X_test, y_test, le,
                     seed, skip_tuning):
    train_rel, val_rel = carve_val_split(X_pool, y_pool, pid_pool)

    X_tf,  y_tf  = X_pool[train_rel], y_pool[train_rel]
    X_val, y_val = X_pool[val_rel],   y_pool[val_rel]
    pid_tf, pid_val = pid_pool[train_rel], pid_pool[val_rel]

    verify_no_patient_overlap(pid_tf, pid_val, "train_final", "val")
    print(f"  Split sizes -> train_final: {len(y_tf)}, "
          f"val: {len(y_val)}, test: {len(y_test)}")

    xgb_common = dict(
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    if skip_tuning:
        best_params = XGBOOST_DEFAULTS.copy()
        print(f"  [skip-tuning] {best_params}")
    else:
        fold_labels = np.concatenate([
            -np.ones(len(train_rel), dtype=int),
             np.zeros(len(val_rel),  dtype=int),
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

    # Final refit on the full train pool (train_final + val)
    sw_full = compute_sample_weight("balanced", y_pool)
    model = XGBClassifier(**xgb_common, **best_params)
    model.fit(X_pool, y_pool, sample_weight=sw_full)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba, le)
    metrics["seed"] = seed
    return metrics, {"seed": seed, **best_params}


# ── Random Forest ─────────────────────────────────────────────────────────────

def run_rf_seed(X_pool, y_pool, pid_pool, X_test, y_test, le,
                seed, skip_tuning):
    train_rel, val_rel = carve_val_split(X_pool, y_pool, pid_pool)

    X_tf,  y_tf  = X_pool[train_rel], y_pool[train_rel]
    X_val, y_val = X_pool[val_rel],   y_pool[val_rel]
    pid_tf, pid_val = pid_pool[train_rel], pid_pool[val_rel]

    verify_no_patient_overlap(pid_tf, pid_val, "train_final", "val")
    print(f"  Split sizes -> train_final: {len(y_tf)}, "
          f"val: {len(y_val)}, test: {len(y_test)}")

    rf_common = dict(class_weight="balanced", random_state=seed, n_jobs=-1)

    if skip_tuning:
        best_params = RF_DEFAULTS.copy()
        print(f"  [skip-tuning] {best_params}")
    else:
        fold_labels = np.concatenate([
            -np.ones(len(train_rel), dtype=int),
             np.zeros(len(val_rel),  dtype=int),
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

    # Final refit on the full train pool (train_final + val)
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

def _read_nam_agg_csv(csv_path: pathlib.Path) -> dict:
    """Read mean/std from aggregated_metrics.csv produced by train_final.py."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    # File has rows for each seed plus a 'mean' and 'std' row
    mean_row = df[df["seed"] == "mean"]
    std_row  = df[df["seed"] == "std"]
    if mean_row.empty or std_row.empty:
        return {}
    m = mean_row.iloc[0]
    s = std_row.iloc[0]
    return {
        "balanced_acc_mean":  float(m.get("balanced_accuracy", np.nan)),
        "balanced_acc_std":   float(s.get("balanced_accuracy", np.nan)),
        "macro_f1_mean":      float(m.get("macro_f1",          np.nan)),
        "macro_f1_std":       float(s.get("macro_f1",          np.nan)),
        "auc_weighted_mean":  float(m.get("weighted_auc_ovr",  np.nan)),
        "auc_weighted_std":   float(s.get("weighted_auc_ovr",  np.nan)),
        "top1_acc_mean":      float(m.get("top1_accuracy",     np.nan)),
        "top1_acc_std":       float(s.get("top1_accuracy",     np.nan)),
    }


def load_nam_numbers() -> dict:
    """Pull NAM metrics from existing chest X-ray train_final outputs."""
    return {
        "plain_nam":        _read_nam_agg_csv(NAM_PLAIN_CSV),
        "concurvity_only":  _read_nam_agg_csv(NAM_CONC_CSV),
        "sparsity_conc":    _read_nam_agg_csv(NAM_SPARSITY_CONC_CSV),
        "sparsity_only":    _read_nam_agg_csv(NAM_SPARSITY_ONLY_CSV),
    }


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
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
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

    add_row("xgboost",       xgb_agg)
    add_row("random_forest", rf_agg)

    pd.DataFrame(rows).to_csv(out_dir / "aggregate_summary.csv", index=False)


def write_summary_txt(xgb_rows, rf_rows, xgb_agg, rf_agg, out_dir):
    lr  = load_lr_numbers()
    nam = load_nam_numbers()

    lines = [
        "=" * 72,
        "ML Baselines on 17-dim BiomedCLIP v4 features - Chest X-ray",
        "=" * 72,
        "",
        "Protocol: 5 seeds (42-46), fixed val split "
        "(GroupShuffleSplit random_state=42, groups=patient_ids)",
        "Val used for hparam tuning; test touched once per (model, seed).",
        "",
    ]

    for model_name, rows, agg in [("XGBoost",       xgb_rows, xgb_agg),
                                   ("Random Forest", rf_rows,  rf_agg)]:
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
        f"  {'Model':<30} {'Bal.Acc':>10} {'Macro F1':>10} "
        f"{'AUC (wt)':>10} {'Top-1':>10}",
        f"  {'-'*62}",
    ]

    def cmp_row(name, d):
        return (f"  {name:<30}"
                f" {fmt(d.get('balanced_acc_mean'), d.get('balanced_acc_std'), 4):>10}"
                f" {fmt(d.get('macro_f1_mean'),     d.get('macro_f1_std'),    4):>10}"
                f" {fmt(d.get('auc_weighted_mean'), d.get('auc_weighted_std'), 4):>10}"
                f" {fmt(d.get('top1_acc_mean'),     d.get('top1_acc_std'),    4):>10}")

    if lr:
        lines.append(cmp_row("LR (single seed)", lr))
    else:
        lines.append(f"  {'LR':<30} {'TODO':>10}")

    lines.append(cmp_row("XGBoost (5 seeds)",      xgb_agg))
    lines.append(cmp_row("Random Forest (5 seeds)", rf_agg))

    for cond_key, label in [
        ("plain_nam",       "NAM plain (5 seeds)"),
        ("concurvity_only", "NAM concurvity-only (5 seeds)"),
        ("sparsity_only",   "NAM sparsity-only (5 seeds)"),
        ("sparsity_conc",   "NAM sparsity+conc (5 seeds)"),
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
        if m is None or (isinstance(m, float) and np.isnan(m)):
            return "—"
        if s is None or s == 0.0 or (isinstance(s, float) and np.isnan(s)):
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

    rows.append(make_row("XGBoost",       xgb_agg, "5 seeds, grid search"))
    rows.append(make_row("Random Forest", rf_agg,  "5 seeds, grid search"))

    for cond_key, label, note in [
        ("plain_nam",       "NAM (plain)",                  "λs=0, λc=0"),
        ("concurvity_only", "NAM (concurvity only)",        "λs=0, λc=3"),
        ("sparsity_only",   "NAM (sparsity only)",          "λs=?, λc=0"),
        ("sparsity_conc",   "NAM (sparsity + concurvity)",  "λs=?, λc=3 ★"),
    ]:
        nd = nam.get(cond_key, {})
        rows.append(make_row(label, nd, note) if nd else
                    f"| {label:<38} | TODO | | | | {note} |")

    md = textwrap.dedent(f"""\
        # Baseline Comparison — Chest X-ray (17-dim BiomedCLIP v4)

        All models trained on the same 17 BiomedCLIP concept-score features.
        Three-way classification: Normal (0) / Bacteria (1) / Virus (2).
        Train/val/test split: patient-disjoint 4658 / ~932 / 1198
        (GroupShuffleSplit random_state=42, groups=patient_ids).
        XGBoost and Random Forest: 5 seeds (42–46), hyperparameters tuned on the
        val set using PredefinedSplit + GridSearchCV (balanced accuracy criterion).
        NAM results from chest X-ray train_final.py runs (5 seeds each).

        ★ = primary NAM condition (sparsity + concurvity — pending train_final.py
            implementation for chest X-ray).

        | Metric definitions: AUC = OvR weighted, Macro F1 = unweighted class average.

        """) + "\n".join(rows) + textwrap.dedent("""

        ## Notes

        - LR run with a single seed (random_state=42) if available.
        - XGBoost final refit uses the winning hyperparameters from grid search
          on the full train pool (train_final + val).
        - Random Forest final refit on the full train pool; class_weight='balanced'
          in the constructor — no explicit sample_weight for refit.
        - NAM weighted AUC matches weighted_auc_ovr from train_final.py outputs.
        """)

    (out_dir / "baseline_comparison.md").write_text(md, encoding="utf-8")
    print(f"\nComparison saved -> {out_dir / 'baseline_comparison.md'}")


# ── Per-seed checkpoint helpers ───────────────────────────────────────────────

def _append_to_csv(path: pathlib.Path, new_df: pd.DataFrame) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def _done_seeds(csv_path: pathlib.Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        return set(pd.read_csv(csv_path)["seed"].tolist())
    except Exception:
        return set()


def _flush_seed(model_tag: str, metrics: dict, hparams: dict,
                out_dir: pathlib.Path) -> None:
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


# ── Pre-run sanity checks ─────────────────────────────────────────────────────

def run_sanity_checks(X, y_int, patient_ids, train_pool_idx, test_idx,
                      train_rel, val_rel, le):
    """Seven sanity checks — mirrors train_final.py run_sanity_checks."""
    print("\n" + "=" * 65)
    print("PRE-RUN SANITY CHECKS")
    print("=" * 65)

    # 1. Feature shape
    assert X.shape == (5856, N_FEATURES), \
        f"[1] Feature shape {X.shape} != (5856, {N_FEATURES})"
    print(f"  [1] Feature shape: {X.shape}  ✓")

    # 2. Split sizes and patient non-overlap
    X_pool  = X[train_pool_idx]
    y_pool  = y_int[train_pool_idx]
    pid_pool = patient_ids[train_pool_idx]
    X_test  = X[test_idx]
    y_test  = y_int[test_idx]
    pid_test = patient_ids[test_idx]
    assert len(train_pool_idx) == 4658 and len(test_idx) == 1198
    verify_no_patient_overlap(pid_pool, pid_test, "train_pool", "test")
    print(f"  [2] Split: train_pool={len(train_pool_idx)}, test={len(test_idx)}, "
          f"patient_overlap=0  ✓")

    # 3. Val split is carved from train_pool only; no test leakage
    assert max(val_rel) < len(train_pool_idx), \
        "[3] val_rel indices escape train_pool bounds"
    assert max(train_rel) < len(train_pool_idx), \
        "[3] train_rel indices escape train_pool bounds"
    pid_tf  = pid_pool[train_rel]
    pid_val = pid_pool[val_rel]
    verify_no_patient_overlap(pid_tf, pid_val, "train_final", "val")
    print(f"  [3] Val carve: train_final={len(train_rel)}, val={len(val_rel)}, "
          f"patient_overlap=0  ✓")

    # 4. Label encoder: exactly 3 classes
    assert list(le.classes_) == sorted(["normal", "bacteria", "virus"]), \
        f"[4] Unexpected classes: {list(le.classes_)}"
    print(f"  [4] Label encoder classes: {list(le.classes_)}  ✓")

    # 5. Class balance on train_final fold
    y_tf = y_pool[train_rel]
    counts = np.bincount(y_tf, minlength=N_CLASSES)
    print(f"  [5] train_final class counts (N={len(y_tf)}): "
          f"{dict(zip(le.classes_, counts.tolist()))}")

    # 6. Class balance on val fold
    y_val = y_pool[val_rel]
    counts_val = np.bincount(y_val, minlength=N_CLASSES)
    print(f"  [6] val class counts (N={len(y_val)}): "
          f"{dict(zip(le.classes_, counts_val.tolist()))}")

    # 7. X_test shape (just a shape check; data already loaded)
    assert X_test.shape == (1198, N_FEATURES), \
        f"[7] X_test shape {X_test.shape} != (1198, {N_FEATURES})"
    print(f"  [7] X_test shape: {X_test.shape}  ✓")

    print("=" * 65)
    print("All sanity checks passed.  Ready to train.\n")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",  choices=["xgboost", "rf", "both"], default="both")
    p.add_argument("--seeds",  type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--skip-tuning", action="store_true",
                   help="Skip grid search and use default hyperparameters")
    p.add_argument("--overwrite", action="store_true",
                   help="Delete existing per-seed CSVs and start fresh (no resume)")
    p.add_argument("--sanity-only", action="store_true",
                   help="Run sanity checks then exit without training")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

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

    X, y_int, labels_subtype, patient_ids, train_pool_idx, test_idx, le = load_data()

    X_pool   = X[train_pool_idx]
    y_pool   = y_int[train_pool_idx]
    pid_pool = patient_ids[train_pool_idx]
    X_test   = X[test_idx]
    y_test   = y_int[test_idx]

    verify_no_patient_overlap(patient_ids[train_pool_idx], patient_ids[test_idx],
                              "train_pool", "test")
    print("No patient overlap between train pool and test  ✓")

    # Fixed val split (deterministic, same across all seeds)
    train_rel, val_rel = carve_val_split(X_pool, y_pool, pid_pool)

    run_sanity_checks(X, y_int, patient_ids, train_pool_idx, test_idx,
                      train_rel, val_rel, le)

    if args.sanity_only:
        print("--sanity-only flag set.  Exiting before training.")
        return

    run_xgb = args.model in ("xgboost", "both")
    run_rf  = args.model in ("rf",      "both")

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
                    X_pool, y_pool, pid_pool, X_test, y_test, le,
                    seed, args.skip_tuning,
                )
                elapsed = time.time() - t0
                print(f"  Test results  bal_acc={metrics['balanced_acc']:.4f}"
                      f"  macro_f1={metrics['macro_f1']:.4f}"
                      f"  auc={metrics['auc_weighted']:.4f}"
                      f"  top1={metrics['top1_acc']:.4f}  [{elapsed:.0f}s]",
                      flush=True)
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
                    X_pool, y_pool, pid_pool, X_test, y_test, le,
                    seed, args.skip_tuning,
                )
                elapsed = time.time() - t0
                print(f"  Test results  bal_acc={metrics['balanced_acc']:.4f}"
                      f"  macro_f1={metrics['macro_f1']:.4f}"
                      f"  auc={metrics['auc_weighted']:.4f}"
                      f"  top1={metrics['top1_acc']:.4f}  [{elapsed:.0f}s]",
                      flush=True)
                _flush_seed("rf", metrics, hparams, OUT_DIR)

    total_elapsed = time.time() - total_t0
    print(f"\nAll seeds complete in {total_elapsed:.0f}s")

    # ── Rebuild summary files from all completed seeds ─────────────────────────
    _rebuild_summaries(OUT_DIR)
    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
