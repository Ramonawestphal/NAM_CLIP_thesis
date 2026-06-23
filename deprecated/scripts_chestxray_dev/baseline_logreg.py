"""
Logistic regression baseline on 17-dim BiomedCLIP v4 concept scores.
Chest X-ray three-way classification (Normal / Bacteria / Virus).

Mirrors HAM10000's scripts/baseline_logreg_v6.py convention:
  - Fits on the FULL train pool (train_pool_idx, 4658 samples) — no val split
  - StandardScaler fit on train pool, transform applied to test set
  - Fixed hyperparameters: C=1.0, max_iter=2000, class_weight='balanced',
    solver='lbfgs' (softmax/multinomial is automatic for >2 classes in sklearn>=1.5)
  - 5 seeds (42–46) for table format consistency with XGBoost/RF;
    results are identical across seeds (lbfgs is deterministic given fixed data)

Hard isolation rules
────────────────────
- Do NOT write to results/baselines_ml/ (HAM10000's directory).
- Do NOT modify any existing XGB/RF baseline results.
- Test set loaded once, touched only in per-seed evaluation block.

Outputs → results/chestxray/baselines_ml/lr/
    per_seed_metrics.csv    one row per seed (all metrics)
    aggregated_metrics.csv  mean ± std across seeds (+ per-class)
    confusion_matrix.csv    seed-mean row-normalised 3×3
    per_class_metrics.csv   seed-mean per-class precision, recall, F1, AUC
    run_config.json         refit convention, scaler fit set, seeds, hparams

Run from project root:
    python scripts/chestxray/baseline_logreg.py
    python scripts/chestxray/baseline_logreg.py --sanity-only
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH  = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v4.npz"
SPLIT_PATH     = _ROOT / "data/splits/chestxray_outer_split.npz"
LABEL_MAP_PATH = _ROOT / "results/chestxray/architecture_selection/label_mapping.json"
OUT_DIR        = _ROOT / "results/chestxray/baselines_ml/lr"

# ── Constants ─────────────────────────────────────────────────────────────────
N_FEATURES  = 17
N_CLASSES   = 3
SEEDS       = [42, 43, 44, 45, 46]
SUBTYPE_TO_INT = {"normal": 0, "bacteria": 1, "virus": 2}
CLASS_ORDER    = ["normal", "bacteria", "virus"]   # display ordering

# Fixed hyperparameters — no tuning
LR_HPARAMS = dict(
    solver       = "lbfgs",
    max_iter     = 2000,
    class_weight = "balanced",
    C            = 1.0,
    # multi_class removed in sklearn >= 1.5; lbfgs uses softmax (multinomial)
    # automatically for >2 classes
)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_label_mapping() -> dict:
    if LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH, encoding="utf-8") as f:
            return json.load(f)
    return SUBTYPE_TO_INT


def load_data():
    """Load features and splits.

    Returns train_pool and test arrays.  test is returned but must only be
    accessed in the per-seed evaluation block.
    """
    feat          = np.load(FEATURES_PATH, allow_pickle=True)
    X             = feat["scores"].astype(np.float32)   # (5856, 17)
    concept_names = feat["concept_names"].tolist()

    split          = np.load(SPLIT_PATH, allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]
    test_idx       = split["test_idx"]
    labels_subtype = split["labels_subtype"]   # str: "normal"/"bacteria"/"virus"
    patient_ids    = split["patient_ids"]

    # LabelEncoder (alphabetical): bacteria=0, normal=1, virus=2
    le = LabelEncoder()
    le.fit(sorted(np.unique(labels_subtype)))
    y_int = le.transform(labels_subtype)

    return {
        "X":              X,
        "y_int":          y_int,
        "concept_names":  concept_names,
        "train_pool_idx": train_pool_idx,
        "test_idx":       test_idx,
        "patient_ids":    patient_ids,
        "le":             le,
    }


# ── Pre-run sanity checks ─────────────────────────────────────────────────────

def run_sanity_checks(data: dict) -> None:
    X              = data["X"]
    y_int          = data["y_int"]
    train_pool_idx = data["train_pool_idx"]
    test_idx       = data["test_idx"]
    patient_ids    = data["patient_ids"]
    le             = data["le"]

    print("\n" + "=" * 65)
    print("PRE-RUN SANITY CHECKS")
    print("=" * 65)

    # [1] Feature shape
    assert X.shape == (5856, N_FEATURES), \
        f"[1] Feature shape {X.shape} != (5856, {N_FEATURES})"
    print(f"  [1] Feature shape: {X.shape}  ✓")

    # [2] Split sizes and patient non-overlap
    assert len(train_pool_idx) == 4658, \
        f"[2] train_pool len={len(train_pool_idx)}, expected 4658"
    assert len(test_idx) == 1198, \
        f"[2] test len={len(test_idx)}, expected 1198"
    assert len(np.intersect1d(train_pool_idx, test_idx)) == 0, \
        "[2] Index overlap between train_pool and test"
    train_patients = set(patient_ids[train_pool_idx].tolist())
    test_patients  = set(patient_ids[test_idx].tolist())
    assert len(train_patients & test_patients) == 0, \
        "[2] Patient overlap between train_pool and test"
    print(f"  [2] Split: train_pool={len(train_pool_idx)}, test={len(test_idx)}, "
          f"patient_overlap=0  ✓")

    # [3] Label encoder: exactly 3 classes
    assert list(le.classes_) == sorted(["normal", "bacteria", "virus"]), \
        f"[3] Unexpected classes: {list(le.classes_)}"
    print(f"  [3] Label encoder classes: {list(le.classes_)}  ✓")

    # [4] Refit data confirmation: LR fits on train_pool, NOT train_final
    X_pool  = X[train_pool_idx]
    y_pool  = y_int[train_pool_idx]
    counts  = np.bincount(y_pool, minlength=N_CLASSES)
    print(f"  [4] LR fits on train_pool ({len(y_pool)} samples) — "
          f"NO val split (mirrors HAM10000 LR convention)")
    print(f"       Class counts (bacteria/normal/virus): "
          f"{dict(zip(le.classes_, counts.tolist()))}")

    # [5] Val split reference (informational — LR does not use it)
    pid_pool = patient_ids[train_pool_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_rel, val_rel = next(gss.split(X_pool, y_pool, groups=pid_pool))
    print(f"  [5] Val split reference (LR does NOT use — informational only):")
    print(f"       train_final[:10] = {train_rel[:10].tolist()}")
    print(f"       val[:10]         = {val_rel[:10].tolist()}")
    print(f"       train_final size = {len(train_rel)}, val size = {len(val_rel)}")

    # [6] Test shape (loaded but not used before eval block)
    assert X[test_idx].shape == (1198, N_FEATURES), \
        f"[6] X_test shape {X[test_idx].shape} != (1198, {N_FEATURES})"
    print(f"  [6] X_test shape: {X[test_idx].shape}  ✓")

    # [7] StandardScaler fit set: train_pool (mirrors HAM10000 LR)
    print(f"  [7] StandardScaler will be fit on train_pool "
          f"({len(X_pool)} samples) — mirrors HAM10000 baseline_logreg_v6.py")

    print("=" * 65)
    print("All sanity checks passed.  Ready to run.\n")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_seed_metrics(y_test, y_pred, y_proba, le: LabelEncoder) -> dict:
    """Full metric set for one seed evaluation."""
    classes    = le.classes_   # ["bacteria", "normal", "virus"] (alphabetical)
    y_test_str = le.inverse_transform(y_test)

    bal_acc      = balanced_accuracy_score(y_test, y_pred)
    top1_acc     = accuracy_score(y_test, y_pred)
    macro_f1     = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_auc    = roc_auc_score(
        y_test_str, y_proba, multi_class="ovr", average="macro",    labels=classes)
    weighted_auc = roc_auc_score(
        y_test_str, y_proba, multi_class="ovr", average="weighted", labels=classes)

    per_class = {}
    for i, cls in enumerate(classes):
        y_bin = (y_test == i).astype(int)
        p_bin = (y_pred == i).astype(int)
        per_class[cls] = {
            "f1":   f1_score(y_bin, p_bin, zero_division=0),
            "prec": float(np.nan_to_num(
                np.sum((p_bin == 1) & (y_bin == 1)) / max(1, np.sum(p_bin == 1)))),
            "rec":  float(np.sum((p_bin == 1) & (y_bin == 1)) / max(1, np.sum(y_bin == 1))),
            "auc":  roc_auc_score(y_bin, y_proba[:, i]),
        }

    row = {
        "bal_acc":     bal_acc,
        "top1_acc":    top1_acc,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc":   macro_auc,
        "weighted_auc": weighted_auc,
    }
    for cls in CLASS_ORDER:
        m = per_class.get(cls, {})
        row[f"f1_{cls}"]   = m.get("f1",   np.nan)
        row[f"prec_{cls}"] = m.get("prec", np.nan)
        row[f"rec_{cls}"]  = m.get("rec",  np.nan)
        row[f"auc_{cls}"]  = m.get("auc",  np.nan)
    return row


# ── Summary writers ───────────────────────────────────────────────────────────

def write_outputs(seed_rows: list, y_test_all: list, y_pred_all: list,
                  y_proba_all: list, le: LabelEncoder, out_dir: pathlib.Path):
    """Write per_seed_metrics, aggregated_metrics, confusion_matrix, per_class_metrics."""
    df_seed = pd.DataFrame(seed_rows)

    # per_seed_metrics.csv
    df_seed.to_csv(out_dir / "per_seed_metrics.csv", index=False)

    # aggregated_metrics.csv — mean ± std over seeds
    numeric_cols = [c for c in df_seed.columns if c != "seed"]
    agg = {}
    for col in numeric_cols:
        agg[f"mean_{col}"] = df_seed[col].mean()
        agg[f"std_{col}"]  = df_seed[col].std(ddof=1)
    pd.DataFrame([agg]).to_csv(out_dir / "aggregated_metrics.csv", index=False)

    # confusion_matrix.csv — seed-mean, row-normalised
    classes = le.classes_
    cms_norm = []
    for y_t, y_p in zip(y_test_all, y_pred_all):
        cm = confusion_matrix(y_t, y_p, labels=list(range(N_CLASSES)))
        cms_norm.append(cm.astype(float) / cm.sum(axis=1, keepdims=True))
    cm_mean = np.mean(cms_norm, axis=0)
    # Use CLASS_ORDER for index/columns (human-readable)
    class_names_le = [classes[i] for i in range(N_CLASSES)]   # le alphabetical order
    pd.DataFrame(cm_mean, index=class_names_le,
                 columns=class_names_le).to_csv(out_dir / "confusion_matrix.csv")

    # per_class_metrics.csv — seed-mean per-class F1/prec/rec/AUC
    pc_rows = []
    for cls in CLASS_ORDER:
        pc_rows.append({
            "class":     cls,
            "mean_f1":   df_seed[f"f1_{cls}"].mean(),
            "std_f1":    df_seed[f"f1_{cls}"].std(ddof=1),
            "mean_prec": df_seed[f"prec_{cls}"].mean(),
            "std_prec":  df_seed[f"prec_{cls}"].std(ddof=1),
            "mean_rec":  df_seed[f"rec_{cls}"].mean(),
            "std_rec":   df_seed[f"rec_{cls}"].std(ddof=1),
            "mean_auc":  df_seed[f"auc_{cls}"].mean(),
            "std_auc":   df_seed[f"auc_{cls}"].std(ddof=1),
        })
    pd.DataFrame(pc_rows).to_csv(out_dir / "per_class_metrics.csv", index=False)


def write_run_config(out_dir: pathlib.Path, seeds: list) -> None:
    cfg = {
        "model":           "LogisticRegression",
        "refit_convention": (
            "Fit on train_pool (4658 samples, train_pool_idx from "
            "chestxray_outer_split.npz) — mirrors HAM10000 baseline_logreg_v6.py. "
            "No val split used (no hyperparameter tuning)."
        ),
        "scaler_fit_set":  "train_pool (same set as model fit)",
        "seeds":           seeds,
        "hyperparameters": LR_HPARAMS,
        "note": (
            "lbfgs with fixed data is deterministic; all seeds produce "
            "identical results. Seeds 43-46 retained for table format "
            "consistency with XGBoost/RF baselines."
        ),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sanity-only", action="store_true",
                   help="Run sanity checks then exit without training")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds

    print("Loading features and splits ...")
    data = load_data()
    le = data["le"]
    X  = data["X"]
    y_int          = data["y_int"]
    train_pool_idx = data["train_pool_idx"]
    test_idx       = data["test_idx"]

    print(f"Feature shape : {X.shape}")
    print(f"Classes (LE)  : {list(le.classes_)}")
    print(f"Train pool    : {len(train_pool_idx)}, Test: {len(test_idx)}")

    run_sanity_checks(data)

    if args.sanity_only:
        print("--sanity-only set.  Exiting before training.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_run_config(OUT_DIR, seeds)

    # ── Data slices ───────────────────────────────────────────────────────────
    X_pool  = X[train_pool_idx]
    y_pool  = y_int[train_pool_idx]
    X_test  = X[test_idx]
    y_test  = y_int[test_idx]

    # StandardScaler fit on train_pool (mirrors HAM10000 convention)
    scaler  = StandardScaler()
    X_pool_sc = scaler.fit_transform(X_pool)
    X_test_sc = scaler.transform(X_test)

    seed_rows   = []
    y_test_all  = []
    y_pred_all  = []
    y_proba_all = []

    print(f"\nRunning LR on {len(seeds)} seeds: {seeds}")
    print("(lbfgs is deterministic given fixed data — all seeds identical)\n")

    for seed in seeds:
        print(f"  seed={seed} ...", end=" ", flush=True)
        clf = LogisticRegression(random_state=seed, **LR_HPARAMS)
        clf.fit(X_pool_sc, y_pool)

        y_pred  = clf.predict(X_test_sc)
        y_proba = clf.predict_proba(X_test_sc)

        metrics = compute_seed_metrics(y_test, y_pred, y_proba, le)
        metrics["seed"] = seed
        seed_rows.append(metrics)
        y_test_all.append(y_test)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)

        print(f"bal_acc={metrics['bal_acc']:.4f}  macro_f1={metrics['macro_f1']:.4f}"
              f"  auc_wt={metrics['weighted_auc']:.4f}  top1={metrics['top1_acc']:.4f}")

    write_outputs(seed_rows, y_test_all, y_pred_all, y_proba_all, le, OUT_DIR)

    # ── Print aggregate summary ───────────────────────────────────────────────
    df = pd.DataFrame(seed_rows)
    print("\n" + "=" * 65)
    print("Aggregate (mean ± std across seeds):")
    print("=" * 65)
    for col, label in [
        ("bal_acc",      "Balanced accuracy  "),
        ("macro_f1",     "Macro F1           "),
        ("macro_auc",    "Macro AUC (OvR)    "),
        ("weighted_auc", "Weighted AUC (OvR) "),
        ("top1_acc",     "Top-1 accuracy     "),
    ]:
        m = df[col].mean()
        s = df[col].std(ddof=1)
        print(f"  {label}: {m:.4f} ± {s:.4f}")

    print("\nPer-class (seed-mean):")
    for cls in CLASS_ORDER:
        m_f1  = df[f"f1_{cls}"].mean()
        m_rec = df[f"rec_{cls}"].mean()
        m_auc = df[f"auc_{cls}"].mean()
        print(f"  {cls:<10}: F1={m_f1:.4f}  Rec={m_rec:.4f}  AUC={m_auc:.4f}")

    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
