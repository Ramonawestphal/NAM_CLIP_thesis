"""
generate_thesis_tables.py
Generate final results tables from trained NAM checkpoints for thesis supervisor.

Outputs -> results/thesis_tables/
  per_class_metrics.csv  — 28 rows (4 conditions × 7 classes), mean±std across 5 seeds
  aggregate_metrics.csv  — 4 rows, one per condition
  per_seed_full.csv      — 20 rows (4 conditions × 5 seeds)
  summary.txt            — human-readable plain-text version

Run from project root:
    python scripts/generate_thesis_tables.py
"""

from __future__ import annotations

import ast
import os
import pickle
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.models.nam_multiclass import NAMMulticlass
from src.models.concurvity import multiclass_concurvity
from src.models.sparsity import feature_group_norms

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
SWEEP_CSV     = "reports/nam/v6_sweep/sweep_results.csv"
OUT_DIR       = "results/thesis_tables"

# ── Fixed hyperparameters (sweep winner config 9) ─────────────────────────────
SEEDS       = [42, 43, 44, 45, 46]
N_FEATURES  = 24
N_CLASSES   = 7
ZERO_THR    = 1e-8   # for n_active (matches plot_shape_functions_final.py)
MAJORITY    = 3      # out of 5 seeds → concept "eliminated" in majority

# ── Condition registry ─────────────────────────────────────────────────────────
# get_ckpt(seed)   → absolute-relative path to best_model.pt
# get_scaler(seed) → absolute-relative path to scaler.pkl
CONDITIONS = [
    {
        "name":       "plain_nam",
        "lambda_s":   0.0,
        "lambda_c":   0.0,
        "get_ckpt":   lambda s: f"reports/nam/v6_sparsity_sweep/lam_0.0/seed_{s}/best_model.pt",
        "get_scaler": lambda s: "reports/nam/v6_sparsity_sweep/lam_0.0/scaler.pkl",
    },
    {
        "name":       "concurvity_only_lc1",
        "lambda_s":   0.0,
        "lambda_c":   1.0,
        "get_ckpt":   lambda s: f"reports/nam/v6_final/seed_{s}/best_model.pt",
        "get_scaler": lambda s: "reports/nam/v6_final/scaler.pkl",
    },
    {
        "name":       "sparsity_only_lc0_warmft",
        "lambda_s":   23.7,
        "lambda_c":   0.0,
        "get_ckpt":   lambda s: f"results/final_models/sparsity_only_lc0_warmft/seed{s}/seed_{s}/best_model.pt",
        "get_scaler": lambda s: f"results/final_models/sparsity_only_lc0_warmft/seed{s}/scaler.pkl",
    },
    {
        "name":       "sparsity_conc_lc1_warmft",
        "lambda_s":   12.0,
        "lambda_c":   1.0,
        "get_ckpt":   lambda s: f"results/final_models/sparsity_conc_lc1_warmft/seed{s}/seed_{s}/best_model.pt",
        "get_scaler": lambda s: f"results/final_models/sparsity_conc_lc1_warmft/seed{s}/scaler.pkl",
    },
]

# ── Load architecture from sweep CSV ──────────────────────────────────────────
print("Reading sweep winner config ...")
sweep_df = pd.read_csv(SWEEP_CSV)
cfg = sweep_df[sweep_df["config_id"] == 9].iloc[0]
HIDDEN_DIMS = tuple(ast.literal_eval(cfg["hidden"]))
DROPOUT     = float(cfg["dropout"])
print(f"  Architecture: hidden={list(HIDDEN_DIMS)}, dropout={DROPOUT}")

# ── Load features and splits ───────────────────────────────────────────────────
print("Loading features and splits ...")
feat         = np.load(FEATURES_PATH, allow_pickle=True)
X_all        = feat["scores"]
y_all        = feat["labels"]
lesion_ids   = feat["lesion_ids"]
concept_names = feat["concept_ids"].tolist()
assert X_all.shape == (10015, N_FEATURES)

split     = np.load(SPLITS_PATH)
train_idx = split["train_idx"]
test_idx  = split["test_idx"]

X_train_raw      = X_all[train_idx]
y_train          = y_all[train_idx]
lesion_ids_train = lesion_ids[train_idx]
X_test_raw       = X_all[test_idx]
y_test           = y_all[test_idx]

class_names = sorted(np.unique(y_all).tolist())
assert len(class_names) == N_CLASSES

# Encode labels
y_test_enc = np.array([class_names.index(c) for c in y_test], dtype=np.int64)

# ── Reconstruct val split (deterministic — same random_state as training) ─────
print("Reconstructing val split (random_state=42) ...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
_train_rel, val_rel = next(gss.split(X_train_raw, y_train, groups=lesion_ids_train))
X_val_raw = X_train_raw[val_rel]
y_val     = y_train[val_rel]
print(f"  val={len(y_val)}, test={len(y_test)}")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Per-seed data collection ───────────────────────────────────────────────────
# per_seed_data[condition_name][seed] = dict of all per-seed metrics
per_seed_data: dict[str, dict[int, dict]] = {}

for cond in CONDITIONS:
    cname = cond["name"]
    print(f"\n{'='*60}")
    print(f"Condition: {cname}  (ls={cond['lambda_s']}, lc={cond['lambda_c']})")
    per_seed_data[cname] = {}

    for seed in SEEDS:
        ckpt_path   = cond["get_ckpt"](seed)
        scaler_path = cond["get_scaler"](seed)

        # Validate paths
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")

        # Load scaler and scale data
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_test_sc = scaler.transform(X_test_raw).astype(np.float32)
        X_val_sc  = scaler.transform(X_val_raw).astype(np.float32)
        X_test_t  = torch.tensor(X_test_sc)
        X_val_t   = torch.tensor(X_val_sc)

        # Build model and load weights
        model = NAMMulticlass(
            n_features=N_FEATURES,
            num_classes=N_CLASSES,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
            concept_names=concept_names,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # ── Test-set inference ────────────────────────────────────────────────
        with torch.no_grad():
            logits, shape_outs_test = model(X_test_t, return_shape_outputs=True)
            proba = torch.softmax(logits, dim=1).numpy()
        preds_enc = logits.argmax(dim=1).numpy()
        y_pred    = [class_names[i] for i in preds_enc]

        auc_weighted = roc_auc_score(
            y_test, proba, multi_class="ovr", average="weighted", labels=class_names
        )
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # Per-class metrics
        report = classification_report(
            y_test, y_pred, labels=class_names, output_dict=True, zero_division=0
        )
        per_class: dict[str, dict] = {}
        for i, cls in enumerate(class_names):
            y_bin = (y_test == cls).astype(int)
            cls_auc = roc_auc_score(y_bin, proba[:, i])
            per_class[cls] = {
                "auc":       cls_auc,
                "precision": report[cls]["precision"],
                "recall":    report[cls]["recall"],
                "f1":        report[cls]["f1-score"],
                "support":   int(report[cls]["support"]),
            }

        # ── Val-set R_perp ────────────────────────────────────────────────────
        with torch.no_grad():
            _, val_shape_outs = model(X_val_t, return_shape_outputs=True)
            r_perp = multiclass_concurvity(val_shape_outs).item()

        # ── n_active ──────────────────────────────────────────────────────────
        norms     = feature_group_norms(model)
        n_active  = sum(1 for v in norms.values() if v > ZERO_THR)
        # Per-concept active flags (in concept_names order)
        active_flags = {c: (norms.get(c, 0.0) > ZERO_THR) for c in concept_names}

        per_seed_data[cname][seed] = {
            "auc_weighted": auc_weighted,
            "bal_acc":      bal_acc,
            "r_perp":       r_perp,
            "n_active":     n_active,
            "per_class":    per_class,
            "active_flags": active_flags,
        }

        print(f"  seed {seed}: n_active={n_active:2d}  "
              f"AUC={auc_weighted:.4f}  bal_acc={bal_acc:.4f}  R_perp={r_perp:.4f}")


# ── Build output dataframes ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Building output tables ...")

# ── 1. per_seed_full.csv ──────────────────────────────────────────────────────
per_seed_rows = []
for cond in CONDITIONS:
    cname = cond["name"]
    for seed in SEEDS:
        d = per_seed_data[cname][seed]
        row = {
            "condition":         cname,
            "seed":              seed,
            "lambda_s":          cond["lambda_s"],
            "lambda_c":          cond["lambda_c"],
            "n_active":          d["n_active"],
            "test_auc_weighted": round(d["auc_weighted"], 4),
            "test_balanced_acc": round(d["bal_acc"],      4),
            "R_perp":            round(d["r_perp"],       4),
        }
        # Per-class AUC and F1
        for cls in class_names:
            row[f"test_auc_{cls}"] = round(d["per_class"][cls]["auc"], 4)
            row[f"test_f1_{cls}"]  = round(d["per_class"][cls]["f1"],  4)
        per_seed_rows.append(row)

per_seed_df = pd.DataFrame(per_seed_rows)
per_seed_path = os.path.join(OUT_DIR, "per_seed_full.csv")
per_seed_df.to_csv(per_seed_path, index=False)
print(f"  Saved: {per_seed_path}")

# ── 2. per_class_metrics.csv ──────────────────────────────────────────────────
pc_rows = []
for cond in CONDITIONS:
    cname = cond["name"]
    for cls in class_names:
        aucs  = [per_seed_data[cname][s]["per_class"][cls]["auc"]       for s in SEEDS]
        f1s   = [per_seed_data[cname][s]["per_class"][cls]["f1"]        for s in SEEDS]
        precs = [per_seed_data[cname][s]["per_class"][cls]["precision"] for s in SEEDS]
        recs  = [per_seed_data[cname][s]["per_class"][cls]["recall"]    for s in SEEDS]
        sup   = per_seed_data[cname][SEEDS[0]]["per_class"][cls]["support"]  # invariant
        pc_rows.append({
            "condition":        cname,
            "class":            cls,
            "support":          sup,
            "n_seeds":          len(SEEDS),
            "auc_mean":         round(float(np.mean(aucs)),  4),
            "auc_std":          round(float(np.std(aucs)),   4),
            "f1_mean":          round(float(np.mean(f1s)),   4),
            "f1_std":           round(float(np.std(f1s)),    4),
            "precision_mean":   round(float(np.mean(precs)), 4),
            "precision_std":    round(float(np.std(precs)),  4),
            "recall_mean":      round(float(np.mean(recs)),  4),
            "recall_std":       round(float(np.std(recs)),   4),
        })

pc_df = pd.DataFrame(pc_rows)
pc_path = os.path.join(OUT_DIR, "per_class_metrics.csv")
pc_df.to_csv(pc_path, index=False)
print(f"  Saved: {pc_path}")

# ── 3. aggregate_metrics.csv ─────────────────────────────────────────────────
# Majority-rule eliminated concepts per condition
def majority_eliminated(cname: str) -> str:
    """Concepts eliminated (norm <= ZERO_THR) in >= MAJORITY seeds."""
    elim = []
    for concept in concept_names:
        n_active_for_concept = sum(
            1 for s in SEEDS
            if per_seed_data[cname][s]["active_flags"].get(concept, False)
        )
        if n_active_for_concept < MAJORITY:
            elim.append(concept)
    return ";".join(elim)

agg_rows = []
for cond in CONDITIONS:
    cname   = cond["name"]
    n_acts  = [per_seed_data[cname][s]["n_active"]     for s in SEEDS]
    aucs    = [per_seed_data[cname][s]["auc_weighted"]  for s in SEEDS]
    baccs   = [per_seed_data[cname][s]["bal_acc"]       for s in SEEDS]
    rperps  = [per_seed_data[cname][s]["r_perp"]        for s in SEEDS]
    agg_rows.append({
        "condition":           cname,
        "lambda_s":            cond["lambda_s"],
        "lambda_c":            cond["lambda_c"],
        "n_active_mean":       round(float(np.mean(n_acts)), 2),
        "n_active_std":        round(float(np.std(n_acts)),  2),
        "auc_weighted_mean":   round(float(np.mean(aucs)),   4),
        "auc_weighted_std":    round(float(np.std(aucs)),    4),
        "balanced_acc_mean":   round(float(np.mean(baccs)),  4),
        "balanced_acc_std":    round(float(np.std(baccs)),   4),
        "R_perp_mean":         round(float(np.mean(rperps)), 4),
        "R_perp_std":          round(float(np.std(rperps)),  4),
        "eliminated_concepts": majority_eliminated(cname),
    })

agg_df = pd.DataFrame(agg_rows)
agg_path = os.path.join(OUT_DIR, "aggregate_metrics.csv")
agg_df.to_csv(agg_path, index=False)
print(f"  Saved: {agg_path}")


# ── 4. summary.txt ────────────────────────────────────────────────────────────
# Build display name map
DISPLAY = {
    "plain_nam":                "plain_nam",
    "concurvity_only_lc1":     "concurvity_only (lc=1.0)",
    "sparsity_only_lc0_warmft": "sparsity_only (ls=23.7)",
    "sparsity_conc_lc1_warmft": "sparsity+conc (ls=12,lc=1)",
}

def fmt(mean: float, std: float, dec: int = 3) -> str:
    fmt_str = f"{{:.{dec}f}}"
    return f"{fmt_str.format(mean)} ± {fmt_str.format(std)}"

def fmt_nact(mean: float, std: float) -> str:
    return f"{mean:.1f} ± {std:.1f}"

lines = []

# ── Aggregate table ───────────────────────────────────────────────────────────
lines.append("=" * 76)
lines.append("Aggregate results, 5 seeds per condition")
lines.append("=" * 76)
lines.append("")
hdr = f"{'Condition':<30} {'n_active':<13} {'AUC (w)':<14} {'Bal.Acc':<14} {'R_perp':<14}"
lines.append(hdr)
lines.append("-" * 76)
for row in agg_rows:
    dn = DISPLAY[row["condition"]]
    col_n    = fmt_nact(row["n_active_mean"],     row["n_active_std"])
    col_auc  = fmt(row["auc_weighted_mean"],      row["auc_weighted_std"])
    col_bacc = fmt(row["balanced_acc_mean"],      row["balanced_acc_std"])
    col_rp   = fmt(row["R_perp_mean"],            row["R_perp_std"])
    lines.append(f"{dn:<30} {col_n:<13} {col_auc:<14} {col_bacc:<14} {col_rp:<14}")

lines.append("")

# Eliminated concepts per sparsity condition
for cname in ["sparsity_only_lc0_warmft", "sparsity_conc_lc1_warmft"]:
    dn   = DISPLAY[cname]
    elim = majority_eliminated(cname).replace(";", ", ")
    lines.append(f"Eliminated in {dn} (>=3/5 seeds):")
    if elim:
        # wrap at 72 chars
        words = elim.split(", ")
        cur_line = "  "
        for w in words:
            if len(cur_line) + len(w) + 2 > 74:
                lines.append(cur_line.rstrip(", "))
                cur_line = "  "
            cur_line += w + ", "
        if cur_line.strip(", "):
            lines.append(cur_line.rstrip(", "))
    lines.append("")

# ── Per-class AUC table ───────────────────────────────────────────────────────
lines.append("")
lines.append("=" * 76)
lines.append("Per-class AUC (OvR, mean ± std across 5 seeds)")
lines.append("=" * 76)
lines.append("")

# Header
cond_headers = [DISPLAY[c["name"]] for c in CONDITIONS]
lines.append(f"{'Class':<10} {'Supp':>6}   "
             + "   ".join(f"{h:<16}" for h in cond_headers))
lines.append("-" * 76)

# Sort classes by support descending (use first condition)
cls_support = {cls: pc_df[(pc_df["condition"] == "plain_nam") &
                           (pc_df["class"] == cls)]["support"].iloc[0]
               for cls in class_names}
sorted_classes = sorted(class_names, key=lambda c: -cls_support[c])

for cls in sorted_classes:
    sup = cls_support[cls]
    cells = []
    for cond in CONDITIONS:
        cname = cond["name"]
        row = pc_df[(pc_df["condition"] == cname) & (pc_df["class"] == cls)].iloc[0]
        cells.append(fmt(row["auc_mean"], row["auc_std"]))
    lines.append(f"{cls:<10} {int(sup):>6}   " + "   ".join(f"{c:<16}" for c in cells))
lines.append("")

# ── Per-class F1 table ────────────────────────────────────────────────────────
lines.append("")
lines.append("=" * 76)
lines.append("Per-class F1 (mean ± std across 5 seeds)")
lines.append("=" * 76)
lines.append("")
lines.append(f"{'Class':<10} {'Supp':>6}   "
             + "   ".join(f"{h:<16}" for h in cond_headers))
lines.append("-" * 76)
for cls in sorted_classes:
    sup = cls_support[cls]
    cells = []
    for cond in CONDITIONS:
        cname = cond["name"]
        row = pc_df[(pc_df["condition"] == cname) & (pc_df["class"] == cls)].iloc[0]
        cells.append(fmt(row["f1_mean"], row["f1_std"]))
    lines.append(f"{cls:<10} {int(sup):>6}   " + "   ".join(f"{c:<16}" for c in cells))
lines.append("")

# ── Per-class Precision table ─────────────────────────────────────────────────
lines.append("")
lines.append("=" * 76)
lines.append("Per-class Precision (mean ± std across 5 seeds)")
lines.append("=" * 76)
lines.append("")
lines.append(f"{'Class':<10} {'Supp':>6}   "
             + "   ".join(f"{h:<16}" for h in cond_headers))
lines.append("-" * 76)
for cls in sorted_classes:
    sup = cls_support[cls]
    cells = []
    for cond in CONDITIONS:
        cname = cond["name"]
        row = pc_df[(pc_df["condition"] == cname) & (pc_df["class"] == cls)].iloc[0]
        cells.append(fmt(row["precision_mean"], row["precision_std"]))
    lines.append(f"{cls:<10} {int(sup):>6}   " + "   ".join(f"{c:<16}" for c in cells))
lines.append("")

# ── Per-class Recall table ────────────────────────────────────────────────────
lines.append("")
lines.append("=" * 76)
lines.append("Per-class Recall (mean ± std across 5 seeds)")
lines.append("=" * 76)
lines.append("")
lines.append(f"{'Class':<10} {'Supp':>6}   "
             + "   ".join(f"{h:<16}" for h in cond_headers))
lines.append("-" * 76)
for cls in sorted_classes:
    sup = cls_support[cls]
    cells = []
    for cond in CONDITIONS:
        cname = cond["name"]
        row = pc_df[(pc_df["condition"] == cname) & (pc_df["class"] == cls)].iloc[0]
        cells.append(fmt(row["recall_mean"], row["recall_std"]))
    lines.append(f"{cls:<10} {int(sup):>6}   " + "   ".join(f"{c:<16}" for c in cells))
lines.append("")

summary_text = "\n".join(lines)
summary_path = os.path.join(OUT_DIR, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text + "\n")
print(f"  Saved: {summary_path}")


# ── Sanity checks ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Sanity checks:")

CHECKS = [
    ("plain_nam",                "auc_weighted_mean",  0.855, 0.01),
    ("concurvity_only_lc1",      "auc_weighted_mean",  0.840, 0.01),
    ("sparsity_conc_lc1_warmft", "auc_weighted_mean",  0.812, 0.01),
    ("plain_nam",                "n_active_mean",       24.0,  2.0),
    ("concurvity_only_lc1",      "n_active_mean",       24.0,  2.0),
    ("sparsity_only_lc0_warmft", "n_active_mean",       11.0,  2.0),
    ("sparsity_conc_lc1_warmft", "n_active_mean",       11.0,  2.0),
]
all_ok = True
for cname, metric, expected, tol in CHECKS:
    row = agg_df[agg_df["condition"] == cname].iloc[0]
    actual = float(row[metric])
    ok = abs(actual - expected) <= tol
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {cname:35s} {metric:25s}: {actual:.4f}  (expected ~{expected}, tol ±{tol})")
    if not ok:
        all_ok = False

if not all_ok:
    print("\n  *** SANITY CHECK FAILED — review metric computation before using tables ***")
    sys.exit(1)
else:
    print("\n  All sanity checks passed.")

# ── File listing ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Output files:")
for fname in ["per_class_metrics.csv", "aggregate_metrics.csv",
              "per_seed_full.csv", "summary.txt"]:
    fpath = os.path.join(OUT_DIR, fname)
    size  = os.path.getsize(fpath)
    print(f"  {fpath}  ({size:,} bytes)")

# ── Print summary.txt to stdout ───────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY.TXT CONTENTS")
print("=" * 60)
print(summary_text)
