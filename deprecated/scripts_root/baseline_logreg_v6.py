"""
Logistic regression baseline on BiomedCLIP v6 concept score features (24 features).

Identical methodology to scripts/baseline_logreg.py (v5/72-feature baseline):
  - StandardScaler fit on train partition only
  - class_weight='balanced' (multinomial, lbfgs, max_iter=2000, C=1.0)

Outputs → reports/baselines/logreg_biomedclip_v6/

Run from project root after extract_features_biomedclip_v6.py:
    python scripts/baseline_logreg_v6.py
"""

from __future__ import annotations

import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
OUT_DIR       = "reports/baselines/logreg_biomedclip_v6"

# v5 reference numbers for comparison
V5_METRICS = {
    "balanced_accuracy": 0.608,
    "macro_f1":          0.485,
    "auc_ovr_weighted":  0.891,
}
LARGE_DROP_THRESHOLD = 0.04   # flag if v6 bal_acc drops more than this vs v5

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading features...")
feat      = np.load(FEATURES_PATH, allow_pickle=True)
scores    = feat["scores"]        # (10015, 24)
labels    = feat["labels"]
image_ids = feat["image_ids"]

assert scores.shape == (10015, 24), f"Unexpected scores shape: {scores.shape}"
assert len(labels) == len(image_ids) == scores.shape[0]

print("Loading splits...")
split     = np.load(SPLITS_PATH)
train_idx = split["train_idx"]
test_idx  = split["test_idx"]

assert len(np.intersect1d(train_idx, test_idx)) == 0
assert len(np.union1d(train_idx, test_idx)) == scores.shape[0]

X_train, X_test = scores[train_idx], scores[test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]
ids_test        = image_ids[test_idx]


def class_dist(y: np.ndarray) -> dict:
    vals, counts = np.unique(y, return_counts=True)
    return {v: c for v, c in zip(vals, counts)}


print(f"\nTrain: {len(y_train)} images")
for cls, cnt in sorted(class_dist(y_train).items()):
    print(f"  {cls}: {cnt}")

print(f"\nTest:  {len(y_test)} images")
for cls, cnt in sorted(class_dist(y_test).items()):
    print(f"  {cls}: {cnt}")

# ── Standardise ───────────────────────────────────────────────────────────────
print("\nStandardising features (z-score, fit on train)...")
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training logistic regression (multinomial, lbfgs, balanced weights)...")
clf = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
    C=1.0,
    random_state=42,
)
clf.fit(X_train, y_train)
print("Training complete.")

# ── Predict ───────────────────────────────────────────────────────────────────
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
classes = clf.classes_

# ── Aggregate metrics ─────────────────────────────────────────────────────────
bal_acc     = balanced_accuracy_score(y_test, y_pred)
macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
top1_acc    = accuracy_score(y_test, y_pred)
auc_ovr     = roc_auc_score(y_test, y_proba, multi_class="ovr",
                             average="weighted", labels=classes)

# ── Per-class metrics ─────────────────────────────────────────────────────────
report_dict = classification_report(y_test, y_pred, labels=classes,
                                    output_dict=True, zero_division=0)
report_df = (
    pd.DataFrame(report_dict).T.loc[list(classes)]
    .astype({"support": int})
    .sort_values("support", ascending=False)
)

per_class_auc = {}
for i, cls in enumerate(classes):
    y_bin = (y_test == cls).astype(int)
    per_class_auc[cls] = roc_auc_score(y_bin, y_proba[:, i])

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm      = confusion_matrix(y_test, y_pred, labels=classes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_df   = pd.DataFrame(cm_norm, index=classes, columns=classes)

# ── v5 vs v6 comparison ───────────────────────────────────────────────────────
v6_metrics = {
    "balanced_accuracy": bal_acc,
    "macro_f1":          macro_f1,
    "auc_ovr_weighted":  auc_ovr,
}
comp_lines = [
    "Logistic Regression: v5 (72-feature) vs v6 (24-feature)",
    f"  Balanced accuracy:  v5 {V5_METRICS['balanced_accuracy']:.3f} -> "
    f"v6 {bal_acc:.3f}  "
    f"(Δ {bal_acc - V5_METRICS['balanced_accuracy']:+.3f})",
    f"  Macro F1:           v5 {V5_METRICS['macro_f1']:.3f} -> "
    f"v6 {macro_f1:.3f}  "
    f"(Δ {macro_f1 - V5_METRICS['macro_f1']:+.3f})",
    f"  Multiclass AUC:     v5 {V5_METRICS['auc_ovr_weighted']:.3f} -> "
    f"v6 {auc_ovr:.3f}  "
    f"(Δ {auc_ovr - V5_METRICS['auc_ovr_weighted']:+.3f})",
]
bal_acc_drop = V5_METRICS["balanced_accuracy"] - bal_acc
if bal_acc_drop > LARGE_DROP_THRESHOLD:
    comp_lines.append(
        f"\n  *** WARNING: balanced accuracy dropped {bal_acc_drop:.3f} "
        f"(threshold {LARGE_DROP_THRESHOLD}) — investigate feature consolidation ***"
    )

auc_block = "Per-class AUC (OvR):\n"
for cls in report_df.index:
    auc_block += f"  {cls:6s}: {per_class_auc[cls]:.4f}\n"

HEADER = "==== Logistic Regression Baseline (BiomedCLIP v6 features, 24 concepts, balanced) ===="

agg_block = (
    f"Aggregate metrics:\n"
    f"  Balanced accuracy            : {bal_acc:.4f}\n"
    f"  Macro F1                     : {macro_f1:.4f}\n"
    f"  Weighted F1                  : {weighted_f1:.4f}\n"
    f"  Top-1 accuracy               : {top1_acc:.4f}\n"
    f"  Multiclass AUC (OvR weighted): {auc_ovr:.4f}\n"
)
per_class_block  = "Per-class metrics (sorted by support desc):\n"
per_class_block += report_df.to_string(float_format="%.4f") + "\n"
cm_block  = "Confusion matrix (row-normalised, rows=true, cols=pred):\n"
cm_block += cm_df.to_string(float_format="%.3f") + "\n"
comp_block = "\n".join(comp_lines)

full_report = "\n".join([
    HEADER, "", agg_block, per_class_block, auc_block, cm_block, comp_block,
    f"\nOutputs → {OUT_DIR}/",
])
print("\n" + full_report)

# ── Save artifacts ────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "metrics_summary.txt"), "w", encoding="utf-8") as f:
    f.write(full_report + "\n")

pd.DataFrame([{
    "balanced_accuracy": bal_acc,
    "macro_f1":          macro_f1,
    "weighted_f1":       weighted_f1,
    "top1_accuracy":     top1_acc,
    "auc_ovr_weighted":  auc_ovr,
}]).to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

report_df.to_csv(os.path.join(OUT_DIR, "classification_report.csv"))

pd.DataFrame(cm, index=classes, columns=classes).to_csv(
    os.path.join(OUT_DIR, "confusion_matrix.csv")
)

prob_cols = {f"prob_{c}": y_proba[:, i] for i, c in enumerate(classes)}
pd.DataFrame({
    "image_id": ids_test, "true_label": y_test, "predicted_label": y_pred,
    **prob_cols,
}).to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

pd.DataFrame([{
    "metric": k,
    "v5":     V5_METRICS[k],
    "v6":     v6_metrics[k],
    "delta":  v6_metrics[k] - V5_METRICS[k],
} for k in V5_METRICS]).to_csv(
    os.path.join(OUT_DIR, "comparison_v5_vs_v6.csv"), index=False
)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=classes,
            yticklabels=classes, cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix (row-normalised)\nLogReg baseline — BiomedCLIP v6")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)

metrics_plot = report_df[["precision", "recall", "f1-score"]].copy()
x, width = np.arange(len(metrics_plot)), 0.25
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, metrics_plot["precision"], width, label="Precision")
ax.bar(x,         metrics_plot["recall"],    width, label="Recall")
ax.bar(x + width, metrics_plot["f1-score"],  width, label="F1")
ax.set_xticks(x)
ax.set_xticklabels(metrics_plot.index, rotation=30, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Per-class Metrics — LogReg baseline (BiomedCLIP v6, balanced)")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "per_class_metrics.png"), dpi=150)
plt.close(fig)

print("Done.")
