"""
Logistic regression baseline on BiomedCLIP v6b concept score features.

v6b = 24 ensembled embeddings (Radford et al. 2021 template ensemble from v5).

Identical methodology to baseline_logreg.py (v5) and baseline_logreg_v6.py:
  - StandardScaler fit on train partition only
  - class_weight='balanced', multinomial lbfgs, max_iter=2000, C=1.0

Outputs → reports/baselines/logreg_biomedclip_v6b/

Run from project root after build_v6b_ensembled.py:
    python scripts/baseline_logreg_v6b.py
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
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6b.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
OUT_DIR       = "reports/baselines/logreg_biomedclip_v6b"

# Three-way reference numbers
V5_METRICS = {
    "balanced_accuracy": 0.608,
    "macro_f1":          0.485,
    "auc_ovr_weighted":  0.891,
    "top1_accuracy":     0.636,
}
V6_METRICS = {
    "balanced_accuracy": 0.555,
    "macro_f1":          0.448,
    "auc_ovr_weighted":  0.860,
    "top1_accuracy":     0.595,
}

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

# ── Predict + metrics ─────────────────────────────────────────────────────────
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
classes = clf.classes_

bal_acc     = balanced_accuracy_score(y_test, y_pred)
macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
top1_acc    = accuracy_score(y_test, y_pred)
auc_ovr     = roc_auc_score(y_test, y_proba, multi_class="ovr",
                             average="weighted", labels=classes)

report_dict = classification_report(y_test, y_pred, labels=classes,
                                    output_dict=True, zero_division=0)
report_df = (
    pd.DataFrame(report_dict).T.loc[list(classes)]
    .astype({"support": int})
    .sort_values("support", ascending=False)
)

per_class_auc_v6b: dict[str, float] = {}
for i, cls in enumerate(classes):
    per_class_auc_v6b[cls] = roc_auc_score((y_test == cls).astype(int), y_proba[:, i])

cm      = confusion_matrix(y_test, y_pred, labels=classes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_df   = pd.DataFrame(cm_norm, index=classes, columns=classes)

# ── Three-way comparison ──────────────────────────────────────────────────────
v6b_metrics = {
    "balanced_accuracy": bal_acc,
    "macro_f1":          macro_f1,
    "auc_ovr_weighted":  auc_ovr,
    "top1_accuracy":     top1_acc,
}

def _row(label: str, key: str) -> str:
    v5  = V5_METRICS[key]
    v6  = V6_METRICS[key]
    v6b = v6b_metrics[key]
    return f"  {label:<30s}  {v5:.3f}     {v6:.3f}            {v6b:.3f}"

threeway = "\n".join([
    "Three-way comparison:",
    f"  {'Metric':<30s}  {'v5 (72)':>7}  {'v6 (24 rewrite)':>15}  {'v6b (24 ensemble)':>17}",
    _row("Balanced accuracy",     "balanced_accuracy"),
    _row("Macro F1",              "macro_f1"),
    _row("Multiclass AUC (OvR)",  "auc_ovr_weighted"),
    _row("Top-1 accuracy",        "top1_accuracy"),
])

# Per-class AUC delta (v6b vs v5) — read v5 per-class AUC from the scores file
# We compute v5 per-class AUC fresh rather than hardcoding, for accuracy.
v5_data = np.load("data/features/biomedclip/ham10000_concept_scores_v5.npz",
                  allow_pickle=True)
v5_scores_test = v5_data["scores"][test_idx]
# Build a "best column" AUC per class for v5: use the max-AUC column per class
per_class_auc_v5: dict[str, float] = {}
for cls in classes:
    y_bin = (y_test == cls).astype(int)
    col_aucs = [roc_auc_score(y_bin, v5_scores_test[:, col])
                for col in range(v5_scores_test.shape[1])]
    per_class_auc_v5[cls] = max(col_aucs)

# For the logreg comparison we want the model-level per-class AUC from the
# v5 logreg baseline. Re-fit quickly on v5 features.
v5_scores_train = v5_data["scores"][train_idx]
scaler_v5 = StandardScaler()
X_v5_tr = scaler_v5.fit_transform(v5_scores_train)
X_v5_te = scaler_v5.transform(v5_scores_test)
clf_v5 = LogisticRegression(solver="lbfgs", max_iter=2000,
                             class_weight="balanced", C=1.0, random_state=42)
clf_v5.fit(X_v5_tr, labels[train_idx])
proba_v5 = clf_v5.predict_proba(X_v5_te)
per_class_auc_v5_lr: dict[str, float] = {}
for i, cls in enumerate(clf_v5.classes_):
    per_class_auc_v5_lr[cls] = roc_auc_score(
        (y_test == cls).astype(int), proba_v5[:, i]
    )

cls_delta_lines = ["Per-class AUC delta (v6b vs v5 logreg, OvR):"]
for cls in sorted(classes, key=lambda c: -per_class_auc_v5_lr.get(c, 0)):
    v5_auc  = per_class_auc_v5_lr.get(cls, float("nan"))
    v6b_auc = per_class_auc_v6b[cls]
    delta   = v6b_auc - v5_auc
    cls_delta_lines.append(
        f"  {cls:<6s}: v6b={v6b_auc:.4f}  v5={v5_auc:.4f}  (Δ {delta:+.4f})"
    )
cls_delta_block = "\n".join(cls_delta_lines)

delta_v5  = bal_acc - V5_METRICS["balanced_accuracy"]
delta_v6  = bal_acc - V6_METRICS["balanced_accuracy"]
decision  = (
    "*** v6b is the strongest 24-feature candidate — beats v6 by ≥ 0.02 ***"
    if delta_v6 >= 0.02 else
    "v6b beats v6 but by < 0.02 — marginal; consider sticking with v5 (72 features)"
    if delta_v6 > 0 else
    "*** v6b does NOT beat v6 — template ensembling worse than hand-rewriting for this prompt set ***"
)

decision_block = "\n".join([
    "Decision criterion (balanced accuracy):",
    f"  v6b vs v5  delta: {delta_v5:+.3f}",
    f"  v6b vs v6  delta: {delta_v6:+.3f}",
    f"  {decision}",
])

# ── Full report ───────────────────────────────────────────────────────────────
auc_block = "Per-class AUC (OvR, v6b model):\n"
for cls in report_df.index:
    auc_block += f"  {cls:6s}: {per_class_auc_v6b[cls]:.4f}\n"

HEADER = ("==== Logistic Regression Baseline "
          "(BiomedCLIP v6b features, 24 ensembled, balanced) ====")

full_report = "\n".join([
    HEADER, "",
    f"Aggregate metrics:\n"
    f"  Balanced accuracy            : {bal_acc:.4f}\n"
    f"  Macro F1                     : {macro_f1:.4f}\n"
    f"  Weighted F1                  : {weighted_f1:.4f}\n"
    f"  Top-1 accuracy               : {top1_acc:.4f}\n"
    f"  Multiclass AUC (OvR weighted): {auc_ovr:.4f}",
    "",
    "Per-class metrics (sorted by support desc):\n"
    + report_df.to_string(float_format="%.4f"),
    "",
    "Confusion matrix (row-normalised):\n" + cm_df.to_string(float_format="%.3f"),
    "",
    auc_block,
    threeway,
    "",
    cls_delta_block,
    "",
    decision_block,
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
    "metric":  k,
    "v5":      V5_METRICS[k],
    "v6":      V6_METRICS[k],
    "v6b":     v6b_metrics[k],
    "v6b_vs_v5": v6b_metrics[k] - V5_METRICS[k],
    "v6b_vs_v6": v6b_metrics[k] - V6_METRICS[k],
} for k in ["balanced_accuracy", "macro_f1", "auc_ovr_weighted", "top1_accuracy"]
]).to_csv(os.path.join(OUT_DIR, "threeway_comparison.csv"), index=False)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=classes,
            yticklabels=classes, cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix (row-normalised)\nLogReg — BiomedCLIP v6b (24 ensembled)")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)

metrics_plot = report_df[["precision", "recall", "f1-score"]].copy()
x, w = np.arange(len(metrics_plot)), 0.25
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w, metrics_plot["precision"], w, label="Precision")
ax.bar(x,     metrics_plot["recall"],    w, label="Recall")
ax.bar(x + w, metrics_plot["f1-score"],  w, label="F1")
ax.set_xticks(x)
ax.set_xticklabels(metrics_plot.index, rotation=30, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Per-class Metrics — LogReg (BiomedCLIP v6b, 24 ensembled, balanced)")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "per_class_metrics.png"), dpi=150)
plt.close(fig)

print("Done.")
