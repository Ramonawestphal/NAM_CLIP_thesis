"""
Logistic regression baseline on BiomedCLIP v5 concept score features.
Linear-head reference point for NAM comparison on identical inputs.
"""

import os
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

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v5.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
OUT_DIR       = "reports/baselines/logreg_biomedclip_v5"

# ── Load features ─────────────────────────────────────────────────────────────
print("Loading features...")
feat = np.load(FEATURES_PATH, allow_pickle=True)
scores    = feat["scores"]        # (10015, 72)
labels    = feat["labels"]        # (10015,) string
image_ids = feat["image_ids"]     # (10015,) string

assert scores.shape == (10015, 72), f"Unexpected scores shape: {scores.shape}"
assert len(labels) == len(image_ids) == scores.shape[0], "Row count mismatch"

# ── Load splits ───────────────────────────────────────────────────────────────
print("Loading splits...")
split = np.load(SPLITS_PATH)
train_idx = split["train_idx"]
test_idx  = split["test_idx"]

# Verify disjoint and complete
assert len(np.intersect1d(train_idx, test_idx)) == 0, "Train/test indices overlap"
assert len(np.union1d(train_idx, test_idx)) == scores.shape[0], \
    f"Indices don't cover all {scores.shape[0]} rows"

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test = scores[train_idx], scores[test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]
ids_test        = image_ids[test_idx]

def class_dist(y):
    vals, counts = np.unique(y, return_counts=True)
    return {v: c for v, c in zip(vals, counts)}

print(f"\nTrain: {len(y_train)} images")
for cls, cnt in sorted(class_dist(y_train).items()):
    print(f"  {cls}: {cnt}")

print(f"\nTest:  {len(y_test)} images")
for cls, cnt in sorted(class_dist(y_test).items()):
    print(f"  {cls}: {cnt}")

# ── Standardize (fit on train only) ───────────────────────────────────────────
print("\nStandardizing features (z-score, fit on train)...")
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
y_proba = clf.predict_proba(X_test)   # (N, 7)
classes = clf.classes_                # sorted class names

# ── Aggregate metrics ─────────────────────────────────────────────────────────
bal_acc   = balanced_accuracy_score(y_test, y_pred)
macro_f1  = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")
top1_acc  = accuracy_score(y_test, y_pred)
auc_ovr   = roc_auc_score(y_test, y_proba, multi_class="ovr",
                           average="weighted", labels=classes)

# ── Per-class metrics ─────────────────────────────────────────────────────────
report_dict = classification_report(y_test, y_pred, labels=classes,
                                    output_dict=True, zero_division=0)
report_df = (
    pd.DataFrame(report_dict)
    .T
    .loc[list(classes)]          # only the 7 class rows
    .rename(columns={"support": "support"})
    .astype({"support": int})
    .sort_values("support", ascending=False)
)

# Per-class AUC (one-vs-rest)
per_class_auc = {}
for i, cls in enumerate(classes):
    y_bin = (y_test == cls).astype(int)
    per_class_auc[cls] = roc_auc_score(y_bin, y_proba[:, i])

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=classes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# ── Format report text ────────────────────────────────────────────────────────
HEADER = "==== Logistic Regression Baseline (BiomedCLIP v5 features, balanced) ===="

agg_block = (
    f"Aggregate metrics:\n"
    f"  Balanced accuracy            : {bal_acc:.4f}\n"
    f"  Macro F1                     : {macro_f1:.4f}\n"
    f"  Weighted F1                  : {weighted_f1:.4f}\n"
    f"  Top-1 accuracy               : {top1_acc:.4f}\n"
    f"  Multiclass AUC (OvR weighted): {auc_ovr:.4f}\n"
)

per_class_block = "Per-class metrics (sorted by support desc):\n"
per_class_block += report_df.to_string(float_format="%.4f") + "\n"

auc_block = "Per-class AUC (OvR):\n"
for cls in report_df.index:
    auc_block += f"  {cls:6s}: {per_class_auc[cls]:.4f}\n"

cm_df = pd.DataFrame(cm_norm, index=classes, columns=classes)
cm_block = "Confusion matrix (row-normalized, rows=true, cols=pred):\n"
cm_block += cm_df.to_string(float_format="%.3f") + "\n"

full_report = "\n".join([HEADER, "", agg_block, per_class_block, auc_block, cm_block])
full_report += f"\nOutputs → {OUT_DIR}/"

print("\n" + full_report)

# ── Save artifacts ────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

# metrics_summary.txt
with open(os.path.join(OUT_DIR, "metrics_summary.txt"), "w", encoding="utf-8") as f:
    f.write(full_report + "\n")

# metrics_summary.csv  (one-row aggregate)
agg_csv = pd.DataFrame([{
    "balanced_accuracy": bal_acc,
    "macro_f1":          macro_f1,
    "weighted_f1":       weighted_f1,
    "top1_accuracy":     top1_acc,
    "auc_ovr_weighted":  auc_ovr,
}])
agg_csv.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

# classification_report.csv
report_df.to_csv(os.path.join(OUT_DIR, "classification_report.csv"))

# confusion_matrix.csv  (raw counts for archival; normalized is in the plot)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(
    os.path.join(OUT_DIR, "confusion_matrix.csv")
)

# predictions.csv
prob_cols = {f"prob_{c}": y_proba[:, i] for i, c in enumerate(classes)}
pred_df = pd.DataFrame({
    "image_id":       ids_test,
    "true_label":     y_test,
    "predicted_label": y_pred,
    **prob_cols,
})
pred_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

# ── Plot: confusion matrix (row-normalized) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    xticklabels=classes,
    yticklabels=classes,
    cmap="Blues",
    vmin=0,
    vmax=1,
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (row-normalized)\nLogReg baseline — BiomedCLIP v5")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)

# ── Plot: per-class precision / recall / F1 ───────────────────────────────────
metrics_plot = report_df[["precision", "recall", "f1-score"]].copy()
x = np.arange(len(metrics_plot))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, metrics_plot["precision"], width, label="Precision")
ax.bar(x,         metrics_plot["recall"],    width, label="Recall")
ax.bar(x + width, metrics_plot["f1-score"],  width, label="F1")
ax.set_xticks(x)
ax.set_xticklabels(metrics_plot.index, rotation=30, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Per-class Metrics — LogReg baseline (BiomedCLIP v5, balanced)")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "per_class_metrics.png"), dpi=150)
plt.close(fig)

print("Done.")
