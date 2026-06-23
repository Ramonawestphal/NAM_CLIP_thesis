"""
Logistic regression baseline on BiomedCLIP v6c concept score features.

v6c = 24 features, one per concept, each being the v5 template with the
highest training-partition intended-class AUC (empirical best-template select).

Identical methodology to all previous LR baselines:
  - StandardScaler fit on train partition only
  - class_weight='balanced', multinomial lbfgs, max_iter=2000, C=1.0

Outputs -> reports/baselines/logreg_biomedclip_v6c/

Run from project root after select_best_templates_v6c.py:
    python scripts/baseline_logreg_v6c.py
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
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6c.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
OUT_DIR       = "reports/baselines/logreg_biomedclip_v6c"

# ── Four-way reference numbers ────────────────────────────────────────────────
REF = {
    "v5":  {"balanced_accuracy": 0.608, "macro_f1": 0.485,
            "auc_ovr_weighted":  0.891, "top1_accuracy": 0.636},
    "v6":  {"balanced_accuracy": 0.555, "macro_f1": 0.448,
            "auc_ovr_weighted":  0.860, "top1_accuracy": 0.595},
    "v6b": {"balanced_accuracy": 0.550, "macro_f1": 0.433,
            "auc_ovr_weighted":  0.857, "top1_accuracy": 0.590},
}

# v5 per-class logreg AUC (OvR) — hardcoded from reports/baselines/logreg_biomedclip_v5/
V5_PER_CLASS_AUC = {
    "nv":    0.9059,
    "mel":   0.8020,
    "bkl":   0.8367,
    "bcc":   0.9449,
    "akiec": 0.9049,
    "vasc":  0.9910,
    "df":    0.9449,
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


def _class_dist(y: np.ndarray) -> dict:
    vals, counts = np.unique(y, return_counts=True)
    return dict(zip(vals, counts))


print(f"\nTrain: {len(y_train)} images")
for cls, cnt in sorted(_class_dist(y_train).items()):
    print(f"  {cls}: {cnt}")
print(f"\nTest:  {len(y_test)} images")
for cls, cnt in sorted(_class_dist(y_test).items()):
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

per_class_auc_v6c: dict[str, float] = {}
for i, cls in enumerate(classes):
    per_class_auc_v6c[cls] = roc_auc_score(
        (y_test == cls).astype(int), y_proba[:, i]
    )

cm      = confusion_matrix(y_test, y_pred, labels=classes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_df   = pd.DataFrame(cm_norm, index=classes, columns=classes)

v6c_metrics = {
    "balanced_accuracy": bal_acc,
    "macro_f1":          macro_f1,
    "auc_ovr_weighted":  auc_ovr,
    "top1_accuracy":     top1_acc,
}

# ── Four-way comparison table ─────────────────────────────────────────────────
metric_labels = {
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1":          "Macro F1",
    "auc_ovr_weighted":  "Multiclass AUC (OvR)",
    "top1_accuracy":     "Top-1 accuracy",
}
col_w = 22
hdr = (f"  {'Metric':<32s}  {'v5 (72)':>{col_w}}  {'v6 (24 rewrite)':>{col_w}}"
       f"  {'v6b (24 ensemble)':>{col_w}}  {'v6c (24 best-select)':>{col_w}}")
rows_4way = [hdr]
for key, lbl in metric_labels.items():
    rows_4way.append(
        f"  {lbl:<32s}  {REF['v5'][key]:>{col_w}.3f}"
        f"  {REF['v6'][key]:>{col_w}.3f}"
        f"  {REF['v6b'][key]:>{col_w}.3f}"
        f"  {v6c_metrics[key]:>{col_w}.3f}"
    )
fourway_block = "Four-way comparison:\n" + "\n".join(rows_4way)

# ── Per-class AUC delta (v6c vs v5) ──────────────────────────────────────────
# Sort by v5 AUC descending (highest-discriminating classes first)
cls_order = sorted(V5_PER_CLASS_AUC, key=lambda c: -V5_PER_CLASS_AUC[c])
cls_delta_lines = ["Per-class AUC delta (v6c vs v5, OvR):"]
for cls in cls_order:
    v5_auc  = V5_PER_CLASS_AUC.get(cls, float("nan"))
    v6c_auc = per_class_auc_v6c.get(cls, float("nan"))
    delta   = v6c_auc - v5_auc
    cls_delta_lines.append(
        f"  {cls:<6s}: v6c={v6c_auc:.4f}  v5={v5_auc:.4f}  (Delta {delta:+.4f})"
    )
cls_delta_block = "\n".join(cls_delta_lines)

# ── Decision block ────────────────────────────────────────────────────────────
d_v6  = bal_acc - REF["v6"]["balanced_accuracy"]
d_v6b = bal_acc - REF["v6b"]["balanced_accuracy"]
d_v5  = bal_acc - REF["v5"]["balanced_accuracy"]

if d_v6 >= 0.015:
    verdict = "*** v6c is the strongest 24-feature configuration (beats v6 by >= 0.015) ***"
elif d_v6 > 0:
    verdict = ("v6c beats v6 but by < 0.015 — marginal improvement; "
               "consider whether 72-feature v5 overhead is acceptable")
else:
    verdict = ("*** v6c does NOT beat v6 — empirical template selection "
               "did not recover signal lost by hand-rewriting ***")

decision_block = "\n".join([
    "Decision (balanced accuracy):",
    f"  v6c vs v6  delta: {d_v6:+.3f}",
    f"  v6c vs v6b delta: {d_v6b:+.3f}",
    f"  v6c vs v5  delta: {d_v5:+.3f}",
    f"  {verdict}",
])

# ── Full report ───────────────────────────────────────────────────────────────
auc_block = "Per-class AUC (OvR, v6c model):\n"
for cls in report_df.index:
    auc_block += f"  {cls:6s}: {per_class_auc_v6c[cls]:.4f}\n"

HEADER = ("==== Logistic Regression Baseline "
          "(BiomedCLIP v6c features, 24 best-template-selected, balanced) ====")

full_report = "\n".join([
    HEADER, "",
    (f"Aggregate metrics:\n"
     f"  Balanced accuracy            : {bal_acc:.4f}\n"
     f"  Macro F1                     : {macro_f1:.4f}\n"
     f"  Weighted F1                  : {weighted_f1:.4f}\n"
     f"  Top-1 accuracy               : {top1_acc:.4f}\n"
     f"  Multiclass AUC (OvR weighted): {auc_ovr:.4f}"),
    "",
    "Per-class metrics (sorted by support desc):\n"
    + report_df.to_string(float_format="%.4f"),
    "",
    "Confusion matrix (row-normalised):\n" + cm_df.to_string(float_format="%.3f"),
    "",
    auc_block,
    fourway_block,
    "",
    cls_delta_block,
    "",
    decision_block,
    f"\nOutputs -> {OUT_DIR}/",
])

print("\n" + full_report)

# ── Save artifacts ────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "metrics_summary.txt"), "w", encoding="utf-8") as f:
    f.write(full_report + "\n")

pd.DataFrame([{
    "balanced_accuracy": bal_acc, "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,  "top1_accuracy": top1_acc,
    "auc_ovr_weighted": auc_ovr,
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
    "metric":        k,
    "v5":            REF["v5"][k],
    "v6":            REF["v6"][k],
    "v6b":           REF["v6b"][k],
    "v6c":           v6c_metrics[k],
    "v6c_vs_v5":     v6c_metrics[k] - REF["v5"][k],
    "v6c_vs_v6":     v6c_metrics[k] - REF["v6"][k],
    "v6c_vs_v6b":    v6c_metrics[k] - REF["v6b"][k],
} for k in metric_labels]).to_csv(
    os.path.join(OUT_DIR, "fourway_comparison.csv"), index=False
)

pd.DataFrame([{
    "class": cls,
    "v6c_auc": per_class_auc_v6c.get(cls, float("nan")),
    "v5_auc":  V5_PER_CLASS_AUC.get(cls, float("nan")),
    "delta":   per_class_auc_v6c.get(cls, float("nan"))
               - V5_PER_CLASS_AUC.get(cls, float("nan")),
} for cls in cls_order]).to_csv(
    os.path.join(OUT_DIR, "per_class_auc_delta.csv"), index=False
)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=classes,
            yticklabels=classes, cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix (row-normalised)\nLogReg — BiomedCLIP v6c (24 best-template)")
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
ax.set_title("Per-class Metrics — LogReg (BiomedCLIP v6c, 24 best-template, balanced)")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "per_class_metrics.png"), dpi=150)
plt.close(fig)

print("Done.")
