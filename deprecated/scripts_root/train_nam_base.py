"""
Base NAM training on BiomedCLIP v5 concept score features (HAM10000).

Phase 1: trains a plain NAMMulticlass with no regularization extensions.
Goal: stable training, match/beat logreg baseline (bal_acc=0.608, AUC=0.891),
produce shape functions for inspection.

Runs 5 seeds with early stopping on val balanced accuracy, then evaluates each
best checkpoint on the held-out test set and aggregates metrics across seeds.

Run from project root:
    python scripts/train_nam_base.py
"""

from __future__ import annotations

import os
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.models.nam_multiclass import NAMMulticlass

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v5.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
OUT_DIR       = "reports/nam/base"

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEEDS         = [42, 43, 44, 45, 46]
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
BATCH_SIZE    = 256
MAX_EPOCHS    = 100
PATIENCE      = 10       # early stopping on val balanced accuracy
HIDDEN_DIMS   = (64, 64, 32)
DROPOUT       = 0.1
N_FEATURES    = 72
N_CLASSES     = 7

# Logistic regression baseline reference (from reports/baselines/logreg_biomedclip_v5/)
LR_BASELINE = {
    "balanced_accuracy": 0.608,
    "macro_f1":          0.485,
    "auc_ovr_weighted":  0.891,
}

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("WARNING: CUDA not available — training on CPU. "
          "Expect ~10–20 min per seed with 100 epochs on this dataset.")
else:
    print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading features...")
feat      = np.load(FEATURES_PATH, allow_pickle=True)
scores    = feat["scores"]     # (10015, 72) float32
labels    = feat["labels"]     # (10015,) str
lesion_ids = feat["lesion_ids"] # (10015,) str
image_ids = feat["image_ids"]  # (10015,) str

assert scores.shape == (10015, N_FEATURES), f"Unexpected scores shape: {scores.shape}"

print("Loading splits...")
split     = np.load(SPLITS_PATH)
train_idx = split["train_idx"]   # 8020 indices
test_idx  = split["test_idx"]    # 1995 indices

assert len(np.intersect1d(train_idx, test_idx)) == 0, "Train/test indices overlap"
assert len(np.union1d(train_idx, test_idx)) == scores.shape[0]

X_all_train     = scores[train_idx]
y_all_train     = labels[train_idx]
lesion_ids_train = lesion_ids[train_idx]
X_test          = scores[test_idx]
y_test          = labels[test_idx]

# ── Label encoding ─────────────────────────────────────────────────────────────
class_names = sorted(np.unique(labels).tolist())   # ['akiec','bcc','bkl','df','mel','nv','vasc']
assert len(class_names) == N_CLASSES, f"Expected {N_CLASSES} classes, got {len(class_names)}"
class_to_idx = {c: i for i, c in enumerate(class_names)}

y_all_train_enc = np.array([class_to_idx[c] for c in y_all_train], dtype=np.int64)
y_test_enc      = np.array([class_to_idx[c] for c in y_test],      dtype=np.int64)

# ── Val split (GroupShuffleSplit 80/20, grouped by lesion_id) ─────────────────
print("Splitting train → train_final + val (GroupShuffleSplit 80/20 by lesion)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_final_rel, val_rel = next(
    gss.split(X_all_train, y_all_train, groups=lesion_ids_train)
)

X_train_raw = X_all_train[train_final_rel]
y_train_enc = y_all_train_enc[train_final_rel]
y_train_str = y_all_train[train_final_rel]
X_val_raw   = X_all_train[val_rel]
y_val_enc   = y_all_train_enc[val_rel]

train_lesions = set(lesion_ids_train[train_final_rel])
val_lesions   = set(lesion_ids_train[val_rel])
assert len(train_lesions & val_lesions) == 0, "Lesion leakage between train_final and val"

print(f"  train_final : {len(y_train_enc):5d} images")
print(f"  val         : {len(y_val_enc):5d} images")
print(f"  test        : {len(y_test_enc):5d} images")

# ── Standardise (fit on train_final only) ────────────────────────────────────
print("Standardising features (z-score, fit on train_final)...")
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val_sc    = scaler.transform(X_val_raw).astype(np.float32)
X_test_sc   = scaler.transform(X_test).astype(np.float32)

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print(f"  Scaler saved → {OUT_DIR}/scaler.pkl")

# ── Class weights (computed on train_final labels) ────────────────────────────
weights = compute_class_weight(
    "balanced",
    classes=np.array(class_names),
    y=y_train_str,
)
weight_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
print(f"  Class weights (balanced): {dict(zip(class_names, weights.round(3)))}")

# ── Tensors ───────────────────────────────────────────────────────────────────
X_val_t  = torch.tensor(X_val_sc,  dtype=torch.float32, device=DEVICE)
y_val_t  = torch.tensor(y_val_enc, dtype=torch.long,    device=DEVICE)
X_test_t = torch.tensor(X_test_sc, dtype=torch.float32, device=DEVICE)
y_test_t = torch.tensor(y_test_enc, dtype=torch.long,   device=DEVICE)

train_dataset = TensorDataset(
    torch.tensor(X_train_sc,  dtype=torch.float32),
    torch.tensor(y_train_enc, dtype=torch.long),
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed helpers
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_test(model: NAMMulticlass) -> dict:
    """Compute full metric suite on the held-out test set."""
    model.eval()
    with torch.no_grad():
        logits  = model(X_test_t)
        proba   = torch.softmax(logits, dim=1).cpu().numpy()
    preds_enc = logits.argmax(dim=1).cpu().numpy()
    y_pred_str = [class_names[i] for i in preds_enc]

    bal_acc    = balanced_accuracy_score(y_test, y_pred_str)
    macro_f1   = f1_score(y_test, y_pred_str, average="macro",     zero_division=0)
    w_f1       = f1_score(y_test, y_pred_str, average="weighted",  zero_division=0)
    top1_acc   = accuracy_score(y_test, y_pred_str)
    auc_ovr    = roc_auc_score(y_test, proba, multi_class="ovr",
                               average="weighted", labels=class_names)

    per_cls_auc = {}
    for i, cls in enumerate(class_names):
        y_bin = (y_test == cls).astype(int)
        per_cls_auc[cls] = roc_auc_score(y_bin, proba[:, i])

    report_dict = classification_report(
        y_test, y_pred_str, labels=class_names, output_dict=True, zero_division=0
    )
    report_df = (
        pd.DataFrame(report_dict).T.loc[class_names]
        .astype({"support": int})
        .sort_values("support", ascending=False)
    )
    report_df["auc"] = [per_cls_auc[c] for c in report_df.index]

    cm = confusion_matrix(y_test, y_pred_str, labels=class_names)

    return {
        "balanced_accuracy": bal_acc,
        "macro_f1":          macro_f1,
        "weighted_f1":       w_f1,
        "top1_accuracy":     top1_acc,
        "auc_ovr_weighted":  auc_ovr,
        "report_df":         report_df,
        "confusion_matrix":  cm,
        "proba":             proba,
        "y_pred_str":        y_pred_str,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (5 seeds)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Training NAM base — {len(SEEDS)} seeds")
print(f"  lr={LR}, weight_decay={WEIGHT_DECAY}, batch={BATCH_SIZE}")
print(f"  max_epochs={MAX_EPOCHS}, patience={PATIENCE}")
print(f"  subnet: {list(HIDDEN_DIMS)} + ReLU + dropout {DROPOUT}")
print(f"  class_weight='balanced'")
print(f"{'='*60}\n")

all_results = []

for seed in SEEDS:
    print(f"── Seed {seed} ──────────────────────────────────────────────")
    torch.manual_seed(seed)
    np.random.seed(seed)

    seed_dir = os.path.join(OUT_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    model = NAMMulticlass(
        n_features=N_FEATURES,
        num_classes=N_CLASSES,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    best_val_balacc = -1.0
    best_epoch      = -1
    patience_ctr    = 0
    training_log    = []
    improved_in_20  = False

    for epoch in range(MAX_EPOCHS):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)
        train_loss = total_loss / len(y_train_enc)

        # ── Val ──
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_preds  = val_logits.argmax(dim=1).cpu().numpy()
        val_balacc = balanced_accuracy_score(y_val_enc, val_preds)

        training_log.append({
            "epoch":           epoch + 1,
            "train_loss":      train_loss,
            "val_loss":        val_loss,
            "val_balanced_acc": val_balacc,
        })

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_balacc={val_balacc:.4f}")

        # ── Early stopping ──
        if val_balacc > best_val_balacc + 1e-4:
            best_val_balacc = val_balacc
            best_epoch      = epoch + 1
            patience_ctr    = 0
            torch.save(model.state_dict(),
                       os.path.join(seed_dir, "best_model.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1} "
                      f"(best epoch {best_epoch}, val_balacc={best_val_balacc:.4f})")
                break

        # ── Convergence check at epoch 20 ──
        if epoch == 19:
            log_so_far = [r["val_balanced_acc"] for r in training_log]
            if max(log_so_far) > log_so_far[0] + 1e-4:
                improved_in_20 = True

    # Save training log
    log_df = pd.DataFrame(training_log)
    log_df.to_csv(os.path.join(seed_dir, "training_log.csv"), index=False)
    print(f"  Training log → {seed_dir}/training_log.csv")

    # ── Evaluate best checkpoint on test set ──
    ckpt_path = os.path.join(seed_dir, "best_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    metrics = evaluate_on_test(model)
    print(f"  Test  bal_acc={metrics['balanced_accuracy']:.4f}  "
          f"macro_f1={metrics['macro_f1']:.4f}  "
          f"AUC={metrics['auc_ovr_weighted']:.4f}")

    all_results.append({
        "seed":           seed,
        "improved_in_20": improved_in_20,
        "log_df":         log_df,
        **{k: v for k, v in metrics.items()
           if k not in ("report_df", "confusion_matrix", "proba", "y_pred_str")},
        "report_df":      metrics["report_df"],
        "confusion_matrix": metrics["confusion_matrix"],
    })

# ── Convergence guard ─────────────────────────────────────────────────────────
if not any(r["improved_in_20"] for r in all_results):
    raise RuntimeError(
        "No seed improved val balanced accuracy in the first 20 epochs across "
        "all seeds. Check for bugs or verify that the features are suitable "
        "for NAM training."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics across seeds
# ─────────────────────────────────────────────────────────────────────────────
AGG_KEYS = ["balanced_accuracy", "macro_f1", "weighted_f1", "top1_accuracy",
            "auc_ovr_weighted"]

agg_rows = []
for r in all_results:
    agg_rows.append({"seed": r["seed"], **{k: r[k] for k in AGG_KEYS}})

agg_df = pd.DataFrame(agg_rows)
mean_row = agg_df[AGG_KEYS].mean().to_dict()
mean_row["seed"] = "mean"
std_row  = agg_df[AGG_KEYS].std().to_dict()
std_row["seed"]  = "std"
agg_df = pd.concat(
    [agg_df, pd.DataFrame([mean_row, std_row])], ignore_index=True
)
agg_df.to_csv(os.path.join(OUT_DIR, "aggregated_metrics.csv"), index=False)

means = {k: float(mean_row[k]) for k in AGG_KEYS}
stds  = {k: float(std_row[k])  for k in AGG_KEYS}

# ── Per-class metrics (seed-mean) ─────────────────────────────────────────────
report_dfs = [r["report_df"] for r in all_results]
report_mean = (
    pd.concat(report_dfs)
    .groupby(level=0)
    .mean()
    .loc[[c for c in report_dfs[0].index]]  # preserve support-sorted order
)
report_mean["support"] = report_mean["support"].round(0).astype(int)
report_mean.to_csv(os.path.join(OUT_DIR, "per_class_metrics.csv"))

# ── Confusion matrix (seed-mean, row-normalised) ──────────────────────────────
cms = np.stack([r["confusion_matrix"] for r in all_results], axis=0)  # (5, 7, 7)
cm_mean = cms.mean(axis=0)
cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)
pd.DataFrame(cm_mean.round(1), index=class_names, columns=class_names).to_csv(
    os.path.join(OUT_DIR, "confusion_matrix.csv")
)

# ── Comparison to logreg ──────────────────────────────────────────────────────
comparison = pd.DataFrame([{
    "metric":    k,
    "nam_mean":  means[k],
    "nam_std":   stds[k],
    "logreg":    LR_BASELINE.get(k, float("nan")),
    "delta":     means[k] - LR_BASELINE.get(k, float("nan")),
} for k in ["balanced_accuracy", "macro_f1", "auc_ovr_weighted"]])
comparison.to_csv(os.path.join(OUT_DIR, "comparison_to_logreg.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Training curves plot (val balanced accuracy, all seeds overlaid)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
for r in all_results:
    log = r["log_df"]
    ax.plot(log["epoch"], log["val_balanced_acc"],
            alpha=0.8, label=f"seed {r['seed']}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Val balanced accuracy")
ax.set_title("NAM Base — Training curves (val balanced accuracy)")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────────────
def _delta(key: str) -> str:
    ref = LR_BASELINE.get(key)
    if ref is None or np.isnan(ref):
        return ""
    d = means[key] - ref
    return f"(Δ {'+' if d >= 0 else ''}{d:.3f})"


REPORT = f"""
==== NAM Base Training (BiomedCLIP v5 features) ====

Trained {len(SEEDS)} seeds with hyperparameters:
  lr={LR}, weight_decay={WEIGHT_DECAY}, batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}
  subnet: {list(HIDDEN_DIMS)} + ReLU + dropout {DROPOUT}
  class_weight='balanced'

Test-set results (mean ± std across seeds):
  Balanced accuracy            : {means['balanced_accuracy']:.4f} ± {stds['balanced_accuracy']:.4f}
  Macro F1                     : {means['macro_f1']:.4f} ± {stds['macro_f1']:.4f}
  Weighted F1                  : {means['weighted_f1']:.4f} ± {stds['weighted_f1']:.4f}
  Top-1 accuracy               : {means['top1_accuracy']:.4f} ± {stds['top1_accuracy']:.4f}
  Multiclass AUC (OvR weighted): {means['auc_ovr_weighted']:.4f} ± {stds['auc_ovr_weighted']:.4f}

Comparison to logistic regression baseline:
  Balanced accuracy : NAM {means['balanced_accuracy']:.3f} vs LR {LR_BASELINE['balanced_accuracy']:.3f}  {_delta('balanced_accuracy')}
  Macro F1          : NAM {means['macro_f1']:.3f} vs LR {LR_BASELINE['macro_f1']:.3f}  {_delta('macro_f1')}
  Multiclass AUC    : NAM {means['auc_ovr_weighted']:.3f} vs LR {LR_BASELINE['auc_ovr_weighted']:.3f}  {_delta('auc_ovr_weighted')}

Per-class metrics (NAM, mean across seeds):
{report_mean[['precision', 'recall', 'f1-score', 'auc', 'support']].to_string(float_format='%.4f')}

Confusion matrix (row-normalised, seed-mean):
{pd.DataFrame(cm_norm.round(3), index=class_names, columns=class_names).to_string()}

Outputs → {OUT_DIR}/
"""
print(REPORT)

with open(os.path.join(OUT_DIR, "metrics_summary.txt"), "w", encoding="utf-8") as f:
    f.write(REPORT.lstrip() + "\n")

print("Done.")
