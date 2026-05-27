"""
NAM v6 final training — 5 seeds with config 9 (multi-seed sweep winner).

Config 9 selected by mean test balanced accuracy across 3 seeds in
sweep_nam_v6_multiseed.py. Hyperparameters are read from
reports/nam/v6_sweep/sweep_results.csv by config_id.

Three-way comparison in the final report:
  - v6 LR baseline    : bal_acc=0.555
  - v6 NAM Phase-1    : bal_acc=0.498  (train_nam_v6.py, [32,16] dropout=0.25 wd=1e-4)
  - v5 NAM            : bal_acc=0.540  (train_nam_base.py, [64,64,32] dropout=0.10 wd=1e-5)

Output -> reports/nam/v6_final/

Run from project root:
    python scripts/train_nam_v6_final.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows consoles often default to cp1252; avoid UnicodeEncodeError on status lines.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import ast
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
from src.models.concurvity import multiclass_concurvity

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
SWEEP_CSV     = "reports/nam/v6_sweep/sweep_results.csv"
OUT_DIR       = "reports/nam/v6_final"

# ── Selected config ID (multi-seed sweep winner) ──────────────────────────────
SELECTED_CONFIG_ID = 9

# ── Reference baselines ───────────────────────────────────────────────────────
V6_LR_BASELINE  = {"balanced_accuracy": 0.555, "macro_f1": 0.448, "auc_ovr_weighted": 0.860}
V6_NAM_PHASE1   = {"balanced_accuracy": 0.498}
V5_NAM_BASELINE = {"balanced_accuracy": 0.540}

# ── Fixed hyperparameters (shared across all seeds) ───────────────────────────
SEEDS        = [42, 43, 44, 45, 46]
LR           = 1e-3
BATCH_SIZE   = 256
MAX_EPOCHS   = 100
PATIENCE     = 15
SCHED_PATIENCE = 5
SCHED_FACTOR   = 0.5
N_FEATURES   = 24
N_CLASSES    = 7

CONVERGENCE_THRESHOLD = 0.50
CONVERGENCE_EPOCH     = 30

# ── Concurvity regularization (Siems et al. 2023) ────────────────────────────
# L_total = L_task + CONCURVITY_LAMBDA * R_perp  (Siems et al. 2023, Eq. 2,
#   multiclass avg: R_perp = mean_k mean_{i<j} |Corr(f_{i,k}, f_{j,k})|)
# 0.0 → plain NAM; loss and gradients are bit-identical to the unregularized run.
# R_perp is always logged as a diagnostic regardless of this value.
# Suggested ablation values: {0.0, 0.001, 0.01, 0.1, 1.0}
CONCURVITY_LAMBDA = 0.0

# ── CLI overrides (applied after all defaults are set) ───────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--concurvity_lambda", type=float, default=None)
_parser.add_argument("--out_dir",           type=str,   default=None)
_parser.add_argument("--max_epochs",        type=int,   default=None)
_parser.add_argument("--seed",              type=int,   default=None)
_cli, _ = _parser.parse_known_args()
if _cli.concurvity_lambda is not None:
    CONCURVITY_LAMBDA = _cli.concurvity_lambda
if _cli.out_dir is not None:
    OUT_DIR = _cli.out_dir
if _cli.max_epochs is not None:
    MAX_EPOCHS = _cli.max_epochs
if _cli.seed is not None:
    SEEDS = [_cli.seed]

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("WARNING: CUDA not available — training on CPU. Expect ~15-25 min per seed.")
else:
    print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Read config 9 hyperparameters from sweep results
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(SWEEP_CSV):
    raise FileNotFoundError(
        f"Sweep results not found at {SWEEP_CSV}. "
        "Run sweep_nam_v6.py first."
    )

sweep_df = pd.read_csv(SWEEP_CSV)
cfg_rows = sweep_df[sweep_df["config_id"] == SELECTED_CONFIG_ID]
if cfg_rows.empty:
    raise ValueError(
        f"Config {SELECTED_CONFIG_ID} not found in {SWEEP_CSV}. "
        f"Available config IDs: {sorted(sweep_df['config_id'].tolist())}"
    )
selected = cfg_rows.iloc[0]

# Parse hidden dims from string representation e.g. "[64, 32]"
HIDDEN_DIMS  = tuple(ast.literal_eval(selected["hidden"]))
DROPOUT      = float(selected["dropout"])
WEIGHT_DECAY = float(selected["weight_decay"])

print(f"\n{'='*65}")
print(f"Selected config {SELECTED_CONFIG_ID} (multi-seed sweep winner)")
print(f"  hidden_dims  : {list(HIDDEN_DIMS)}")
print(f"  dropout      : {DROPOUT}")
print(f"  weight_decay : {WEIGHT_DECAY:.0e}")
print(f"  single-seed val balacc  : {selected['best_val_balacc']:.4f}")
print(f"  single-seed test balacc : {selected['test_balacc']:.4f}")
print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Load + split + standardise
# ─────────────────────────────────────────────────────────────────────────────
print("Loading features...")
feat       = np.load(FEATURES_PATH, allow_pickle=True)
scores     = feat["scores"]
labels     = feat["labels"]
lesion_ids = feat["lesion_ids"]
image_ids  = feat["image_ids"]
assert scores.shape == (10015, N_FEATURES), f"Unexpected shape: {scores.shape}"

print("Loading splits...")
split     = np.load(SPLITS_PATH)
train_idx = split["train_idx"]
test_idx  = split["test_idx"]
assert len(np.intersect1d(train_idx, test_idx)) == 0
assert len(np.union1d(train_idx, test_idx)) == scores.shape[0]

X_all_train      = scores[train_idx]
y_all_train      = labels[train_idx]
lesion_ids_train = lesion_ids[train_idx]
X_test           = scores[test_idx]
y_test           = labels[test_idx]

class_names  = sorted(np.unique(labels).tolist())
assert len(class_names) == N_CLASSES
class_to_idx = {c: i for i, c in enumerate(class_names)}
y_all_train_enc = np.array([class_to_idx[c] for c in y_all_train], dtype=np.int64)
y_test_enc      = np.array([class_to_idx[c] for c in y_test],      dtype=np.int64)

# Val split — same random_state=42 as v6_base for direct comparability
print("Carving validation set (GroupShuffleSplit 80/20 by lesion_id)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_final_rel, val_rel = next(
    gss.split(X_all_train, y_all_train, groups=lesion_ids_train)
)
X_train_raw = X_all_train[train_final_rel]
y_train_enc = y_all_train_enc[train_final_rel]
y_train_str = y_all_train[train_final_rel]
X_val_raw   = X_all_train[val_rel]
y_val_enc   = y_all_train_enc[val_rel]

assert len(
    set(lesion_ids_train[train_final_rel]) & set(lesion_ids_train[val_rel])
) == 0, "Lesion leakage between train_final and val"
print(f"  train_final : {len(y_train_enc):5d} images")
print(f"  val         : {len(y_val_enc):5d} images")
print(f"  test        : {len(y_test_enc):5d} images")

print("Standardising (z-score, fit on train_final)...")
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val_sc   = scaler.transform(X_val_raw).astype(np.float32)
X_test_sc  = scaler.transform(X_test).astype(np.float32)

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

weights = compute_class_weight("balanced", classes=np.array(class_names), y=y_train_str)
weight_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
print(f"  Class weights (balanced): {dict(zip(class_names, weights.round(3)))}")

X_val_t  = torch.tensor(X_val_sc,  dtype=torch.float32, device=DEVICE)
y_val_t  = torch.tensor(y_val_enc, dtype=torch.long,    device=DEVICE)
X_test_t = torch.tensor(X_test_sc, dtype=torch.float32, device=DEVICE)

train_dataset = TensorDataset(
    torch.tensor(X_train_sc,  dtype=torch.float32),
    torch.tensor(y_train_enc, dtype=torch.long),
)


# ─────────────────────────────────────────────────────────────────────────────
# Test-set evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_test(model: NAMMulticlass) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
    preds_enc  = logits.argmax(dim=1).cpu().numpy()
    y_pred_str = [class_names[i] for i in preds_enc]

    bal_acc  = balanced_accuracy_score(y_test, y_pred_str)
    macro_f1 = f1_score(y_test, y_pred_str, average="macro",    zero_division=0)
    w_f1     = f1_score(y_test, y_pred_str, average="weighted", zero_division=0)
    top1_acc = accuracy_score(y_test, y_pred_str)
    auc_ovr  = roc_auc_score(y_test, proba, multi_class="ovr",
                              average="weighted", labels=class_names)

    per_cls_auc = {
        cls: roc_auc_score((y_test == cls).astype(int), proba[:, i])
        for i, cls in enumerate(class_names)
    }
    report_dict = classification_report(
        y_test, y_pred_str, labels=class_names, output_dict=True, zero_division=0
    )
    report_df = (
        pd.DataFrame(report_dict).T.loc[class_names]
        .astype({"support": int})
        .sort_values("support", ascending=False)
    )
    report_df["auc"] = [per_cls_auc[c] for c in report_df.index]

    return {
        "balanced_accuracy": bal_acc,
        "macro_f1":          macro_f1,
        "weighted_f1":       w_f1,
        "top1_accuracy":     top1_acc,
        "auc_ovr_weighted":  auc_ovr,
        "report_df":         report_df,
        "confusion_matrix":  confusion_matrix(y_test, y_pred_str, labels=class_names),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (5 seeds)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"NAM v6 final training — {len(SEEDS)} seeds")
print(f"  Winning config: hidden={list(HIDDEN_DIMS)}, dropout={DROPOUT}, wd={WEIGHT_DECAY:.0e}")
print(f"  lr={LR} (ReduceLROnPlateau patience={SCHED_PATIENCE} factor={SCHED_FACTOR})")
print(f"  batch={BATCH_SIZE}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}")
print(f"  class_weight='balanced'")
print(f"{'='*65}\n")

all_results = []

for seed in SEEDS:
    print(f"-- Seed {seed} " + "-" * 45)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE, min_lr=1e-6,
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    loader    = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    best_val_balacc = -1.0
    best_epoch      = -1
    patience_ctr    = 0
    training_log    = []
    reached_threshold_in_30 = False

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss  = 0.0
        total_rperp = 0.0
        n_batches   = 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            if CONCURVITY_LAMBDA > 0:
                # L_total = L_task + lambda * R_perp  (Siems et al. 2023)
                logits, shape_outs = model(X_b, return_shape_outputs=True)
                task_loss = criterion(logits, y_b)
                r_perp_b  = multiclass_concurvity(shape_outs)
                loss = task_loss + CONCURVITY_LAMBDA * r_perp_b
            else:
                # Plain NAM: identical loss and gradients to unregularized run.
                # R_perp is computed under no_grad + eval (no PRNG side-effects).
                loss = criterion(model(X_b), y_b)
                with torch.no_grad():
                    model.eval()
                    r_perp_b = multiclass_concurvity(model.shape_outputs(X_b))
                    model.train()
            loss.backward()
            optimizer.step()
            total_loss  += loss.item() * len(y_b)
            total_rperp += r_perp_b.item()
            n_batches   += 1
        train_loss  = total_loss / len(y_train_enc)
        train_rperp = total_rperp / n_batches

        model.eval()
        with torch.no_grad():
            val_logits, val_shape_outs = model(X_val_t, return_shape_outputs=True)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_preds  = val_logits.argmax(dim=1).cpu().numpy()
            val_rperp  = multiclass_concurvity(val_shape_outs).item()
        val_balacc  = balanced_accuracy_score(y_val_enc, val_preds)
        current_lr  = optimizer.param_groups[0]["lr"]

        training_log.append({
            "epoch":            epoch + 1,
            "train_loss":       train_loss,
            "val_loss":         val_loss,
            "val_balanced_acc": val_balacc,
            "lr":               current_lr,
            "r_perp_train":     train_rperp,
            "r_perp_val":       val_rperp,
        })

        scheduler.step(val_balacc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_balacc={val_balacc:.4f} "
                  f"lr={current_lr:.2e}")

        if epoch < CONVERGENCE_EPOCH and val_balacc >= CONVERGENCE_THRESHOLD:
            reached_threshold_in_30 = True

        if val_balacc > best_val_balacc + 1e-4:
            best_val_balacc = val_balacc
            best_epoch      = epoch + 1
            patience_ctr    = 0
            torch.save(model.state_dict(), os.path.join(seed_dir, "best_model.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1} "
                      f"(best epoch {best_epoch}, val_balacc={best_val_balacc:.4f})")
                break

    log_df = pd.DataFrame(training_log)
    log_df.to_csv(os.path.join(seed_dir, "training_log.csv"), index=False)
    _best_idx           = log_df["val_balanced_acc"].idxmax()
    r_perp_val_at_best  = float(log_df.loc[_best_idx, "r_perp_val"])
    r_perp_train_at_best = float(log_df.loc[_best_idx, "r_perp_train"])

    model.load_state_dict(
        torch.load(os.path.join(seed_dir, "best_model.pt"),
                   map_location=DEVICE, weights_only=True)
    )
    metrics = evaluate_on_test(model)
    print(f"  Test  bal_acc={metrics['balanced_accuracy']:.4f}  "
          f"macro_f1={metrics['macro_f1']:.4f}  "
          f"AUC={metrics['auc_ovr_weighted']:.4f}  "
          f"R_perp_val@best={r_perp_val_at_best:.4f}")

    all_results.append({
        "seed":                    seed,
        "reached_threshold_in_30": reached_threshold_in_30,
        "best_epoch":              best_epoch,
        "best_val_balacc":         best_val_balacc,
        "r_perp_val_at_best":      r_perp_val_at_best,
        "r_perp_train_at_best":    r_perp_train_at_best,
        "log_df":                  log_df,
        **{k: v for k, v in metrics.items()
           if k not in ("report_df", "confusion_matrix")},
        "report_df":        metrics["report_df"],
        "confusion_matrix": metrics["confusion_matrix"],
    })

# ── Convergence guard ─────────────────────────────────────────────────────────
if not any(r["reached_threshold_in_30"] for r in all_results):
    raise RuntimeError(
        f"No seed reached val balanced accuracy >= {CONVERGENCE_THRESHOLD} "
        f"within the first {CONVERGENCE_EPOCH} epochs. Check data and features."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────
AGG_KEYS = ["balanced_accuracy", "macro_f1", "weighted_f1",
            "top1_accuracy", "auc_ovr_weighted"]
RPERP_KEYS = ["r_perp_val_at_best", "r_perp_train_at_best"]

agg_rows = [{"seed": r["seed"],
             **{k: r[k] for k in AGG_KEYS},
             **{k: r[k] for k in RPERP_KEYS}} for r in all_results]
agg_df   = pd.DataFrame(agg_rows)
mean_row = {**agg_df[AGG_KEYS + RPERP_KEYS].mean().to_dict(), "seed": "mean"}
std_row  = {**agg_df[AGG_KEYS + RPERP_KEYS].std().to_dict(),  "seed": "std"}
agg_df   = pd.concat([agg_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
agg_df.to_csv(os.path.join(OUT_DIR, "aggregated_metrics.csv"), index=False)

means = {k: float(mean_row[k]) for k in AGG_KEYS + RPERP_KEYS}
stds  = {k: float(std_row[k])  for k in AGG_KEYS + RPERP_KEYS}

# Per-class metrics (seed-mean)
report_mean = (
    pd.concat([r["report_df"] for r in all_results])
    .groupby(level=0).mean()
    .loc[[c for c in all_results[0]["report_df"].index]]
)
report_mean["support"] = report_mean["support"].round(0).astype(int)
report_mean.to_csv(os.path.join(OUT_DIR, "per_class_metrics.csv"))

# Confusion matrix (seed-mean, row-normalised)
cms    = np.stack([r["confusion_matrix"] for r in all_results], axis=0)
cm_mean = cms.mean(axis=0)
cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)
pd.DataFrame(cm_norm.round(4), index=class_names, columns=class_names).to_csv(
    os.path.join(OUT_DIR, "confusion_matrix.csv")
)

# Save config used
pd.DataFrame([{
    "hidden_dims":   str(list(HIDDEN_DIMS)),
    "dropout":       DROPOUT,
    "weight_decay":  WEIGHT_DECAY,
    "lr":            LR,
    "batch_size":    BATCH_SIZE,
    "max_epochs":    MAX_EPOCHS,
    "patience":      PATIENCE,
    "sched_patience": SCHED_PATIENCE,
    "sched_factor":  SCHED_FACTOR,
    "concurvity_lambda": CONCURVITY_LAMBDA,
    "sweep_winner_config_id": int(selected["config_id"]),
    "sweep_winner_val_balacc": float(selected["best_val_balacc"]),
}]).to_csv(os.path.join(OUT_DIR, "winning_config.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
for r in all_results:
    log = r["log_df"]
    ax.plot(log["epoch"], log["val_balanced_acc"],
            alpha=0.8, label=f"seed {r['seed']}")
ax.axhline(V6_LR_BASELINE["balanced_accuracy"], color="gray",
           linestyle="--", linewidth=1.2, label="v6 LR baseline (0.555)")
ax.axhline(V6_NAM_PHASE1["balanced_accuracy"], color="red",
           linestyle=":", linewidth=1, label="v6 NAM Phase-1 (0.498)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Val balanced accuracy")
ax.set_title(f"NAM v6 Final — Training curves\n"
             f"hidden={list(HIDDEN_DIMS)}, dropout={DROPOUT}, wd={WEIGHT_DECAY:.0e}")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────────────
def _delta(key: str, ref_val: float) -> str:
    d = means[key] - ref_val
    return f"({'+' if d >= 0 else ''}{d:.3f})"


cm_str = pd.DataFrame(cm_norm.round(3), index=class_names,
                      columns=class_names).to_string()

REPORT = f"""
==== NAM v6 Final Training (BiomedCLIP v6 features, 24 concepts) ====

Sweep winner: config {int(selected['config_id'])}
  hidden_dims  : {list(HIDDEN_DIMS)}
  dropout      : {DROPOUT}
  weight_decay : {WEIGHT_DECAY:.0e}
  sweep val balacc (seed 42) : {selected['best_val_balacc']:.4f}

Trained {len(SEEDS)} seeds with fixed lr={LR}, batch={BATCH_SIZE},
  max_epochs={MAX_EPOCHS}, patience={PATIENCE},
  ReduceLROnPlateau(patience={SCHED_PATIENCE}, factor={SCHED_FACTOR})

Concurvity regularization (Siems et al. 2023, arXiv:2305.11475):
  lambda = {CONCURVITY_LAMBDA}
  Objective: L_total = L_task + {CONCURVITY_LAMBDA} * R_perp
  R_perp = (1/K) * sum_k  mean_{{i<j}} |Corr(f_{{i,k}}(X_i), f_{{j,k}}(X_j))|
  (multiclass avg over K={N_CLASSES} classes; Pearson corr over batch dim)

Test-set results (mean +/- std across seeds):
  Balanced accuracy            : {means['balanced_accuracy']:.4f} +/- {stds['balanced_accuracy']:.4f}
  Macro F1                     : {means['macro_f1']:.4f} +/- {stds['macro_f1']:.4f}
  Weighted F1                  : {means['weighted_f1']:.4f} +/- {stds['weighted_f1']:.4f}
  Top-1 accuracy               : {means['top1_accuracy']:.4f} +/- {stds['top1_accuracy']:.4f}
  Multiclass AUC (OvR weighted): {means['auc_ovr_weighted']:.4f} +/- {stds['auc_ovr_weighted']:.4f}

Concurvity diagnostics (at best-val epoch, mean +/- std across seeds):
  R_perp val  : {means['r_perp_val_at_best']:.4f} +/- {stds['r_perp_val_at_best']:.4f}
  R_perp train: {means['r_perp_train_at_best']:.4f} +/- {stds['r_perp_train_at_best']:.4f}

Three-way comparison:
  vs v6 LR baseline  (0.555): {means['balanced_accuracy']:.4f} {_delta('balanced_accuracy', 0.555)}
  vs v6 NAM Phase-1  (0.498): {means['balanced_accuracy']:.4f} {_delta('balanced_accuracy', 0.498)}
  vs v5 NAM baseline (0.540): {means['balanced_accuracy']:.4f} {_delta('balanced_accuracy', 0.540)}

Per-class metrics (NAM v6 final, mean across seeds):
{report_mean[['precision', 'recall', 'f1-score', 'auc', 'support']].to_string(float_format='%.4f')}

Confusion matrix (row-normalised, seed-mean):
{cm_str}

Outputs -> {OUT_DIR}/
"""
print(REPORT)

with open(os.path.join(OUT_DIR, "metrics_summary.txt"), "w", encoding="utf-8") as f:
    f.write(REPORT.lstrip() + "\n")

print("Done.")
