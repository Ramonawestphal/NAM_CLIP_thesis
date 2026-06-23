"""
Hyperparameter sweep for NAM v6 Phase-1 (BiomedCLIP v6 features, 24 concepts).

Grid: 3 subnet widths x 2 dropout values x 2 weight-decay values = 12 configs.
Single seed (42) for speed. Uses the same train_final/val split as v6_base.

Outputs -> reports/nam/v6_sweep/
    config_{N}/best_model.pt
    config_{N}/training_log.csv
    sweep_results.csv

Run from project root:
    python scripts/sweep_nam_v6.py
"""

from __future__ import annotations

import os
import random
import sys
import pathlib
import itertools

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
    roc_auc_score,
)

from src.models.nam_multiclass import NAMMulticlass

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
OUT_DIR       = "reports/nam/v6_sweep"

# ── Fixed hyperparameters (shared across all sweep configs) ───────────────────
SEED          = 42
LR            = 1e-3
BATCH_SIZE    = 256
MAX_EPOCHS    = 80       # reduced from 100 — faster sweep; final run uses 100
PATIENCE      = 15
SCHED_PATIENCE = 5
SCHED_FACTOR   = 0.5
N_FEATURES    = 24
N_CLASSES     = 7

# Suspicious val acc threshold — flag for manual check
SUSPICIOUS_VAL_THRESHOLD = 0.62

# Baselines for comparison in winner block
V6_LR_BASELINE   = 0.555
V6_NAM_PHASE1    = 0.498

# ── Sweep grid ────────────────────────────────────────────────────────────────
GRID = list(itertools.product(
    [(32, 16), (32, 32), (64, 32)],   # subnet_hidden
    [0.10, 0.20],                      # dropout
    [1e-5, 1e-4],                      # weight_decay
))
# Order: hidden varies slowest, then dropout, weight_decay fastest
# Config IDs 1-12

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("WARNING: CUDA not available — sweep on CPU (~3-5 min per config).")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (done once, shared across all configs)
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading features and splits...")
feat       = np.load(FEATURES_PATH, allow_pickle=True)
scores     = feat["scores"]
labels     = feat["labels"]
lesion_ids = feat["lesion_ids"]

split     = np.load(SPLITS_PATH)
train_idx = split["train_idx"]
test_idx  = split["test_idx"]

X_all_train      = scores[train_idx]
y_all_train      = labels[train_idx]
lesion_ids_train = lesion_ids[train_idx]
X_test_raw       = scores[test_idx]
y_test           = labels[test_idx]

class_names  = sorted(np.unique(labels).tolist())
class_to_idx = {c: i for i, c in enumerate(class_names)}
y_all_train_enc = np.array([class_to_idx[c] for c in y_all_train], dtype=np.int64)
y_test_enc      = np.array([class_to_idx[c] for c in y_test],      dtype=np.int64)

# Val split — same random_state=42 as v6_base for direct comparability
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
) == 0, "Lesion leakage"

# Standardise (fit once on train_final — same as v6_base)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val_sc   = scaler.transform(X_val_raw).astype(np.float32)
X_test_sc  = scaler.transform(X_test_raw).astype(np.float32)

# Class weights (fit on train_final labels)
weights = compute_class_weight("balanced", classes=np.array(class_names), y=y_train_str)
weight_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

# Fixed tensors
X_val_t  = torch.tensor(X_val_sc,  dtype=torch.float32, device=DEVICE)
y_val_t  = torch.tensor(y_val_enc, dtype=torch.long,    device=DEVICE)
X_test_t = torch.tensor(X_test_sc, dtype=torch.float32, device=DEVICE)

train_dataset = TensorDataset(
    torch.tensor(X_train_sc,  dtype=torch.float32),
    torch.tensor(y_train_enc, dtype=torch.long),
)

print(f"  train_final: {len(y_train_enc)}  val: {len(y_val_enc)}  test: {len(y_test_enc)}")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-config training function
# ─────────────────────────────────────────────────────────────────────────────
def _train_one_config(
    config_id:    int,
    hidden_dims:  tuple,
    dropout:      float,
    weight_decay: float,
    config_dir:   str,
) -> dict:
    """Train one NAM config for SEED seeds. Returns test metrics dict."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model = NAMMulticlass(
        n_features=N_FEATURES, num_classes=N_CLASSES,
        hidden_dims=hidden_dims, dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=weight_decay
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

    for epoch in range(MAX_EPOCHS):
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

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_preds  = val_logits.argmax(dim=1).cpu().numpy()
        val_balacc = balanced_accuracy_score(y_val_enc, val_preds)
        current_lr = optimizer.param_groups[0]["lr"]

        training_log.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "val_balanced_acc": val_balacc, "lr": current_lr,
        })
        scheduler.step(val_balacc)

        if val_balacc > best_val_balacc + 1e-4:
            best_val_balacc = val_balacc
            best_epoch      = epoch + 1
            patience_ctr    = 0
            torch.save(model.state_dict(), os.path.join(config_dir, "best_model.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    # Save log
    log_df = pd.DataFrame(training_log)
    log_df.to_csv(os.path.join(config_dir, "training_log.csv"), index=False)
    final_train_loss = float(log_df.loc[log_df["val_balanced_acc"].idxmax(), "train_loss"])

    # Evaluate best checkpoint on test set
    model.load_state_dict(
        torch.load(os.path.join(config_dir, "best_model.pt"),
                   map_location=DEVICE, weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_proba  = torch.softmax(test_logits, dim=1).cpu().numpy()
    test_preds_enc = test_logits.argmax(dim=1).cpu().numpy()
    y_pred_str     = [class_names[i] for i in test_preds_enc]

    test_balacc = balanced_accuracy_score(y_test, y_pred_str)
    test_macro_f1 = f1_score(y_test, y_pred_str, average="macro", zero_division=0)
    test_auc = roc_auc_score(y_test, test_proba, multi_class="ovr",
                              average="weighted", labels=class_names)

    # Suspicion flag
    if best_val_balacc >= SUSPICIOUS_VAL_THRESHOLD:
        val_test_delta = best_val_balacc - test_balacc
        flag = (f"  *** SUSPICIOUS: val_balacc={best_val_balacc:.4f} >= "
                f"{SUSPICIOUS_VAL_THRESHOLD} threshold. "
                f"val-test delta={val_test_delta:+.4f}. "
                f"{'INVESTIGATE (large gap)' if val_test_delta > 0.08 else 'OK (gap reasonable)'}  ***")
        print(flag)

    return {
        "config_id":        config_id,
        "hidden":           str(list(hidden_dims)),
        "dropout":          dropout,
        "weight_decay":     weight_decay,
        "best_epoch":       best_epoch,
        "best_val_balacc":  best_val_balacc,
        "test_balacc":      test_balacc,
        "test_macro_f1":    test_macro_f1,
        "test_auc":         test_auc,
        "final_train_loss": final_train_loss,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sweep loop
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"NAM v6 hyperparameter sweep — {len(GRID)} configs, seed {SEED}")
print(f"  Fixed: lr={LR}, batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}")
print(f"  Grid : hidden x dropout x weight_decay")
print(f"{'='*65}\n")

all_results = []
current_best_val = -1.0
current_best_cfg = None

for cfg_idx, (hidden_dims, dropout, weight_decay) in enumerate(GRID, start=1):
    cfg_label = (f"hidden={list(hidden_dims)}, dropout={dropout}, "
                 f"weight_decay={weight_decay:.0e}")
    print(f"Config {cfg_idx:2d}/{len(GRID)}: {cfg_label}")

    config_dir = os.path.join(OUT_DIR, f"config_{cfg_idx}")
    os.makedirs(config_dir, exist_ok=True)

    result = _train_one_config(cfg_idx, hidden_dims, dropout, weight_decay, config_dir)
    all_results.append(result)

    print(f"  -> best_val={result['best_val_balacc']:.4f} @ ep{result['best_epoch']}  "
          f"test={result['test_balacc']:.4f}  "
          f"macro_f1={result['test_macro_f1']:.4f}  "
          f"auc={result['test_auc']:.4f}  "
          f"train_loss={result['final_train_loss']:.4f}")

    if result["best_val_balacc"] > current_best_val:
        current_best_val = result["best_val_balacc"]
        current_best_cfg = cfg_label
        print(f"  ** New best val so far: {current_best_val:.4f} **")
    print()

# ── Save results ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUT_DIR, "sweep_results.csv"), index=False)

# ── Training curves (val_balanced_acc overlay) ────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
for r in all_results:
    log = pd.read_csv(os.path.join(OUT_DIR, f"config_{r['config_id']}", "training_log.csv"))
    lbl = (f"C{r['config_id']} h={r['hidden']} d={r['dropout']} "
           f"wd={r['weight_decay']:.0e}")
    ax.plot(log["epoch"], log["val_balanced_acc"], alpha=0.7, linewidth=1, label=lbl)
ax.axhline(V6_LR_BASELINE, color="black", linestyle="--", linewidth=1.2,
           label=f"v6 LR baseline ({V6_LR_BASELINE})")
ax.axhline(V6_NAM_PHASE1, color="red", linestyle=":", linewidth=1,
           label=f"v6 NAM Phase-1 ({V6_NAM_PHASE1})")
ax.set_xlabel("Epoch"); ax.set_ylabel("Val balanced accuracy")
ax.set_title("NAM v6 sweep — validation curves (all 12 configs)")
ax.legend(fontsize=6, ncol=2, loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sweep_val_curves.png"), dpi=150)
plt.close(fig)

# ── Print top 5 + winner ──────────────────────────────────────────────────────
top5 = results_df.sort_values("best_val_balacc", ascending=False).head(5)
winner = results_df.sort_values("best_val_balacc", ascending=False).iloc[0]

print("\nTop 5 configurations by val balanced accuracy:")
print(top5[["config_id", "hidden", "dropout", "weight_decay",
            "best_epoch", "best_val_balacc", "test_balacc",
            "test_macro_f1", "test_auc", "final_train_loss"]].to_string(index=False))

print(f"""
==== Sweep Winner ====
  Config ID: {int(winner['config_id'])}
  Config   : hidden={winner['hidden']}, dropout={winner['dropout']}, weight_decay={winner['weight_decay']:.0e}
  Best val balanced accuracy : {winner['best_val_balacc']:.4f}
  Single-seed test bal. acc  : {winner['test_balacc']:.4f}
  Single-seed test macro F1  : {winner['test_macro_f1']:.4f}
  Single-seed test AUC       : {winner['test_auc']:.4f}
  vs v6 LR baseline  (0.555) : {winner['test_balacc'] - V6_LR_BASELINE:+.4f}
  vs v6 NAM Phase-1  (0.498) : {winner['test_balacc'] - V6_NAM_PHASE1:+.4f}
======================

Outputs -> {OUT_DIR}/
NOTE: Winner checkpoint already saved at {OUT_DIR}/config_{int(winner['config_id'])}/best_model.pt
Run train_nam_v6_final.py to retrain with 5 seeds using the winning config.
""")
