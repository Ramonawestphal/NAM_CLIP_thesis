"""
Multi-seed validation of top sweep candidates for NAM v6.

Trains configs 9, 10, and 12 (all hidden=[64,32], varying dropout/weight_decay)
across 3 seeds each to resolve which configuration is robustly best given that
single-seed val accuracy is within typical seed noise across the top candidates.

Selection criterion: mean test balanced accuracy across seeds (not val), because
the val-test diagnostic showed val systematically overestimates test accuracy.
Test is used only for model selection among candidate configurations — the
partition itself is never touched during training.

Existing sweep outputs at reports/nam/v6_sweep/ are not modified.
New outputs -> reports/nam/v6_sweep_multiseed/

Run from project root:
    python scripts/sweep_nam_v6_multiseed.py
"""

from __future__ import annotations

import os
import random
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import warnings

import numpy as np
import pandas as pd
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
    roc_auc_score,
)

from src.models.nam_multiclass import NAMMulticlass

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
OUT_DIR       = "reports/nam/v6_sweep_multiseed"

# ── Seeds and candidate configurations ───────────────────────────────────────
SEEDS = [42, 43, 44]

CANDIDATES = [
    {"config_id": 9,  "hidden": (64, 32), "dropout": 0.10, "weight_decay": 1e-5},
    {"config_id": 10, "hidden": (64, 32), "dropout": 0.10, "weight_decay": 1e-4},
    {"config_id": 12, "hidden": (64, 32), "dropout": 0.20, "weight_decay": 1e-4},
]

# ── Fixed training settings (identical to original sweep) ────────────────────
LR             = 1e-3
BATCH_SIZE     = 256
MAX_EPOCHS     = 80
PATIENCE       = 15
SCHED_PATIENCE = 5
SCHED_FACTOR   = 0.5
N_FEATURES     = 24
N_CLASSES      = 7

# ── Reference baselines ───────────────────────────────────────────────────────
V6_LR_BASELINE = 0.555

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("WARNING: CUDA not available — training on CPU (~5 min per run, ~45 min total).")
else:
    print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Print run plan
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"NAM v6 multi-seed validation — {len(CANDIDATES)} configs x {len(SEEDS)} seeds "
      f"= {len(CANDIDATES) * len(SEEDS)} runs")
print(f"  Fixed: lr={LR}, batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}")
print(f"  ReduceLROnPlateau: factor={SCHED_FACTOR}, patience={SCHED_PATIENCE}")
print(f"  Seeds: {SEEDS}")
print(f"\n  Configurations:")
for c in CANDIDATES:
    print(f"    Config {c['config_id']:2d}: hidden={list(c['hidden'])}, "
          f"dropout={c['dropout']}, weight_decay={c['weight_decay']:.0e}")
print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (done once, shared across all runs)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading features and splits...")
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

# Same val split as all v6 runs (random_state=42, GroupShuffleSplit)
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

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val_sc   = scaler.transform(X_val_raw).astype(np.float32)
X_test_sc  = scaler.transform(X_test_raw).astype(np.float32)

weights = compute_class_weight("balanced", classes=np.array(class_names), y=y_train_str)
weight_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

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
# Per-(config, seed) training function
# ─────────────────────────────────────────────────────────────────────────────
def _train_one(
    config_id:    int,
    hidden_dims:  tuple,
    dropout:      float,
    weight_decay: float,
    seed:         int,
    run_dir:      str,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1:3d} | train_loss={train_loss:.4f} "
                  f"val_balacc={val_balacc:.4f} lr={current_lr:.2e}")

        if val_balacc > best_val_balacc + 1e-4:
            best_val_balacc = val_balacc
            best_epoch      = epoch + 1
            patience_ctr    = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"      Early stop at epoch {epoch+1} "
                      f"(best epoch {best_epoch}, val={best_val_balacc:.4f})")
                break

    log_df = pd.DataFrame(training_log)
    log_df.to_csv(os.path.join(run_dir, "training_log.csv"), index=False)
    final_train_loss = float(log_df.loc[log_df["val_balanced_acc"].idxmax(), "train_loss"])

    # Evaluate best checkpoint on test set
    model.load_state_dict(
        torch.load(os.path.join(run_dir, "best_model.pt"),
                   map_location=DEVICE, weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_proba  = torch.softmax(test_logits, dim=1).cpu().numpy()
    test_preds_enc = test_logits.argmax(dim=1).cpu().numpy()
    y_pred_str     = [class_names[i] for i in test_preds_enc]

    test_balacc   = balanced_accuracy_score(y_test, y_pred_str)
    test_macro_f1 = f1_score(y_test, y_pred_str, average="macro",    zero_division=0)
    test_wtd_f1   = f1_score(y_test, y_pred_str, average="weighted", zero_division=0)
    test_top1     = accuracy_score(y_test, y_pred_str)
    test_auc      = roc_auc_score(y_test, test_proba, multi_class="ovr",
                                   average="weighted", labels=class_names)

    return {
        "config_id":        config_id,
        "seed":             seed,
        "best_val_balacc":  best_val_balacc,
        "best_epoch":       best_epoch,
        "test_balacc":      test_balacc,
        "test_macro_f1":    test_macro_f1,
        "test_weighted_f1": test_wtd_f1,
        "test_top1_acc":    test_top1,
        "test_auc":         test_auc,
        "final_train_loss": final_train_loss,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
all_records = []
total_runs  = len(CANDIDATES) * len(SEEDS)
run_counter = 0

for cfg in CANDIDATES:
    cfg_id    = cfg["config_id"]
    hidden    = cfg["hidden"]
    dropout   = cfg["dropout"]
    wd        = cfg["weight_decay"]
    cfg_dir   = os.path.join(OUT_DIR, f"config_{cfg_id}")

    print(f"\n{'─'*65}")
    print(f"Config {cfg_id}: hidden={list(hidden)}, dropout={dropout}, "
          f"weight_decay={wd:.0e}")
    print(f"{'─'*65}")

    for seed in SEEDS:
        run_counter += 1
        print(f"\n  Run {run_counter}/{total_runs} — seed {seed}")

        seed_dir = os.path.join(cfg_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        result = _train_one(cfg_id, hidden, dropout, wd, seed, seed_dir)
        all_records.append(result)

        print(f"      -> val={result['best_val_balacc']:.4f}  "
              f"test={result['test_balacc']:.4f}  "
              f"macro_f1={result['test_macro_f1']:.4f}  "
              f"auc={result['test_auc']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate and save
# ─────────────────────────────────────────────────────────────────────────────
records_df = pd.DataFrame(all_records)
records_df.to_csv(os.path.join(OUT_DIR, "all_seed_results.csv"), index=False)

agg_rows = []
for cfg in CANDIDATES:
    cfg_id  = cfg["config_id"]
    subset  = records_df[records_df["config_id"] == cfg_id]
    agg_rows.append({
        "config_id":      cfg_id,
        "hidden":         str(list(cfg["hidden"])),
        "dropout":        cfg["dropout"],
        "weight_decay":   cfg["weight_decay"],
        "mean_val":       subset["best_val_balacc"].mean(),
        "std_val":        subset["best_val_balacc"].std(),
        "mean_test":      subset["test_balacc"].mean(),
        "std_test":       subset["test_balacc"].std(),
        "mean_macro_f1":  subset["test_macro_f1"].mean(),
        "std_macro_f1":   subset["test_macro_f1"].std(),
        "mean_auc":       subset["test_auc"].mean(),
        "std_auc":        subset["test_auc"].std(),
        "mean_train_loss": subset["final_train_loss"].mean(),
        "n_seeds":        len(subset),
    })

agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv(os.path.join(OUT_DIR, "multiseed_results.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Per-config summary table
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("Per-config results:")
print(f"{'='*65}")

for cfg in CANDIDATES:
    cfg_id  = cfg["config_id"]
    subset  = records_df[records_df["config_id"] == cfg_id]
    agg     = agg_df[agg_df["config_id"] == cfg_id].iloc[0]

    print(f"\nConfig {cfg_id} (hidden={list(cfg['hidden'])}, "
          f"dropout={cfg['dropout']}, weight_decay={cfg['weight_decay']:.0e}):")
    for _, row in subset.iterrows():
        print(f"  Seed {int(row['seed'])}: "
              f"val={row['best_val_balacc']:.4f}  "
              f"test={row['test_balacc']:.4f}  "
              f"macro_f1={row['test_macro_f1']:.4f}  "
              f"auc={row['test_auc']:.4f}")
    print(f"  Mean   : val={agg['mean_val']:.4f} +/- {agg['std_val']:.4f}  "
          f"test={agg['mean_test']:.4f} +/- {agg['std_test']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Winner selection
# ─────────────────────────────────────────────────────────────────────────────
agg_sorted = agg_df.sort_values("mean_test", ascending=False).reset_index(drop=True)
winner_row = agg_sorted.iloc[0]
others     = agg_sorted.iloc[1:]

winner_cfg = next(c for c in CANDIDATES if c["config_id"] == int(winner_row["config_id"]))

# Robustness check: is winner-vs-runner-up gap larger than the std of either?
runner_up  = agg_sorted.iloc[1]
gap        = float(winner_row["mean_test"]) - float(runner_up["mean_test"])
winner_std = float(winner_row["std_test"])
runner_std = float(runner_up["std_test"])
is_tie     = gap < max(winner_std, runner_std)

# Tiebreaker: lower dropout (simpler), then lower weight_decay
def _tiebreak_rank(cfg_id: int) -> tuple:
    c = next(x for x in CANDIDATES if x["config_id"] == cfg_id)
    return (c["dropout"], c["weight_decay"])

print(f"\n{'='*65}")
print("==== Multi-seed Sweep Winner ====")
print(f"  Config ID: {int(winner_row['config_id'])}")
print(f"  Config   : hidden={list(winner_cfg['hidden'])}, "
      f"dropout={winner_cfg['dropout']}, "
      f"weight_decay={winner_cfg['weight_decay']:.0e}")
print(f"  Mean test balanced accuracy : "
      f"{winner_row['mean_test']:.4f} +/- {winner_row['std_test']:.4f} (across {int(winner_row['n_seeds'])} seeds)")
print(f"  Mean test AUC               : "
      f"{winner_row['mean_auc']:.4f} +/- {winner_row['std_auc']:.4f}")
print(f"  vs v6 LR baseline (0.555)   : "
      f"Delta {winner_row['mean_test'] - V6_LR_BASELINE:+.4f}")
print(f"\n  Other configs:")
for _, row in others.iterrows():
    delta_from_winner = float(row["mean_test"]) - float(winner_row["mean_test"])
    print(f"    Config {int(row['config_id'])}: "
          f"{row['mean_test']:.4f} +/- {row['std_test']:.4f} "
          f"(Delta from winner: {delta_from_winner:+.4f})")
print(f"\n  Robustness check:")
print(f"    Winner-vs-runner-up gap (mean test): {gap:.4f}")
print(f"    std_test of winner     : {winner_std:.4f}")
print(f"    std_test of runner-up  : {runner_std:.4f}")

if is_tie:
    # Apply tiebreaker
    tied_ids  = [int(winner_row["config_id"]), int(runner_up["config_id"])]
    tb_winner = min(tied_ids, key=_tiebreak_rank)
    tb_cfg    = next(c for c in CANDIDATES if c["config_id"] == tb_winner)
    print(f"\n  *** TIE: gap ({gap:.4f}) < max std ({max(winner_std, runner_std):.4f})")
    print(f"  Applying tiebreaker: prefer lower dropout, then lower weight_decay.")
    print(f"  Tiebreak winner -> Config {tb_winner}: "
          f"hidden={list(tb_cfg['hidden'])}, dropout={tb_cfg['dropout']}, "
          f"weight_decay={tb_cfg['weight_decay']:.0e}")
    print(f"  Use config {tb_winner} for 5-seed final training.")
else:
    print(f"\n    Gap ({gap:.4f}) > both stds -> winner is clear.")
    print(f"    Use config {int(winner_row['config_id'])} for 5-seed final training.")

print("=================================")
print(f"\nOutputs -> {OUT_DIR}/")
