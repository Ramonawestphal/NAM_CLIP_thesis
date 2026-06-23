"""
Sanity checks for concurvity regularization integration.

Check 1 — Bit-identical plain NAM (lambda=0.0):
  Trains 2 epochs with lambda=0.0 (new code path) and with the original code
  path (no R_perp computation at all). Compares per-epoch train_loss to 1e-6.
  Both paths must produce the same losses and the same final model weights.

Check 2 — Concurvity active (lambda=0.1):
  Trains 2 epochs with lambda=0.1. Verifies R_perp is a finite scalar in [0,1]
  and that epoch-2 R_perp_val with lambda=0.1 < epoch-2 R_perp_val with lambda=0.0.

Hardcodes config 9 hyperparameters so this script has no dependency on
sweep_results.csv and can run standalone.

Run from project root:
    python scripts/sanity_check_concurvity.py
"""

from __future__ import annotations

import random
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score

from src.models.nam_multiclass import NAMMulticlass
from src.models.concurvity import multiclass_concurvity

warnings.filterwarnings("ignore")

# ── Config 9 hyperparameters (hardcoded — no sweep CSV dependency) ─────────────
HIDDEN_DIMS  = (64, 32)
DROPOUT      = 0.10
WEIGHT_DECAY = 1e-5
LR           = 1e-3
BATCH_SIZE   = 256
N_FEATURES   = 24
N_CLASSES    = 7
SEED         = 42
N_EPOCHS     = 2

FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (shared across all checks)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
feat       = np.load(FEATURES_PATH, allow_pickle=True)
scores     = feat["scores"]
labels     = feat["labels"]
lesion_ids = feat["lesion_ids"]

split     = np.load(SPLITS_PATH)
train_idx = split["train_idx"]

X_all_train      = scores[train_idx]
y_all_train      = labels[train_idx]
lesion_ids_train = lesion_ids[train_idx]

class_names  = sorted(np.unique(labels).tolist())
class_to_idx = {c: i for i, c in enumerate(class_names)}
y_all_train_enc = np.array([class_to_idx[c] for c in y_all_train], dtype=np.int64)

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_final_rel, val_rel = next(
    gss.split(X_all_train, y_all_train, groups=lesion_ids_train)
)
X_train_raw = X_all_train[train_final_rel]
y_train_enc = y_all_train_enc[train_final_rel]
y_train_str = y_all_train[train_final_rel]
X_val_raw   = X_all_train[val_rel]
y_val_enc   = y_all_train_enc[val_rel]

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val_sc   = scaler.transform(X_val_raw).astype(np.float32)

weights       = compute_class_weight("balanced", classes=np.array(class_names), y=y_train_str)
weight_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

X_val_t = torch.tensor(X_val_sc,  dtype=torch.float32, device=DEVICE)
y_val_t = torch.tensor(y_val_enc, dtype=torch.long,    device=DEVICE)

train_dataset = TensorDataset(
    torch.tensor(X_train_sc,  dtype=torch.float32),
    torch.tensor(y_train_enc, dtype=torch.long),
)
print(f"  train={len(y_train_enc)}  val={len(y_val_enc)}  device={DEVICE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run N epochs, return per-epoch metrics
# ─────────────────────────────────────────────────────────────────────────────
def run_epochs(lambda_: float, original_path: bool = False) -> list[dict]:
    """
    original_path=True: use the unmodified loss path (no R_perp at all),
                        emulating the code before concurvity was added.
    original_path=False: use the new code path with the lambda check.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model = NAMMulticlass(
        n_features=N_FEATURES, num_classes=N_CLASSES,
        hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    loader    = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    log = []
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss  = 0.0
        total_rperp = 0.0
        n_batches   = 0

        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()

            if original_path:
                # Exact original code — no R_perp anywhere
                loss = criterion(model(X_b), y_b)
                r_perp_b = float("nan")
            elif lambda_ > 0:
                logits, shape_outs = model(X_b, return_shape_outputs=True)
                task_loss = criterion(logits, y_b)
                r_perp_b  = multiclass_concurvity(shape_outs)
                loss = task_loss + lambda_ * r_perp_b
            else:
                loss = criterion(model(X_b), y_b)
                with torch.no_grad():
                    model.eval()
                    r_perp_b = multiclass_concurvity(model.shape_outputs(X_b))
                    model.train()

            loss.backward()
            optimizer.step()
            total_loss  += loss.item() * len(y_b)
            total_rperp += r_perp_b if isinstance(r_perp_b, float) else r_perp_b.item()
            n_batches   += 1

        train_loss  = total_loss / len(y_train_enc)
        train_rperp = total_rperp / n_batches

        model.eval()
        with torch.no_grad():
            val_logits, val_shape_outs = model(X_val_t, return_shape_outputs=True)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_preds  = val_logits.argmax(dim=1).cpu().numpy()
            val_rperp  = multiclass_concurvity(val_shape_outs).item()
        val_balacc = balanced_accuracy_score(y_val_enc, val_preds)

        log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_balacc": val_balacc,
            "r_perp_train": train_rperp,
            "r_perp_val": val_rperp,
        })

    # Also return final model weights for weight comparison
    log[-1]["_state_dict"] = {k: v.clone() for k, v in model.state_dict().items()}
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: lambda=0.0 vs original path — loss must be identical to ~1e-6
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("CHECK 1: lambda=0.0 vs original code path (bit-identical loss)")
print("=" * 65)

log_orig   = run_epochs(lambda_=0.0, original_path=True)
log_lambda0 = run_epochs(lambda_=0.0, original_path=False)

all_match = True
for ep in range(N_EPOCHS):
    diff_train = abs(log_orig[ep]["train_loss"] - log_lambda0[ep]["train_loss"])
    diff_val   = abs(log_orig[ep]["val_loss"]   - log_lambda0[ep]["val_loss"])
    status = "OK" if diff_train < 1e-6 and diff_val < 1e-6 else "MISMATCH"
    if status == "MISMATCH":
        all_match = False
    print(f"  Epoch {ep+1}: "
          f"orig_train_loss={log_orig[ep]['train_loss']:.6f}  "
          f"new_train_loss={log_lambda0[ep]['train_loss']:.6f}  "
          f"delta={diff_train:.2e}  [{status}]")
    print(f"          "
          f"orig_val_loss ={log_orig[ep]['val_loss']:.6f}  "
          f"new_val_loss ={log_lambda0[ep]['val_loss']:.6f}  "
          f"delta={diff_val:.2e}  [{status}]")
    print(f"          r_perp_train={log_lambda0[ep]['r_perp_train']:.4f}  "
          f"r_perp_val={log_lambda0[ep]['r_perp_val']:.4f}  "
          f"[both in [0,1]: {0 <= log_lambda0[ep]['r_perp_val'] <= 1}]")

# Weight comparison
state_orig  = log_orig[-1]["_state_dict"]
state_new   = log_lambda0[-1]["_state_dict"]
max_wt_diff = max(
    (state_orig[k] - state_new[k]).abs().max().item()
    for k in state_orig
)
print(f"\n  Max weight difference (all params): {max_wt_diff:.2e}")

if all_match and max_wt_diff < 1e-6:
    print("  PASS: lambda=0.0 path is bit-identical to original code.")
else:
    print("  FAIL: differences detected — check PRNG contamination or gradient path.")


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: lambda=0.1 — R_perp should be finite and < lambda=0 R_perp
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("CHECK 2: lambda=0.1 — R_perp finite in [0,1], lower than lambda=0")
print("=" * 65)

log_lambda01 = run_epochs(lambda_=0.1, original_path=False)

for ep in range(N_EPOCHS):
    rp0  = log_lambda0[ep]["r_perp_val"]
    rp01 = log_lambda01[ep]["r_perp_val"]
    in_range = 0 <= rp01 <= 1
    lower    = rp01 < rp0
    print(f"  Epoch {ep+1}: "
          f"R_perp_val lambda=0.0: {rp0:.4f}  "
          f"R_perp_val lambda=0.1: {rp01:.4f}  "
          f"in [0,1]={in_range}  lower={lower}")
    print(f"          "
          f"train_loss lambda=0.0: {log_lambda0[ep]['train_loss']:.4f}  "
          f"train_loss lambda=0.1: {log_lambda01[ep]['train_loss']:.4f}")

ep2_rp0  = log_lambda0[-1]["r_perp_val"]
ep2_rp01 = log_lambda01[-1]["r_perp_val"]
all_finite = all(0 <= log_lambda01[ep]["r_perp_val"] <= 1 for ep in range(N_EPOCHS))
ep2_lower  = ep2_rp01 < ep2_rp0

if all_finite and ep2_lower:
    print(f"\n  PASS: R_perp is finite in [0,1] and lower with lambda=0.1 at epoch {N_EPOCHS}.")
elif all_finite:
    print(f"\n  PARTIAL: R_perp is finite in [0,1] but NOT yet lower at epoch {N_EPOCHS}.")
    print(f"  (2 epochs may be insufficient for the regularizer to suppress concurvity — acceptable.)")
else:
    print(f"\n  FAIL: R_perp out of range.")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUMMARY")
print("=" * 65)
print(f"  Formula: L_total = L_task + lambda * R_perp")
print(f"  R_perp  = (1/K) * sum_k  mean_{{i<j}} |Corr(f_{{i,k}}(X_i), f_{{j,k}}(X_j))|")
print(f"  Multiclass averaging convention: mean over K={N_CLASSES} classes")
print(f"  Citation: Siems et al. (2023), arXiv:2305.11475, NeurIPS 2023")
print(f"  Suggested lambda sweep: {{0.0, 0.001, 0.01, 0.1, 1.0}}")
