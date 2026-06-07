"""
Shared utilities for the v7 corrected NAM pipeline.

Fixes addressed here:
  Issue 3  — set_all_seeds() calls random.seed (absent in train_nam_base.py)
  Issue 7  — standardize() returns a per-run scaler; callers save it per seed_dir
  Issue 8  — set_all_seeds() sets torch.cuda.manual_seed_all + cudnn determinism
  Issue 9  — train_one_run() accepts warmup_epochs and applies concurvity warm-up
  Issue 10 — weight_decay kept at the config value (not zeroed for concurvity);
             documented here as deliberate.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

import sys, pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.nam_multiclass import NAMMulticlass
from src.models.concurvity import multiclass_concurvity
from src.models.sparsity import group_lasso_penalty, feature_group_norms, apply_proximal_step

# ── Fixed dataset constants ────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
N_FEATURES    = 24
N_CLASSES     = 7

# ── Architecture sweep grid (identical to sweep_nam_v6.py) ────────────────────
import itertools
SWEEP_GRID = list(itertools.product(
    [(32, 16), (32, 32), (64, 32)],   # hidden_dims
    [0.10, 0.20],                      # dropout
    [1e-5, 1e-4],                      # weight_decay
))
# Config IDs 1-12, same mapping as sweep_nam_v6.py


# ─────────────────────────────────────────────────────────────────────────────
# Seeding (Issue 3, Issue 8)
# ─────────────────────────────────────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    """Seed all RNGs for reproducibility.

    Issue 3 fix: includes random.seed (absent in original train_nam_base.py).
    Issue 8 fix: seeds CUDA and sets determinism flags.
    Note: torch.backends.cudnn.deterministic=True may slow down training ~10%.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(
    features_path: str = FEATURES_PATH,
    splits_path:   str = SPLITS_PATH,
) -> dict:
    """Load feature matrix and train/test indices.  Does NOT create val split.

    Returns:
        scores       : (10015, 24) float32
        labels       : (10015,) str
        lesion_ids   : (10015,) str
        concept_names: list[str] len=24
        class_names  : list[str] len=7 sorted
        train_idx    : int array len=8020
        test_idx     : int array len=1995
    """
    feat          = np.load(features_path, allow_pickle=True)
    scores        = feat["scores"]
    labels        = feat["labels"]
    lesion_ids    = feat["lesion_ids"]
    concept_names = feat["concept_ids"].tolist()
    assert scores.shape == (10015, N_FEATURES), f"Bad shape: {scores.shape}"
    assert len(concept_names) == N_FEATURES

    split     = np.load(splits_path)
    train_idx = split["train_idx"]
    test_idx  = split["test_idx"]
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(np.union1d(train_idx, test_idx)) == scores.shape[0]

    class_names = sorted(np.unique(labels).tolist())
    assert len(class_names) == N_CLASSES

    return {
        "scores":        scores,
        "labels":        labels,
        "lesion_ids":    lesion_ids,
        "concept_names": concept_names,
        "class_names":   class_names,
        "train_idx":     train_idx,
        "test_idx":      test_idx,
    }


def make_fixed_val_split(
    X_all_train:       np.ndarray,
    y_all_train:       np.ndarray,
    lesion_ids_train:  np.ndarray,
    class_names:       List[str],
    val_random_state:  int = 42,
) -> dict:
    """Carve a fixed 80/20 val split from the training pool.

    GroupShuffleSplit with random_state=42 matches the v6 scripts.
    Returns relative indices into X_all_train (not absolute dataset indices).
    """
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=val_random_state)
    train_rel, val_rel = next(
        gss.split(X_all_train, y_all_train, groups=lesion_ids_train)
    )
    assert len(
        set(lesion_ids_train[train_rel]) & set(lesion_ids_train[val_rel])
    ) == 0, "Lesion leakage in val split"

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_all_enc    = np.array([class_to_idx[c] for c in y_all_train], dtype=np.int64)

    return {
        "train_rel": train_rel,
        "val_rel":   val_rel,
        "X_train":   X_all_train[train_rel],
        "y_train_enc": y_all_enc[train_rel],
        "y_train_str": y_all_train[train_rel],
        "X_val":     X_all_train[val_rel],
        "y_val_enc": y_all_enc[val_rel],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standardisation (Issue 7)
# ─────────────────────────────────────────────────────────────────────────────

def standardize(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], StandardScaler]:
    """Fit StandardScaler on X_train only; transform val and optionally test.

    Issue 7 fix: callers must save the returned scaler in the per-seed directory.
    Returns: (X_train_sc, X_val_sc, X_test_sc_or_None, scaler).
    """
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train).astype(np.float32)
    X_val_sc  = scaler.transform(X_val).astype(np.float32)
    X_test_sc = scaler.transform(X_test).astype(np.float32) if X_test is not None else None
    return X_tr_sc, X_val_sc, X_test_sc, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Model and optimiser construction
# ─────────────────────────────────────────────────────────────────────────────

def make_model(
    hidden_dims:   Sequence[int],
    dropout:       float,
    concept_names: List[str],
    device:        torch.device,
) -> NAMMulticlass:
    return NAMMulticlass(
        n_features=N_FEATURES,
        num_classes=N_CLASSES,
        hidden_dims=tuple(hidden_dims),
        dropout=dropout,
        concept_names=concept_names,
    ).to(device)


def make_optimizer_scheduler(
    model:          nn.Module,
    lr:             float,
    weight_decay:   float,
    sched_patience: int   = 5,
    sched_factor:   float = 0.5,
    scheduler_mode: str   = "max",
) -> Tuple[torch.optim.Adam, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Returns (optimizer, scheduler).

    Issue 10: weight_decay is kept at the config value throughout.  We do NOT
    zero it for concurvity or sparsity runs.  This keeps all conditions on
    identical gradient dynamics (a deliberate protocol choice; Siems et al. use
    no weight decay but we prioritise cross-condition comparability).
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_mode, factor=sched_factor,
        patience=sched_patience, min_lr=1e-6,
    )
    return optimizer, scheduler


def class_weight_tensor(
    y_train_str: np.ndarray,
    class_names: List[str],
    device:      torch.device,
) -> torch.Tensor:
    weights = compute_class_weight(
        "balanced", classes=np.array(class_names), y=y_train_str
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Core training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_run(
    *,
    model:              NAMMulticlass,
    optimizer:          torch.optim.Optimizer,
    scheduler,
    criterion:          nn.Module,
    train_dataset:      TensorDataset,
    X_val_t:            torch.Tensor,
    y_val_t:            torch.Tensor,
    y_val_enc:          np.ndarray,
    max_epochs:         int,
    patience:           int,
    batch_size:         int,
    device:             torch.device,
    concurvity_lambda:  float = 0.0,
    warmup_epochs:      int   = 0,
    sparsity_lambda:    float = 0.0,
    proximal_sparsity:  bool  = True,
    save_path:          Optional[str] = None,
    verbose_every:      int   = 10,
    scheduler_watches:  str   = "val_balacc",
) -> dict:
    """General-purpose training loop used by all v7 scripts.

    Issue 9 fix: concurvity penalty is zeroed for epochs 0..(warmup_epochs-1).
      Default is warmup_epochs=0 (no warm-up); see diagnostic experiment in
      results/v7/diagnostic_warmup/ which confirmed this for HAM10000/lambda_c=3.

    Issue 4 fix (warm-start context): save_path is saved on every *improvement*
      to val_balacc (same as original), so the returned best checkpoint is the
      best-val checkpoint within this run, not the end-of-patience state.

    Returns dict with keys:
      best_val_balacc, best_epoch, log_df, best_state_dict (if save_path given)
    """
    pin_memory = (device.type == "cuda")
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )

    best_val_balacc = -1.0
    best_epoch      = -1
    patience_ctr    = 0
    training_log    = []
    best_state_dict = None
    early_stopped   = False

    for epoch in range(max_epochs):
        # ── Issue 9: concurvity warm-up ───────────────────────────────────────
        eff_lambda_c = concurvity_lambda if epoch >= warmup_epochs else 0.0

        # ── Train pass ────────────────────────────────────────────────────────
        model.train()
        total_loss    = 0.0
        total_rperp   = 0.0
        total_rsparse = 0.0
        n_batches     = 0

        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            if eff_lambda_c > 0:
                logits, shape_outs = model(X_b, return_shape_outputs=True)
                task_loss = criterion(logits, y_b)
                r_perp_b  = multiclass_concurvity(shape_outs)
                loss = task_loss + eff_lambda_c * r_perp_b
            else:
                logits = model(X_b)
                loss   = criterion(logits, y_b)
                with torch.no_grad():
                    model.eval()
                    r_perp_b = multiclass_concurvity(model.shape_outputs(X_b))
                    model.train()

            # Subgradient sparsity (ablation only; proximal is default)
            if sparsity_lambda > 0 and not proximal_sparsity:
                loss = loss + sparsity_lambda * group_lasso_penalty(model)

            loss.backward()
            optimizer.step()

            # Proximal sparsity (default)
            if sparsity_lambda > 0 and proximal_sparsity:
                apply_proximal_step(
                    model,
                    optimizer.param_groups[0]["lr"],
                    sparsity_lambda,
                )

            with torch.no_grad():
                r_sparse_b = group_lasso_penalty(model)

            total_loss    += loss.item() * len(y_b)
            total_rperp   += r_perp_b.item()
            total_rsparse += r_sparse_b.item()
            n_batches     += 1

        n_train = len(train_dataset)
        train_loss    = total_loss    / n_train
        train_rperp   = total_rperp   / n_batches
        train_rsparse = total_rsparse / n_batches

        # ── Val pass ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits, val_shape = model(X_val_t, return_shape_outputs=True)
            val_loss_ce = criterion(val_logits, y_val_t).item()
            val_preds   = val_logits.argmax(dim=1).cpu().numpy()
            val_rperp   = multiclass_concurvity(val_shape).item()
        val_balacc = balanced_accuracy_score(y_val_enc, val_preds)
        # Loss used by the LR scheduler and (in warm-start) for patience
        val_loss_full = val_loss_ce + eff_lambda_c * val_rperp

        current_lr = optimizer.param_groups[0]["lr"]

        training_log.append({
            "epoch":            epoch + 1,
            "train_loss":       train_loss,
            "val_loss":         val_loss_ce,
            "val_loss_full":    val_loss_full,
            "val_balanced_acc": val_balacc,
            "lr":               current_lr,
            "r_perp_train":     train_rperp,
            "r_perp_val":       val_rperp,
            "r_sparse_train":   train_rsparse,
            "warmup_active":    int(eff_lambda_c == 0 and concurvity_lambda > 0),
        })

        # Step scheduler on val_balacc (matches original v6 scripts)
        if scheduler_watches == "val_balacc":
            scheduler.step(val_balacc)
        else:
            scheduler.step(val_loss_full)

        if (epoch + 1) % verbose_every == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d} | loss={train_loss:.4f} "
                f"val_balacc={val_balacc:.4f} lr={current_lr:.2e} "
                f"R_perp={train_rperp:.4f} R_sparse={train_rsparse:.6f}"
                + (f" [warmup]" if eff_lambda_c == 0 and concurvity_lambda > 0 else "")
            )

        # ── Fix A2: post-warmup checkpoint reset ─────────────────────────────
        # When concurvity activates (epoch == warmup_epochs), discard any
        # checkpoint earned during warmup and restart patience/best tracking.
        # This ensures the saved model is the best under the full regularised
        # objective, not from the unpenalised warm-up phase.
        # For warmup_epochs == 0 the condition is never True (no-op).
        if warmup_epochs > 0 and epoch == warmup_epochs:
            best_val_balacc = -1.0
            best_epoch      = -1
            patience_ctr    = 0
            best_state_dict = None
            # Remove any warmup checkpoint so it cannot be loaded as fallback
            if save_path is not None and os.path.exists(save_path):
                os.remove(save_path)
            print(
                f"  [reset] Checkpoint tracking restarted at epoch {epoch+1} "
                f"(concurvity now active, lambda_c={concurvity_lambda:.4g})"
            )

        # ── Best checkpoint (Issue 4 fix: save on improvement, not at end) ───
        if val_balacc > best_val_balacc + 1e-4:
            best_val_balacc = val_balacc
            best_epoch      = epoch + 1
            patience_ctr    = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                early_stopped = True
                print(
                    f"  Early stop at epoch {epoch+1} "
                    f"(best epoch {best_epoch}, val_balacc={best_val_balacc:.4f})"
                )
                break

    import pandas as pd
    log_df = pd.DataFrame(training_log)

    # Restore best weights into the model
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    elif save_path is not None and os.path.exists(save_path):
        model.load_state_dict(
            torch.load(save_path, map_location=device, weights_only=True)
        )

    return {
        "best_val_balacc": best_val_balacc,
        "best_epoch":      best_epoch,
        "log_df":          log_df,
        "early_stopped":   early_stopped,
        "total_epochs":    len(training_log),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test set evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(
    model:       NAMMulticlass,
    X_test_t:    torch.Tensor,
    y_test:      np.ndarray,
    class_names: List[str],
) -> dict:
    """Full metric suite on held-out test set.  Called once per seed per run."""
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
    auc_ovr  = roc_auc_score(
        y_test, proba, multi_class="ovr", average="weighted", labels=class_names
    )

    per_cls_auc = {
        cls: roc_auc_score((y_test == cls).astype(int), proba[:, i])
        for i, cls in enumerate(class_names)
    }
    report_dict = classification_report(
        y_test, y_pred_str, labels=class_names, output_dict=True, zero_division=0
    )
    report_df = (
        __import__("pandas").DataFrame(report_dict).T.loc[class_names]
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
        "proba":             proba,
        "y_pred_str":        y_pred_str,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helper (Issue 5 fix)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_seed_results(rows: list, keys: list) -> dict:
    """Compute mean and std (ddof=1, i.e. pandas default) across seeds.

    Issue 5 fix: uses pandas .std() consistently (N-1 denominator).
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    means = df[keys].mean().to_dict()
    stds  = df[keys].std().to_dict()   # ddof=1 (pandas default)
    return {"means": means, "stds": stds}


# ─────────────────────────────────────────────────────────────────────────────
# Flag file helper
# ─────────────────────────────────────────────────────────────────────────────

def write_step_flag(results_v7_dir: str, step_n: int) -> None:
    from datetime import datetime, timezone
    flag_path = os.path.join(results_v7_dir, f"STEP_{step_n}_COMPLETE.flag")
    with open(flag_path, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat() + "\n")
    print(f"  [flag] {flag_path}")
