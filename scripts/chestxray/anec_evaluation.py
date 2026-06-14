"""
STEP 6 — ANEC (Accuracy at Number of Effective Concepts) evaluation.

Chest X-ray analogue of HAM10000 v7 anec_evaluation.py.

For each condition ∈ {sparsity_only, sparsity_conc} and seed ∈ {42…46}:
  1. Re-run the dense phase deterministically (set_all_seeds → same torch state as sweep)
  2. Traverse the warm-start sparsity path to the maximum step needed for K ∈ {5,8,10,15}
  3. At every step, verify n_active / val_balacc / val_r_perp match path.csv (determinism gate)
  4. Capture model state at the first-pass step for each K
  5. Evaluate each captured checkpoint on the held-out test set

Outputs (results/chestxray/anec_evaluation/):
  by_seed.csv, aggregated.csv, confusion_matrices.csv,
  surviving_concepts_summary.csv, summary_table.md, rule_a_secondary.md,
  run_config.json, STEP_6_COMPLETE.flag

Key differences from HAM10000 v7 anec_evaluation.py:
  - No pre-saved path checkpoints → re-traverse (HAM10000 loaded step-level .pt files)
  - 3-class problem (Normal / Bacteria / Virus) → per-class F1 + AUC + recall
  - Both macro and weighted multi-class AUC reported
  - test_idx IS loaded (this is the only test-set evaluation in the sparsity workstream)

Usage (from project root):
    python scripts/chestxray/anec_evaluation.py
    python scripts/chestxray/anec_evaluation.py --sanity_only
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts.v7._common import set_all_seeds, make_fixed_val_split
from src.models.nam_multiclass import NAMMulticlass
from src.models.concurvity import multiclass_concurvity
from src.models.sparsity import feature_group_norms, apply_proximal_step

# ── Constants ─────────────────────────────────────────────────────────────────
CONDITIONS  = ["sparsity_only", "sparsity_conc"]
SEEDS       = [42, 43, 44, 45, 46]
K_BUDGETS   = [15, 10, 8, 5]   # descending — largest first
CLASS_NAMES = ["normal", "bacteria", "virus"]  # index 0/1/2

LR               = 1e-3
BATCH_SIZE       = 256
N_FEATURES       = 17
NUM_CLASSES      = 3
VAL_RANDOM_STATE = 42
ZERO_THRESHOLD   = 1e-6

# Dense phase
MAX_DENSE_EPOCHS = 100
DENSE_PATIENCE   = 15
SCHED_PAT        = 5
SCHED_FAC        = 0.5

# Warm-start phase
MAX_WARM_EPOCHS = 30
WARM_PATIENCE   = 6
WARM_MIN_DELTA  = 1e-4

# Lambda schedule (must match sweep exactly)
LAMBDA_0  = 1.0
EPSILON   = 0.04
MAX_LAMBDA = 1e3

# Dense sanity check reference values (STEP 2 / STEP 4)
REF_PLAIN_NAM = {
    "mean_test_balacc": 0.7376, "std_test_balacc": 0.0091,
}
REF_CONCURVITY_ONLY = {
    "mean_test_balacc": 0.7214, "std_test_balacc": 0.0044,
}

# Determinism tolerance (val_balacc / val_r_perp checked to 4 decimal places)
DETERM_TOL = 5e-4   # ~ half a unit in the 4th decimal place

# Paths
FEATURES_PATH          = "data/features/biomedclip/chestxray_concept_scores_v4.npz"
SPLIT_PATH             = "data/splits/chestxray_outer_split.npz"
LABEL_MAP_PATH         = "results/chestxray/architecture_selection/label_mapping.json"
ARCH_WINNER_JSON       = "results/chestxray/architecture_selection/winning_config.json"
CONCURVITY_WINNER_JSON = "results/chestxray/concurvity_sweep/winner.json"
SWEEP_BASE             = "results/chestxray/sparsity_sweep"
OUT_DIR                = "results/chestxray/anec_evaluation"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_label_mapping() -> dict:
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"normal": 0, "bacteria": 1, "virus": 2}


def load_data(subtype_to_int: dict) -> dict:
    """Load all data including test_idx (ANEC uses test set)."""
    feat          = np.load(FEATURES_PATH, allow_pickle=True)
    scores        = feat["scores"]
    concept_names = feat["concept_names"].tolist()

    split          = np.load(SPLIT_PATH, allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]
    test_idx       = split["test_idx"]
    labels_subtype = split["labels_subtype"]
    patient_ids    = split["patient_ids"]

    labels_all = np.array(
        [subtype_to_int[s] for s in labels_subtype], dtype=np.int64
    )
    return {
        "scores":          scores,
        "concept_names":   concept_names,
        "labels_all":      labels_all,
        "train_pool_idx":  train_pool_idx,
        "test_idx":        test_idx,
        "patient_ids":     patient_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Operating point selection (reads path CSVs only; no training)
# ─────────────────────────────────────────────────────────────────────────────

def find_k_row(df: pd.DataFrame, K: int):
    """First step where n_active ≤ K (exact match preferred; fallback to below)."""
    exact = df[df["n_active"] == K]
    if len(exact) > 0:
        return exact.iloc[0], False
    below = df[df["n_active"] < K]
    if len(below) > 0:
        return below.iloc[0], True
    return None, False


def compute_operating_points() -> pd.DataFrame:
    """Read all path CSVs and build the 40-row operating-point table."""
    rows = []
    for cond in CONDITIONS:
        lam_c = 3.0 if cond == "sparsity_conc" else 0.0
        for seed in SEEDS:
            csv_path = os.path.join(SWEEP_BASE, cond, f"seed_{seed}", "path.csv")
            df = pd.read_csv(csv_path)
            for K in K_BUDGETS:
                row_info, fallback = find_k_row(df, K)
                if row_info is not None:
                    rows.append({
                        "condition":          cond,
                        "seed":              seed,
                        "target_K":          K,
                        "achieved_n_active": int(row_info["n_active"]),
                        "step":              int(row_info["step"]),
                        "lambda_s":          float(row_info["lambda_s"]),
                        "val_balacc_at_step": float(row_info["val_balacc_best_in_step"]),
                        "val_r_perp_at_step": float(row_info["val_r_perp_best_in_step"]),
                        "fallback":          fallback,
                    })
                else:
                    rows.append({
                        "condition": cond, "seed": seed, "target_K": K,
                        "achieved_n_active": None, "step": None, "lambda_s": None,
                        "val_balacc_at_step": None, "val_r_perp_at_step": None,
                        "fallback": None,
                    })
    return pd.DataFrame(rows)


def get_max_step_needed(op_df: pd.DataFrame, condition: str, seed: int) -> int:
    """Max step needed for this (condition, seed) across all K."""
    sub = op_df[
        (op_df["condition"] == condition) &
        (op_df["seed"] == seed) &
        (op_df["step"].notna())
    ]
    if len(sub) == 0:
        return 0
    return int(sub["step"].max())


# ─────────────────────────────────────────────────────────────────────────────
# Rule A (secondary analysis, reads path CSV only)
# ─────────────────────────────────────────────────────────────────────────────

def apply_rule_a(df: pd.DataFrame, dense_val_balacc: float) -> dict | None:
    """Last step where val_balacc_best_in_step ≥ dense_val_balacc − 0.02."""
    threshold = dense_val_balacc - 0.02
    col       = "val_balacc_best_in_step"
    above     = df[df[col] >= threshold]
    if len(above) == 0:
        return None
    row = above.iloc[-1]
    return {
        "step":       int(row["step"]),
        "lambda_s":   float(row["lambda_s"]),
        "n_active":   int(row["n_active"]),
        "val_balacc": float(row[col]),
        "threshold":  threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(
    model: nn.Module,
    X_test_t: torch.Tensor,
    y_test: np.ndarray,       # int64
) -> dict:
    """Evaluate on test set; return all required metrics."""
    model.eval()
    logits, _ = model(X_test_t, return_shape_outputs=True)
    probs     = F.softmax(logits, dim=1).cpu().numpy()
    preds     = probs.argmax(axis=1)

    balacc   = balanced_accuracy_score(y_test, preds)
    top1     = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    try:
        macro_auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = float("nan")
    try:
        weighted_auc = roc_auc_score(y_test, probs, multi_class="ovr", average="weighted")
    except Exception:
        weighted_auc = float("nan")

    per_metrics: dict = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        y_bin  = (y_test == c_idx).astype(int)
        p_pred = (preds == c_idx).astype(int)
        per_metrics[f"test_f1_{c_name}"]     = f1_score(y_bin, p_pred, zero_division=0)
        per_metrics[f"test_recall_{c_name}"] = recall_score(y_bin, p_pred, zero_division=0)
        try:
            per_metrics[f"test_auc_{c_name}"] = roc_auc_score(y_bin, probs[:, c_idx])
        except Exception:
            per_metrics[f"test_auc_{c_name}"] = float("nan")

    # Confusion matrix (raw and row-normalised)
    n_cls  = NUM_CLASSES
    cm_raw = np.zeros((n_cls, n_cls), dtype=np.int64)
    for true, pred in zip(y_test, preds):
        cm_raw[int(true), int(pred)] += 1
    row_sums = cm_raw.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_raw.astype(float) / row_sums

    cm_raw_flat  = {f"cm_raw_{i}{j}":  int(cm_raw[i, j])
                    for i in range(n_cls) for j in range(n_cls)}
    cm_norm_flat = {f"cm_norm_{i}{j}": round(float(cm_norm[i, j]), 6)
                    for i in range(n_cls) for j in range(n_cls)}

    return {
        "test_balacc":       balacc,
        "test_top1_acc":     top1,
        "test_macro_f1":     macro_f1,
        "test_macro_auc":    macro_auc,
        "test_auc_weighted": weighted_auc,
        **per_metrics,
        **cm_raw_flat,
        **cm_norm_flat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dense + warm-start path retraversal (mirrors run_sparsity_sweep.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def retraverse_path(
    *,
    seed:              int,
    condition:         str,
    concurvity_lambda: float,
    hidden_dims:       tuple,
    dropout:           float,
    weight_decay:      float,
    concept_names:     list,
    X_train_final_raw: np.ndarray,
    X_val_raw:         np.ndarray,
    y_train_final:     np.ndarray,
    y_val:             np.ndarray,
    path_df:           pd.DataFrame,   # reference path.csv for determinism check
    max_step_needed:   int,
    target_steps:      set,            # set of steps at which to capture model state
    dense_json_ref:    dict,           # stored dense JSON from sweep
) -> dict:
    """
    Fully replicates run_one_seed() from run_sparsity_sweep.py, capturing
    model states at the requested steps and verifying against path.csv.

    Test-set evaluation is NOT done here — caller applies the returned
    scaler to X_test_raw and evaluates captured checkpoints externally.

    Returns:
      "dense_val_balacc": float
      "dense_r_perp": float
      "dense_repro_ok": bool
      "determ_ok": bool
      "determ_first_fail": step or None
      "captured": dict mapping step -> {"model_state", "norms", "n_active",
                                        "val_balacc", "val_r_perp", "lambda_t"}
      "scaler": fitted StandardScaler
    """
    # ── Seed + scaler ─────────────────────────────────────────────────────────
    set_all_seeds(seed)
    pin_memory = (DEVICE.type == "cuda")

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train_final_raw).astype(np.float32)
    X_val_sc = scaler.transform(X_val_raw).astype(np.float32)
    # ── Class weights ─────────────────────────────────────────────────────────
    counts    = np.bincount(y_train_final, minlength=NUM_CLASSES)
    n_tr      = len(y_train_final)
    weights   = n_tr / (NUM_CLASSES * counts.astype(np.float64))
    w_tensor  = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)

    X_val_t = torch.tensor(X_val_sc, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val,    dtype=torch.long,    device=DEVICE)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc,       dtype=torch.float32),
        torch.tensor(y_train_final, dtype=torch.long),
    )

    def _make_loader(shuffle: bool) -> DataLoader:
        return DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=pin_memory
        )

    # ── Build model ───────────────────────────────────────────────────────────
    model = NAMMulticlass(
        n_features=N_FEATURES,
        num_classes=NUM_CLASSES,
        hidden_dims=hidden_dims,
        dropout=dropout,
        concept_names=concept_names,
    ).to(DEVICE)

    def _eval_val_full():
        model.eval()
        with torch.no_grad():
            logits, shape_outs = model(X_val_t, return_shape_outputs=True)
            val_loss_ce = criterion(logits, y_val_t).item()
            preds       = logits.argmax(dim=1).cpu().numpy()
            r_perp      = multiclass_concurvity(shape_outs).item()
        balacc = balanced_accuracy_score(y_val, preds)
        val_loss_full = val_loss_ce + concurvity_lambda * r_perp
        return float(balacc), float(r_perp), float(val_loss_ce), float(val_loss_full)

    # ── Phase 1: Dense training (mirrors run_sparsity_sweep.py exactly) ───────
    t0_dense = time.time()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHED_FAC, patience=SCHED_PAT, min_lr=1e-6
    )
    best_val_balacc  = -1.0
    patience_ctr     = 0
    best_dense_state = None

    for epoch in range(MAX_DENSE_EPOCHS):
        model.train()
        for X_b, y_b in _make_loader(shuffle=True):
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            if concurvity_lambda > 0:
                logits, shape_outs = model(X_b, return_shape_outputs=True)
                loss = (criterion(logits, y_b)
                        + concurvity_lambda * multiclass_concurvity(shape_outs))
            else:
                loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        val_balacc, r_perp, _, _ = _eval_val_full()
        scheduler.step(val_balacc)

        if val_balacc > best_val_balacc + 1e-4:
            best_val_balacc  = val_balacc
            patience_ctr     = 0
            best_dense_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= DENSE_PATIENCE:
                break

    if best_dense_state is not None:
        model.load_state_dict(
            {k: v.to(DEVICE) for k, v in best_dense_state.items()}
        )

    dense_val_balacc, dense_r_perp, _, _ = _eval_val_full()
    dense_elapsed = time.time() - t0_dense
    print(f"    [dense] val_balacc={dense_val_balacc:.4f}  r_perp={dense_r_perp:.4f}  "
          f"({dense_elapsed:.0f}s)")

    # ── Dense reproduction check ──────────────────────────────────────────────
    ref_balacc  = float(dense_json_ref["dense_val_balacc"])
    dense_repro_ok = abs(dense_val_balacc - ref_balacc) < DETERM_TOL
    print(f"    [dense repro] re-run={dense_val_balacc:.4f}  "
          f"stored={ref_balacc:.4f}  "
          f"{'✓' if dense_repro_ok else '✗ MISMATCH'}")

    # ── Phase 2: Build lambda schedule (same as sweep) ────────────────────────
    lambda_schedule: list[float] = []
    t = 0
    while len(lambda_schedule) < 200:   # generous upper bound
        lam = LAMBDA_0 * (1.0 + EPSILON) ** t
        if lam > MAX_LAMBDA:
            break
        lambda_schedule.append(lam)
        t += 1

    # Truncate to max_step_needed (1-indexed in path.csv, so we need steps 1..N)
    if max_step_needed > 0:
        lambda_schedule = lambda_schedule[:max_step_needed]

    print(f"    [path] traversing {len(lambda_schedule)} steps "
          f"(max_step_needed={max_step_needed})")

    # ── Traverse warm-start path ──────────────────────────────────────────────
    prev_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    prev_norms      = feature_group_norms(model)
    prev_n_active   = sum(1 for v in prev_norms.values() if v > ZERO_THRESHOLD)

    # Build reference lookup from path.csv: step -> row
    path_ref: dict[int, dict] = {}
    for _, row in path_df.iterrows():
        path_ref[int(row["step"])] = row.to_dict()

    captured: dict[int, dict] = {}   # step -> {"model_state", "norms", "step"}
    determ_ok         = True
    determ_first_fail = None
    determ_tally      = {"ok": 0, "fail": 0}
    print_banner_steps = set()   # steps to print even without events
    for s in range(0, max_step_needed + 1, 25):
        if s > 0:
            print_banner_steps.add(s)
    if max_step_needed > 0:
        print_banner_steps.add(max_step_needed)

    t0_path = time.time()
    for step_idx, lambda_t in enumerate(lambda_schedule):
        step = step_idx + 1   # 1-indexed

        # Warm-start from BEST previous-step state (Issue 4 fix)
        model.load_state_dict({k: v.to(DEVICE) for k, v in prev_state_dict.items()})

        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHED_FAC, patience=SCHED_PAT, min_lr=1e-6
        )

        best_step_val_loss = float("inf")
        best_step_state    = None
        no_improve_ctr     = 0

        for epoch in range(MAX_WARM_EPOCHS):
            model.train()
            for X_b, y_b in _make_loader(shuffle=True):
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                if concurvity_lambda > 0:
                    logits, shape_outs = model(X_b, return_shape_outputs=True)
                    loss = (criterion(logits, y_b)
                            + concurvity_lambda * multiclass_concurvity(shape_outs))
                else:
                    loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()
                apply_proximal_step(
                    model,
                    lr=optimizer.param_groups[0]["lr"],
                    sparsity_lambda=lambda_t,
                )

            _, _, _, val_loss_full = _eval_val_full()
            scheduler.step(val_loss_full)

            if val_loss_full < best_step_val_loss - WARM_MIN_DELTA:
                best_step_val_loss = val_loss_full
                no_improve_ctr     = 0
                best_step_state    = {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}
            else:
                no_improve_ctr += 1
                if no_improve_ctr >= WARM_PATIENCE:
                    break

        # Restore best-within-step state (Issue 4 fix)
        if best_step_state is not None:
            model.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_step_state.items()}
            )

        val_balacc_best, val_r_perp_best, _, _ = _eval_val_full()
        norms    = feature_group_norms(model)
        n_active = sum(1 for v in norms.values() if v > ZERO_THRESHOLD)

        # ── Determinism check ─────────────────────────────────────────────────
        ref = path_ref.get(step)
        if ref is not None:
            ref_n  = int(ref["n_active"])
            ref_ba = float(ref["val_balacc_best_in_step"])
            ref_rp = float(ref["val_r_perp_best_in_step"])

            ok_n  = (n_active == ref_n)
            ok_ba = abs(val_balacc_best - ref_ba) < DETERM_TOL
            ok_rp = abs(val_r_perp_best - ref_rp) < DETERM_TOL
            step_ok = ok_n and ok_ba and ok_rp

            if step_ok:
                determ_tally["ok"] += 1
            else:
                determ_tally["fail"] += 1
                if determ_ok:   # first failure
                    determ_ok         = False
                    determ_first_fail = step
                    print(f"\n    [DETERMINISM FAIL] step={step}")
                    print(f"      n_active:   got={n_active}, ref={ref_n}  "
                          f"{'✓' if ok_n else '✗'}")
                    print(f"      val_balacc: got={val_balacc_best:.6f}, "
                          f"ref={ref_ba:.6f}  diff={abs(val_balacc_best-ref_ba):.2e}  "
                          f"{'✓' if ok_ba else '✗'}")
                    print(f"      val_r_perp: got={val_r_perp_best:.6f}, "
                          f"ref={ref_rp:.6f}  diff={abs(val_r_perp_best-ref_rp):.2e}  "
                          f"{'✓' if ok_rp else '✗'}")

            # Print progress at elimination events and banner steps
            is_elim = (n_active < prev_n_active)
            is_tgt  = (step in target_steps)
            if is_elim or is_tgt or step in print_banner_steps:
                tag = ""
                if is_elim:    tag += " ELIM"
                if is_tgt:     tag += " [K-TARGET]"
                status = "✓" if step_ok else "✗"
                print(
                    f"    step {step:3d}  λ={lambda_t:.3e}  n={n_active:2d}  "
                    f"balacc={val_balacc_best:.4f}  r_perp={val_r_perp_best:.4f}  "
                    f"{status}{tag}"
                )
        else:
            # step beyond path.csv rows (early termination in original sweep)
            pass

        # ── Capture model state at K-target steps ─────────────────────────────
        if step in target_steps:
            captured[step] = {
                "step":        step,
                "lambda_t":    lambda_t,
                "n_active":    n_active,
                "val_balacc":  val_balacc_best,
                "val_r_perp":  val_r_perp_best,
                "model_state": {k: v.cpu().clone()
                                for k, v in model.state_dict().items()},
                "norms":       {k: float(v) for k, v in norms.items()},
            }

        # Advance
        prev_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        prev_norms      = norms
        prev_n_active   = n_active

        if n_active == 0:
            break

    path_elapsed = time.time() - t0_path
    print(f"    [path] done in {path_elapsed/60:.1f} min  "
          f"determ: {determ_tally['ok']} ok, {determ_tally['fail']} fail")
    if determ_ok:
        print(f"    [determinism] PASS for {condition} seed {seed}")
    else:
        print(f"    [determinism] FAIL — first fail at step {determ_first_fail}")

    return {
        "dense_val_balacc":   dense_val_balacc,
        "dense_r_perp":       dense_r_perp,
        "dense_repro_ok":     dense_repro_ok,
        "determ_ok":          determ_ok,
        "determ_first_fail":  determ_first_fail,
        "captured":           captured,
        "scaler":             scaler,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pre-run sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(
    data:          dict,
    subtype_to_int: dict,
    val_split:     dict,
    op_df:         pd.DataFrame,
    operative_lam_c: float,
) -> None:
    print("\n" + "=" * 68)
    print("PRE-RUN SANITY CHECKS (1–6)")
    print("=" * 68)

    all_ok = True

    # 1. Sweep artefacts exist
    missing = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            p = os.path.join(SWEEP_BASE, cond, f"seed_{seed}", "path.csv")
            if not os.path.exists(p):
                missing.append(p)
    if missing:
        print(f"  [1] FAIL: {len(missing)} path.csv(s) missing:")
        for m in missing:
            print(f"       {m}")
        all_ok = False
    else:
        print(f"  [1] All 10 path.csv artefacts present  ✓")

    # 2. Path CSV row counts
    bad_counts = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            p   = os.path.join(SWEEP_BASE, cond, f"seed_{seed}", "path.csv")
            df  = pd.read_csv(p)
            nrows = len(df)
            # Accept 1..150 rows (may terminate early if all eliminated before step 150)
            if nrows < 1 or nrows > 150:
                bad_counts.append((cond, seed, nrows))
    if bad_counts:
        print(f"  [2] FAIL: unexpected row counts: {bad_counts}")
        all_ok = False
    else:
        print(f"  [2] All path CSVs have 1–150 rows  ✓")
        # Print counts briefly
        for cond in CONDITIONS:
            seeds_counts = []
            for seed in SEEDS:
                p  = os.path.join(SWEEP_BASE, cond, f"seed_{seed}", "path.csv")
                seeds_counts.append(str(len(pd.read_csv(p))))
            print(f"      {cond}: {', '.join(seeds_counts)} rows (seeds 42–46)")

    # 3. First-pass K table
    print(f"\n  [3] First-pass K operating points:")
    print(f"      {'condition':18s}  {'seed':4s}  {'K':4s}  "
          f"{'achieved':8s}  {'step':5s}  {'lambda_s':10s}  {'fallback':8s}")
    print(f"      {'-'*18}  {'-'*4}  {'-'*4}  {'-'*8}  {'-'*5}  {'-'*10}  {'-'*8}")
    n_fallbacks = 0
    n_missing   = 0
    for _, row in op_df.iterrows():
        if row["step"] is None:
            n_missing += 1
            print(f"      {row['condition']:18s}  {int(row['seed']):4d}  "
                  f"{int(row['target_K']):4d}  {'NOT REACHED':>8}")
        else:
            fb  = "YES" if row["fallback"] else "no"
            dev = f"(dev={int(row['target_K'])-int(row['achieved_n_active'])})" if row["fallback"] else ""
            if row["fallback"]:
                n_fallbacks += 1
            print(f"      {row['condition']:18s}  {int(row['seed']):4d}  "
                  f"{int(row['target_K']):4d}  {int(row['achieved_n_active']):8d}  "
                  f"{int(row['step']):5d}  {float(row['lambda_s']):10.4e}  "
                  f"{fb:8s} {dev}")
    print(f"\n      Summary: {n_fallbacks} fallbacks, {n_missing} NOT REACHED")
    if n_missing > 0:
        print(f"  [3] WARNING: {n_missing} (condition, seed, K) combinations not reachable.")
    else:
        print(f"  [3] All 40 operating points reachable  ✓")

    # 4. Val split reproducibility
    EXPECTED_TR10  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    EXPECTED_VAL10 = [29, 33, 47, 48, 52, 53, 55, 63, 64, 65]
    train_rel = val_split["train_rel"]
    val_rel   = val_split["val_rel"]
    got_tr10  = train_rel[:10].tolist()
    got_val10 = val_rel[:10].tolist()
    if got_tr10 != EXPECTED_TR10 or got_val10 != EXPECTED_VAL10:
        print(f"\n  [4] FAIL: val-split mismatch!")
        print(f"      expected train[:10] = {EXPECTED_TR10}")
        print(f"      got      train[:10] = {got_tr10}")
        all_ok = False
    else:
        print(f"\n  [4] Val-split indices match STEP 2/3/4/smoke/sweep  ✓")
        print(f"      train[:10]={got_tr10}  val[:10]={got_val10}")

    # 5. winner.json operative
    with open(CONCURVITY_WINNER_JSON, encoding="utf-8") as f:
        cw = json.load(f)
    if cw.get("selection_pending", True) or "operative_lambda_c" not in cw:
        print(f"\n  [5] FAIL: winner.json invalid.")
        all_ok = False
    else:
        print(f"\n  [5] winner.json: selection_pending=false, "
              f"operative_lambda_c={float(cw['operative_lambda_c'])}  ✓")

    # 6. Test set integrity
    test_idx  = data["test_idx"]
    labels_all = data["labels_all"]
    patient_ids = data["patient_ids"]
    train_pool_idx = data["train_pool_idx"]

    expected_test_n = 1198
    test_n          = len(test_idx)
    if test_n != expected_test_n:
        print(f"\n  [6] FAIL: test_idx.shape = ({test_n},), expected ({expected_test_n},)")
        all_ok = False
    else:
        print(f"\n  [6] test_idx.shape = ({test_n},)  ✓")

    y_test         = labels_all[test_idx]
    test_counts    = np.bincount(y_test, minlength=NUM_CLASSES)
    int_to_name    = {v: k for k, v in subtype_to_int.items()}
    for i in range(NUM_CLASSES):
        print(f"      {int_to_name[i]:10s}: {test_counts[i]:4d} "
              f"({100.*test_counts[i]/test_n:.1f}%)")

    train_pids = set(patient_ids[train_pool_idx])
    test_pids  = set(patient_ids[test_idx])
    overlap    = train_pids & test_pids
    if overlap:
        print(f"  [6] FAIL: {len(overlap)} patients overlap between train_pool and test!")
        all_ok = False
    else:
        print(f"      Patient overlap between train_pool and test: 0  ✓")

    print("\n" + "=" * 68)
    if all_ok:
        print("All 6 pre-run sanity checks passed.\n")
    else:
        print("ONE OR MORE PRE-RUN CHECKS FAILED. Review above output.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean_std(vals: list, ddof: int = 1) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    a   = np.array(vals, dtype=float)
    std = float(np.std(a, ddof=ddof)) if len(vals) > 1 else 0.0
    return float(np.mean(a)), std


def build_aggregated(by_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Seed-mean and seed-std per (condition, target_K), ddof=1."""
    numeric_cols = [c for c in by_seed_df.columns
                    if by_seed_df[c].dtype in (np.float64, np.int64, float, int)
                    and c not in ("seed", "target_K", "step", "achieved_n_active")]

    rows = []
    for cond in CONDITIONS:
        for K in K_BUDGETS + ["dense"]:
            sub = by_seed_df[
                (by_seed_df["condition"] == cond) &
                (by_seed_df["target_K"] == K) &
                (by_seed_df["test_balacc"].notna())
            ]
            if len(sub) == 0:
                continue
            ddof = 1 if len(sub) > 1 else 0
            row  = {"condition": cond, "target_K": K, "n_seeds": len(sub)}
            for col in numeric_cols:
                if col in sub.columns:
                    vals = sub[col].dropna().tolist()
                    m, s = _mean_std(vals, ddof=ddof)
                    row[f"mean_{col}"] = m
                    row[f"std_{col}"]  = s
            rows.append(row)
    return pd.DataFrame(rows)


def build_surviving_concepts_summary(
    all_surviving: dict,   # (cond, seed, K) -> list of concept names
    concept_names: list,
) -> pd.DataFrame:
    rows = []
    for cond in CONDITIONS:
        for K in K_BUDGETS:
            for c_name in concept_names:
                n_surv = sum(
                    1 for seed in SEEDS
                    if c_name in all_surviving.get((cond, seed, K), [])
                )
                rows.append({
                    "condition":       cond,
                    "target_K":        K,
                    "concept_name":    c_name,
                    "n_seeds_surviving": n_surv,
                    "survival_rate":   round(n_surv / len(SEEDS), 3),
                })
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["condition", "target_K", "n_seeds_surviving"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return df


def build_confusion_matrices_csv(
    all_cms: dict,   # (cond, seed, K) -> 3×3 float (row-normalised)
) -> pd.DataFrame:
    """One row per (condition, K) with seed-mean 3×3 row-normalised matrix flattened."""
    rows = []
    for cond in CONDITIONS:
        for K in K_BUDGETS:
            seed_cms = [all_cms[(cond, s, K)] for s in SEEDS
                        if (cond, s, K) in all_cms]
            if not seed_cms:
                continue
            mean_cm = np.mean(seed_cms, axis=0)
            row = {"condition": cond, "target_K": K}
            for i in range(NUM_CLASSES):
                for j in range(NUM_CLASSES):
                    row[f"mean_cm_{CLASS_NAMES[i]}_pred_{CLASS_NAMES[j]}"] = \
                        round(float(mean_cm[i, j]), 4)
            rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_table(
    agg_df: pd.DataFrame,
    dense_sanity: dict,   # cond -> {"mean_test_balacc", "seeds": [...]}
    out_dir: str,
) -> None:
    def _get(row_dict, col, fmt=".3f"):
        m = row_dict.get(f"mean_{col}", float("nan"))
        s = row_dict.get(f"std_{col}",  float("nan"))
        if np.isnan(m):
            return "—"
        return f"{m:{fmt}}±{s:{fmt}}"

    tbl = {}
    for _, row in agg_df.iterrows():
        key = (row["condition"], row["target_K"])
        tbl[key] = row.to_dict()

    lines = [
        "# ANEC Evaluation — Summary Table",
        "",
        "Accuracy at Number of Effective Concepts (ANEC). "
        "Chest X-ray three-way task (Normal / Bacteria / Virus). "
        "Test set (n=1,198). Mean ± std across 5 seeds (42–46).",
        "",
        "## Primary ANEC table",
        "",
        "| K | sparsity_only n_active | sparsity_only bal_acc | sparsity_only macro_AUC |"
        " sparsity_only R_perp | sparsity_conc n_active | sparsity_conc bal_acc |"
        " sparsity_conc macro_AUC | sparsity_conc R_perp | Δ bal_acc | Δ macro_AUC |",
        "|---|------------------------|----------------------|--------------------------|"
        "---------------------|------------------------|----------------------|"
        "--------------------------|---------------------|-----------|------------|",
    ]

    for K in K_BUDGETS:
        so = tbl.get(("sparsity_only", K), {})
        sc = tbl.get(("sparsity_conc",  K), {})

        na_so  = _get(so, "achieved_n_active", ".1f")
        ba_so  = _get(so, "test_balacc")
        auc_so = _get(so, "test_macro_auc")
        rp_so  = _get(so, "val_r_perp_at_step")

        na_sc  = _get(sc, "achieved_n_active", ".1f")
        ba_sc  = _get(sc, "test_balacc")
        auc_sc = _get(sc, "test_macro_auc")
        rp_sc  = _get(sc, "val_r_perp_at_step")

        m_ba_so  = so.get("mean_test_balacc",  float("nan"))
        m_ba_sc  = sc.get("mean_test_balacc",  float("nan"))
        m_auc_so = so.get("mean_test_macro_auc", float("nan"))
        m_auc_sc = sc.get("mean_test_macro_auc", float("nan"))

        d_ba  = f"{m_ba_sc-m_ba_so:+.3f}" if not (np.isnan(m_ba_so) or np.isnan(m_ba_sc)) else "—"
        d_auc = f"{m_auc_sc-m_auc_so:+.3f}" if not (np.isnan(m_auc_so) or np.isnan(m_auc_sc)) else "—"

        lines.append(
            f"| {K:2d} | {na_so} | {ba_so} | {auc_so} | {rp_so} |"
            f" {na_sc} | {ba_sc} | {auc_sc} | {rp_sc} | {d_ba} | {d_auc} |"
        )

    # Weighted AUC table
    lines += [
        "",
        "## Weighted AUC (OvR) table",
        "",
        "| K | sparsity_only AUC_w | sparsity_conc AUC_w | Δ AUC_w |",
        "|---|---------------------|---------------------|---------|",
    ]
    for K in K_BUDGETS:
        so = tbl.get(("sparsity_only", K), {})
        sc = tbl.get(("sparsity_conc",  K), {})
        auc_w_so = _get(so, "test_auc_weighted")
        auc_w_sc = _get(sc, "test_auc_weighted")
        m_w_so   = so.get("mean_test_auc_weighted", float("nan"))
        m_w_sc   = sc.get("mean_test_auc_weighted", float("nan"))
        d_w = f"{m_w_sc-m_w_so:+.3f}" if not (np.isnan(m_w_so) or np.isnan(m_w_sc)) else "—"
        lines.append(f"| {K:2d} | {auc_w_so} | {auc_w_sc} | {d_w} |")

    # Per-class table
    lines += [
        "",
        "## Per-class metrics at K=10 (seed-mean ± std)",
        "",
        "| Condition | Class | F1 | AUC | Recall |",
        "|-----------|-------|-----|-----|--------|",
    ]
    for cond in CONDITIONS:
        row = tbl.get((cond, 10), {})
        for c_name in CLASS_NAMES:
            f1  = _get(row, f"test_f1_{c_name}")
            auc = _get(row, f"test_auc_{c_name}")
            rec = _get(row, f"test_recall_{c_name}")
            lines.append(f"| {cond} | {c_name} | {f1} | {auc} | {rec} |")

    # Dense sanity check
    lines += [
        "",
        "## Dense sanity check (full model, λ_s=0)",
        "",
        "| Condition | Re-run mean test_balacc | Expected (STEP 2/4) | Δ | Pass? |",
        "|-----------|------------------------|---------------------|---|-------|",
    ]
    for cond in CONDITIONS:
        if cond not in dense_sanity:
            continue
        ds   = dense_sanity[cond]
        mean = ds["mean_test_balacc"]
        ref  = REF_PLAIN_NAM["mean_test_balacc"] if cond == "sparsity_only" \
               else REF_CONCURVITY_ONLY["mean_test_balacc"]
        diff = mean - ref
        ok   = abs(diff) < 0.005
        lines.append(
            f"| {cond} | {mean:.4f} | {ref:.4f} | {diff:+.4f} | "
            f"{'✅' if ok else '⚠️ MISMATCH'} |"
        )

    # HAM10000 cross-dataset comparison (numerical, no narrative)
    lines += [
        "",
        "## Cross-dataset Δ comparison (numerical, no narrative)",
        "",
        "HAM10000 v7 (from STEP 6b): K=20 Δ=0.000, K=15 Δ=+0.019, K=10 Δ=+0.141, K=8 Δ=+0.251, K=5 Δ=+0.216",
        "",
        "| K | Chest X-ray Δ bal_acc (sparsity_conc − sparsity_only) | HAM10000 Δ bal_acc |",
        "|---|-------------------------------------------------------|--------------------|",
    ]
    ham10k = {15: 0.019, 10: 0.141, 8: 0.251, 5: 0.216}
    for K in K_BUDGETS:
        so = tbl.get(("sparsity_only", K), {})
        sc = tbl.get(("sparsity_conc",  K), {})
        m_ba_so = so.get("mean_test_balacc", float("nan"))
        m_ba_sc = sc.get("mean_test_balacc", float("nan"))
        if not (np.isnan(m_ba_so) or np.isnan(m_ba_sc)):
            delta = m_ba_sc - m_ba_so
            d_str = f"{delta:+.3f}"
        else:
            d_str = "—"
        ham_str = f"{ham10k.get(K, 0.0):+.3f}" if K in ham10k else "—"
        lines.append(f"| {K:2d} | {d_str} | {ham_str} |")

    lines += [
        "",
        "---",
        "Δ = sparsity_conc − sparsity_only (positive: concurvity regularisation helps at this K).",
        "R_perp: mean absolute Pearson correlation of per-concept shape outputs on val set.",
        "K anchor: 10 (Miller 1956 working memory range 7±2).",
        "",
        "NOTE: Cross-dataset comparison is numerical only. Narrative interpretation "
        "belongs in the thesis discussion. Do NOT retroactively adjust operating points "
        "to match pre-registered block-collapse predictions — that comparison happens "
        "in a separate post-ANEC analysis.",
        "",
        "*Full per-seed data: `by_seed.csv`. Aggregated: `aggregated.csv`.*",
    ]

    with open(os.path.join(out_dir, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_rule_a_secondary(rule_a_rows: list, out_dir: str) -> None:
    lines = [
        "# Rule A Secondary Analysis",
        "",
        "Rule A: last step where val_balacc_best_in_step ≥ dense_val_balacc − 0.02.",
        "Identifies the maximum sparsity achievable at no accuracy cost on the val set.",
        "Computed from path.csv (no training involved — purely a path trajectory summary).",
        "",
        "| Condition | Seed | Dense val_balacc | Threshold | Rule A step | Lambda | n_active | val_balacc |",
        "|-----------|------|-----------------|-----------|-------------|--------|----------|------------|",
    ]
    for r in sorted(rule_a_rows, key=lambda x: (x["condition"], x["seed"])):
        if r["rule_a"] is None:
            lines.append(
                f"| {r['condition']} | {r['seed']} | {r['dense_val_balacc']:.4f} | "
                f"{r['threshold']:.4f} | — | — | — | — |"
            )
        else:
            ra = r["rule_a"]
            lines.append(
                f"| {r['condition']} | {r['seed']} | {r['dense_val_balacc']:.4f} | "
                f"{ra['threshold']:.4f} | {ra['step']} | {ra['lambda_s']:.4e} | "
                f"{ra['n_active']} | {ra['val_balacc']:.4f} |"
            )

    for cond in CONDITIONS:
        sub = [r for r in rule_a_rows if r["condition"] == cond and r["rule_a"]]
        if sub:
            n_acts = [r["rule_a"]["n_active"] for r in sub]
            lams   = [r["rule_a"]["lambda_s"]  for r in sub]
            lines += [
                "",
                f"**{cond}** — mean n_active at Rule A: "
                f"{np.mean(n_acts):.1f} (range {min(n_acts)}–{max(n_acts)}), "
                f"median lambda: {np.median(lams):.4e}",
            ]

    lines += [
        "",
        "---",
        "Rule A provides a secondary anchor alongside the ANEC K-budget trajectory.",
        "The ANEC K∈{5,8,10,15} grid is the primary evaluation.",
    ]

    with open(os.path.join(out_dir, "rule_a_secondary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sanity_only", action="store_true",
                        help="Run sanity checks then exit without training.")
    parser.add_argument("--arch_winner_json", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seed list for partial runs.")
    args = parser.parse_args()

    seeds_to_run = args.seeds or SEEDS
    out_dir      = args.out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ── Load architecture winner ───────────────────────────────────────────────
    arch_json = args.arch_winner_json or ARCH_WINNER_JSON
    with open(arch_json, encoding="utf-8") as f:
        arch = json.load(f)
    hidden_dims  = tuple(arch["hidden_dims"])
    dropout      = float(arch["dropout"])
    weight_decay = float(arch["weight_decay"])

    # ── Load concurvity operative lambda ──────────────────────────────────────
    with open(CONCURVITY_WINNER_JSON, encoding="utf-8") as f:
        cw = json.load(f)
    operative_lam_c = float(cw["operative_lambda_c"])
    CONC_LAM = {"sparsity_only": 0.0, "sparsity_conc": operative_lam_c}

    # ── Load data ──────────────────────────────────────────────────────────────
    subtype_to_int = load_label_mapping()
    int_to_name    = {v: k for k, v in subtype_to_int.items()}

    data           = load_data(subtype_to_int)
    scores         = data["scores"]
    concept_names  = data["concept_names"]
    labels_all     = data["labels_all"]
    train_pool_idx = data["train_pool_idx"]
    test_idx       = data["test_idx"]
    patient_ids    = data["patient_ids"]

    X_train_pool = scores[train_pool_idx]
    y_train_pool = labels_all[train_pool_idx]
    groups_pool  = patient_ids[train_pool_idx]
    X_test_raw   = scores[test_idx].astype(np.float32)
    y_test       = labels_all[test_idx]

    # ── Fixed val split ────────────────────────────────────────────────────────
    val_split = make_fixed_val_split(
        X_train_pool,
        y_train_pool.astype(str),
        groups_pool,
        ["0", "1", "2"],
        val_random_state=VAL_RANDOM_STATE,
    )
    train_rel = val_split["train_rel"]
    val_rel   = val_split["val_rel"]

    X_train_final_raw = X_train_pool[train_rel]
    X_val_raw_arr     = X_train_pool[val_rel]
    y_train_final     = y_train_pool[train_rel]
    y_val             = y_train_pool[val_rel]

    # ── Operating points ──────────────────────────────────────────────────────
    op_df = compute_operating_points()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    run_sanity_checks(data, subtype_to_int, val_split, op_df, operative_lam_c)

    if args.sanity_only:
        print("--sanity_only flag set. Exiting before evaluation.")
        return

    # ── Banner ─────────────────────────────────────────────────────────────────
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT), text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_hash = None

    print(f"\n{'='*68}")
    print(f"Chest X-ray ANEC Evaluation (STEP 6)")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Seeds: {seeds_to_run}")
    print(f"  K budgets: {K_BUDGETS}")
    print(f"  operative_lambda_c = {operative_lam_c} (from winner.json)")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {out_dir}/")
    print(f"{'='*68}\n")

    # ── Save run_config.json ──────────────────────────────────────────────────
    run_cfg = {
        "conditions":              CONDITIONS,
        "seeds":                   seeds_to_run,
        "k_budgets":               K_BUDGETS,
        "operative_lambda_c":      operative_lam_c,
        "concurvity_winner_source": CONCURVITY_WINNER_JSON,
        "val_random_state":        VAL_RANDOM_STATE,
        "lambda_0":                LAMBDA_0,
        "epsilon":                 EPSILON,
        "determ_tolerance":        DETERM_TOL,
        "config_id":               arch["config_id"],
        "hidden_dims":             list(hidden_dims),
        "dropout":                 dropout,
        "weight_decay":            weight_decay,
        "n_features":              N_FEATURES,
        "num_classes":             NUM_CLASSES,
        "n_test":                  int(len(test_idx)),
        "sweep_base":              SWEEP_BASE,
        "test_set_used":           True,
        "dense_sanity_ref":        {
            "sparsity_only": REF_PLAIN_NAM,
            "sparsity_conc":  REF_CONCURVITY_ONLY,
        },
        "timestamp":               datetime.now(timezone.utc).isoformat(),
        "git_commit":              git_hash,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    # ── Main evaluation loop ──────────────────────────────────────────────────
    by_seed_rows:     list[dict]  = []
    rule_a_rows:      list[dict]  = []
    dense_sanity:     dict        = {}   # cond -> list of test_balacc
    all_surviving:    dict        = {}   # (cond, seed, K) -> list of concept names
    all_cms:          dict        = {}   # (cond, seed, K) -> 3x3 array

    determ_report = {}   # (cond, seed) -> {"ok": bool, "first_fail": step or None}
    dense_repro_report = {}   # (cond, seed) -> bool

    t_total = time.time()

    for condition in CONDITIONS:
        lam_c = CONC_LAM[condition]
        conc_tag = str(lam_c).replace(".", "p")
        dense_sanity[condition] = {"balaccs": []}

        print(f"\n{'━'*68}")
        print(f"CONDITION: {condition}  |  λ_c={lam_c}")
        print(f"{'━'*68}")

        for seed in seeds_to_run:
            print(f"\n{'─'*60}")
            print(f"  [{condition}  seed={seed}]")
            print(f"{'─'*60}")

            # ── Build K-target steps for this seed ────────────────────────────
            seed_op = op_df[
                (op_df["condition"] == condition) & (op_df["seed"] == seed)
            ]
            # Map from step -> list of (target_K, achieved_n_active)
            step_to_targets: dict[int, list] = {}
            for _, oprow in seed_op.iterrows():
                if oprow["step"] is not None:
                    s = int(oprow["step"])
                    step_to_targets.setdefault(s, []).append(
                        (int(oprow["target_K"]), int(oprow["achieved_n_active"]))
                    )

            target_steps   = set(step_to_targets.keys())
            max_step_needed = get_max_step_needed(op_df, condition, seed)

            # ── Load path.csv for determinism check ───────────────────────────
            path_csv = os.path.join(SWEEP_BASE, condition, f"seed_{seed}", "path.csv")
            path_df  = pd.read_csv(path_csv)

            # ── Load dense meta JSON ───────────────────────────────────────────
            dense_json_path = os.path.join(
                SWEEP_BASE, condition, f"seed_{seed}",
                f"dense_seed{seed}_conc{conc_tag}.json"
            )
            with open(dense_json_path, encoding="utf-8") as f:
                dense_json_ref = json.load(f)
            dense_val_balacc_ref = float(dense_json_ref["dense_val_balacc"])

            # ── Retraverse path ───────────────────────────────────────────────
            # retraverse_path fits the scaler internally (same set_all_seeds(seed)
            # call ensures it is identical to the sweep's scaler). We use the
            # returned scaler to scale X_test_raw for test evaluation.
            t_seed = time.time()
            result = retraverse_path(
                seed=seed,
                condition=condition,
                concurvity_lambda=lam_c,
                hidden_dims=hidden_dims,
                dropout=dropout,
                weight_decay=weight_decay,
                concept_names=concept_names,
                X_train_final_raw=X_train_final_raw,
                X_val_raw=X_val_raw_arr,
                y_train_final=y_train_final,
                y_val=y_val,
                path_df=path_df,
                max_step_needed=max_step_needed,
                target_steps=target_steps,
                dense_json_ref=dense_json_ref,
            )

            scaler = result["scaler"]
            X_test_sc = scaler.transform(X_test_raw).astype(np.float32)
            X_test_t  = torch.tensor(X_test_sc, dtype=torch.float32, device=DEVICE)

            # Dense test evaluation using re-run model (rebuild from scratch fast path:
            # load the saved .pt checkpoint — model state is identical to re-run since
            # dense_repro_ok checked val_balacc match)
            determ_report[(condition, seed)]    = {
                "ok":         result["determ_ok"],
                "first_fail": result["determ_first_fail"],
            }
            dense_repro_report[(condition, seed)] = result["dense_repro_ok"]

            # Re-build model for dense test eval by loading the saved .pt
            dense_model = NAMMulticlass(
                n_features=N_FEATURES,
                num_classes=NUM_CLASSES,
                hidden_dims=hidden_dims,
                dropout=dropout,
                concept_names=concept_names,
            ).to(DEVICE)
            dense_pt = os.path.join(
                SWEEP_BASE, condition, f"seed_{seed}",
                f"dense_seed{seed}_conc{conc_tag}.pt"
            )
            dense_model.load_state_dict(
                torch.load(dense_pt, map_location=DEVICE, weights_only=True)
            )
            dense_model.eval()
            dense_test_m = evaluate_test(dense_model, X_test_t, y_test)
            dense_sanity[condition]["balaccs"].append(dense_test_m["test_balacc"])

            # ── Dense row ────────────────────────────────────────────────────
            by_seed_rows.append({
                "condition":          condition,
                "seed":               seed,
                "target_K":          "dense",
                "achieved_n_active": N_FEATURES,
                "step":              0,
                "lambda_s":          0.0,
                "val_balacc_at_step": float(dense_json_ref["dense_val_balacc"]),
                "val_r_perp_at_step": float(dense_json_ref.get("dense_r_perp", float("nan"))),
                "surviving_concepts": ";".join(concept_names),
                **dense_test_m,
            })

            # ── Rule A ────────────────────────────────────────────────────────
            ra = apply_rule_a(path_df, dense_val_balacc_ref)
            rule_a_rows.append({
                "condition":       condition,
                "seed":            seed,
                "dense_val_balacc": dense_val_balacc_ref,
                "threshold":       dense_val_balacc_ref - 0.02,
                "rule_a":          ra,
            })
            if ra:
                print(f"    [Rule A] step={ra['step']}, λ={ra['lambda_s']:.4e}, "
                      f"n_active={ra['n_active']}, val_balacc={ra['val_balacc']:.4f}")
            else:
                print(f"    [Rule A] No qualifying step.")

            # ── Captured K models: evaluate on test set ───────────────────────
            if not result["determ_ok"]:
                print(f"\n  *** DETERMINISM FAIL for {condition} seed {seed} ***")
                print(f"  *** Captured models at these steps are unreliable. ***")
                print(f"  *** STOPPING SWEEP. ***")
                # Still write partial by_seed.csv for debugging
                pd.DataFrame(by_seed_rows).to_csv(
                    os.path.join(out_dir, "by_seed_PARTIAL.csv"), index=False
                )
                sys.exit(1)

            # Evaluate each captured checkpoint
            model = NAMMulticlass(
                n_features=N_FEATURES,
                num_classes=NUM_CLASSES,
                hidden_dims=hidden_dims,
                dropout=dropout,
                concept_names=concept_names,
            ).to(DEVICE)

            evaluated_steps: dict[int, dict] = {}   # step -> test metrics + meta

            for step, cap in result["captured"].items():
                if step in evaluated_steps:
                    continue
                model.load_state_dict(
                    {k: v.to(DEVICE) for k, v in cap["model_state"].items()}
                )
                model.eval()
                test_m = evaluate_test(model, X_test_t, y_test)
                evaluated_steps[step] = {
                    "test_m":   test_m,
                    "norms":    cap["norms"],
                    "n_active": cap["n_active"],
                    "val_balacc": cap["val_balacc"],
                    "val_r_perp": cap["val_r_perp"],
                    "lambda_t":   cap["lambda_t"],
                }

            # Build by_seed rows for each K
            for step, targets in step_to_targets.items():
                if step not in evaluated_steps:
                    continue
                ev = evaluated_steps[step]
                surv = [c for c in concept_names if ev["norms"].get(c, 0) > ZERO_THRESHOLD]

                # Extract confusion matrix for the normalised summary
                raw_cm = np.array([[ev["test_m"][f"cm_raw_{i}{j}"]
                                    for j in range(NUM_CLASSES)]
                                   for i in range(NUM_CLASSES)], dtype=float)
                row_s  = raw_cm.sum(axis=1, keepdims=True)
                row_s[row_s == 0] = 1.0
                norm_cm = raw_cm / row_s

                for (target_K, achieved_n_active) in targets:
                    all_surviving[(condition, seed, target_K)] = surv
                    all_cms[(condition, seed, target_K)]        = norm_cm

                    row = {
                        "condition":          condition,
                        "seed":               seed,
                        "target_K":          target_K,
                        "achieved_n_active": achieved_n_active,
                        "step":              step,
                        "lambda_s":          ev["lambda_t"],
                        "val_balacc_at_step": ev["val_balacc"],
                        "val_r_perp_at_step": ev["val_r_perp"],
                        "surviving_concepts": ";".join(surv),
                        **ev["test_m"],
                    }
                    by_seed_rows.append(row)

                    print(
                        f"    K={target_K:2d} (achieved={achieved_n_active}): "
                        f"step={step}  test_balacc={ev['test_m']['test_balacc']:.4f}  "
                        f"macro_auc={ev['test_m']['test_macro_auc']:.4f}  "
                        f"r_perp={ev['val_r_perp']:.4f}"
                    )

            elapsed_seed = time.time() - t_seed
            print(f"\n  [{condition} seed {seed}] done in {elapsed_seed/60:.1f} min")

        # ── Condition-level dense sanity check ────────────────────────────────
        ds         = dense_sanity[condition]
        mean_dense = float(np.mean(ds["balaccs"])) if ds["balaccs"] else float("nan")
        std_dense  = float(np.std(ds["balaccs"], ddof=1)) if len(ds["balaccs"]) > 1 else 0.0
        ds["mean_test_balacc"] = mean_dense
        ds["std_test_balacc"]  = std_dense

        ref  = REF_PLAIN_NAM if condition == "sparsity_only" else REF_CONCURVITY_ONLY
        diff = mean_dense - ref["mean_test_balacc"]
        ok   = abs(diff) < 0.005

        print(f"\n  [Dense sanity check — {condition}]")
        print(f"    Re-run mean test_balacc = {mean_dense:.4f} ± {std_dense:.4f}")
        print(f"    Expected (STEP 2/4)     = {ref['mean_test_balacc']:.4f} ± "
              f"{ref['std_test_balacc']:.4f}")
        print(f"    Δ = {diff:+.4f}  {'✅ PASS' if ok else '⚠️  MISMATCH (>0.005)'}")

    # ── Write all artefacts ───────────────────────────────────────────────────
    by_seed_df = pd.DataFrame(by_seed_rows)
    by_seed_df.to_csv(os.path.join(out_dir, "by_seed.csv"), index=False)
    print(f"\nWritten: by_seed.csv  ({len(by_seed_df)} rows)")

    agg_df = build_aggregated(by_seed_df)
    agg_df.to_csv(os.path.join(out_dir, "aggregated.csv"), index=False)
    print(f"Written: aggregated.csv  ({len(agg_df)} rows)")

    surv_df = build_surviving_concepts_summary(all_surviving, concept_names)
    surv_df.to_csv(os.path.join(out_dir, "surviving_concepts_summary.csv"), index=False)
    print(f"Written: surviving_concepts_summary.csv  ({len(surv_df)} rows)")

    cm_df = build_confusion_matrices_csv(all_cms)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrices.csv"), index=False)
    print(f"Written: confusion_matrices.csv  ({len(cm_df)} rows)")

    write_summary_table(agg_df, dense_sanity, out_dir)
    print(f"Written: summary_table.md")

    write_rule_a_secondary(rule_a_rows, out_dir)
    print(f"Written: rule_a_secondary.md")

    # ── Determinism and dense repro report ────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"DETERMINISM REPORT")
    print(f"{'='*68}")
    determ_all_ok  = True
    dense_all_ok   = True
    for cond in CONDITIONS:
        for seed in seeds_to_run:
            dr = determ_report.get((cond, seed))
            ddr = dense_repro_report.get((cond, seed))
            if dr is None:
                continue
            d_ok = dr["ok"]
            de_ok = ddr if ddr is not None else True
            if not d_ok:
                determ_all_ok = False
            if not de_ok:
                dense_all_ok = False
            ff_str = f"  first_fail=step {dr['first_fail']}" if not d_ok else ""
            print(
                f"  {cond:<18}  seed={seed}  "
                f"determ={'✓' if d_ok else f'✗{ff_str}'}  "
                f"dense_repro={'✓' if de_ok else '✗'}"
            )
    print(f"\n  Overall determinism: {'ALL PASS ✓' if determ_all_ok else 'FAIL ✗'}")
    print(f"  Overall dense repro: {'ALL PASS ✓' if dense_all_ok else 'FAIL ✗'}")

    # ── Final ANEC table ──────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"HEADLINE ANEC TABLE (mean ± std, 5 seeds)")
    print(f"{'='*68}")
    print(f"  {'K':4s}  {'condition':18s}  {'n_active':8s}  "
          f"{'test_balacc':12s}  {'macro_auc':10s}  {'wt_auc':10s}  "
          f"{'val_r_perp':10s}")
    print(f"  {'-'*4}  {'-'*18}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for K in K_BUDGETS:
        for cond in CONDITIONS:
            sub = agg_df[(agg_df["condition"] == cond) & (agg_df["target_K"] == K)]
            if len(sub) == 0:
                continue
            row = sub.iloc[0]
            def _f(col, fmt=".4f"):
                m = row.get(f"mean_{col}", float("nan"))
                s = row.get(f"std_{col}",  float("nan"))
                return f"{m:{fmt}}±{s:.4f}" if not np.isnan(m) else "—"
            print(
                f"  {K:4d}  {cond:18s}  {_f('achieved_n_active', '.1f'):8s}  "
                f"{_f('test_balacc'):12s}  {_f('test_macro_auc'):10s}  "
                f"{_f('test_auc_weighted'):10s}  {_f('val_r_perp_at_step'):10s}"
            )
        if K != K_BUDGETS[-1]:
            # Print Δ row
            so_row = agg_df[(agg_df["condition"] == "sparsity_only") & (agg_df["target_K"] == K)]
            sc_row = agg_df[(agg_df["condition"] == "sparsity_conc")  & (agg_df["target_K"] == K)]
            if len(so_row) > 0 and len(sc_row) > 0:
                d_ba  = sc_row.iloc[0].get("mean_test_balacc",  float("nan")) - \
                         so_row.iloc[0].get("mean_test_balacc",  float("nan"))
                d_auc = sc_row.iloc[0].get("mean_test_macro_auc", float("nan")) - \
                         so_row.iloc[0].get("mean_test_macro_auc", float("nan"))
                d_rp  = sc_row.iloc[0].get("mean_val_r_perp_at_step", float("nan")) - \
                         so_row.iloc[0].get("mean_val_r_perp_at_step", float("nan"))
                if not any(np.isnan(x) for x in [d_ba, d_auc, d_rp]):
                    print(
                        f"  {'Δ':4s}  {'(conc − only)':18s}  {'':8s}  "
                        f"{d_ba:+12.4f}  {d_auc:+10.4f}  {'':10s}  {d_rp:+10.4f}"
                    )
            print()

    total_min = (time.time() - t_total) / 60.0
    print(f"\n  Total elapsed: {total_min:.1f} min")
    print(f"  Output dir:    {out_dir}/")

    # ── STEP_6_COMPLETE.flag ──────────────────────────────────────────────────
    with open(os.path.join(out_dir, "STEP_6_COMPLETE.flag"), "w",
              encoding="utf-8") as f:
        f.write(f"STEP_6 completed at {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"seeds={seeds_to_run}\nconditions={CONDITIONS}\n")
        f.write(f"determ_all_ok={determ_all_ok}\ndense_repro_all_ok={dense_all_ok}\n")

    print(f"\n  STEP_6_COMPLETE.flag written.")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
