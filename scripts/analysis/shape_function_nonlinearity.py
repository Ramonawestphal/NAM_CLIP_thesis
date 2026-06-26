"""
Shape function nonlinearity check — both datasets' primary sparsity_conc K=10 models.

For each (dataset, seed, concept, class), fit an OLS line to the empirical shape
function values f_k(x) on the train_final distribution and compute R².

  R² ≈ 1  → concept is nearly linear in this class direction (NAM adds little)
  R² ≪ 1  → concept is genuinely nonlinear (neural shape function justified)

Datasets / conditions:
  HAM10000    : sparsity_concurvity, K=10
                Step-level checkpoints exist — no re-run needed.
  Chest X-ray : sparsity_conc,       K=10
                Dense checkpoint exists; must re-traverse path deterministically
                to the K=10 step (no step-level checkpoints in the sweep).

Primary checkpoints are saved to:
  results/HAM10000/primary_checkpoints/seed_{N}/
  results/chestxray/primary_checkpoints/seed_{N}/

Usage (from project root):
    python scripts/analysis/shape_function_nonlinearity.py
    python scripts/analysis/shape_function_nonlinearity.py --sanity_only

Outputs (results/analysis/nonlinearity/):
    shape_function_r2.csv       — per (dataset, seed, concept, class) R²
    r2_distribution_summary.csv — aggregated stats by dataset
    r2_per_concept.csv          — mean R² per (dataset, concept) across seeds + classes
    r2_histograms.png           — R² distribution by dataset
    r2_per_class_heatmap.png    — concept × class mean R², one subplot per dataset
    summary_report.md
    run_config.json

Constraints:
  - Test set is NOT loaded.  R² is computed on the train_final distribution only.
  - Do NOT modify scripts/HAM10000/, scripts/chestxray/, src/, or any prior artefact.
  - All shared helpers imported from scripts/HAM10000/_common.py unmodified.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts.HAM10000._common import (
    load_raw_data,
    make_fixed_val_split,
    set_all_seeds,
    standardize,
)
from src.models.concurvity import multiclass_concurvity
from src.models.nam_multiclass import NAMMulticlass
from src.models.sparsity import apply_proximal_step, feature_group_norms

# ── Fixed constants ────────────────────────────────────────────────────────────
SEEDS = [42, 43, 44, 45, 46]

# HAM10000
HAM_COND_DIR   = "results/HAM10000/sparsity_sweep/sparsity_concurvity"
HAM_WINNER_JSON = "results/HAM10000/architecture_search_cv/winner.json"
HAM_N_FEATURES  = 24
HAM_N_CLASSES   = 7

# Chest X-ray
CXR_SWEEP_BASE   = "results/chestxray/sparsity_sweep"
CXR_COND         = "sparsity_conc"
CXR_WINNER_JSON  = "results/chestxray/architecture_selection/winning_config.json"
CXR_CONCURVITY_WINNER_JSON = "results/chestxray/concurvity_sweep/winner.json"
CXR_FEATURES_PATH = "data/features/biomedclip/chestxray_concept_scores_v4.npz"
CXR_SPLIT_PATH    = "data/splits/chestxray_outer_split.npz"
CXR_LABEL_MAP_PATH = "results/chestxray/architecture_selection/label_mapping.json"
CXR_N_FEATURES    = 17
CXR_NUM_CLASSES   = 3
CXR_CLASS_NAMES   = ["normal", "bacteria", "virus"]

# Chest X-ray training hyper-params (must match sweep exactly)
CXR_LR              = 1e-3
CXR_BATCH_SIZE      = 256
CXR_VAL_RANDOM_STATE = 42
CXR_ZERO_THRESHOLD  = 1e-6
CXR_MAX_DENSE_EPOCHS = 100
CXR_DENSE_PATIENCE  = 15
CXR_SCHED_PAT       = 5
CXR_SCHED_FAC       = 0.5
CXR_MAX_WARM_EPOCHS = 30
CXR_WARM_PATIENCE   = 6
CXR_WARM_MIN_DELTA  = 1e-4
CXR_LAMBDA_0        = 1.0
CXR_EPSILON         = 0.04
CXR_MAX_LAMBDA      = 1e3

# Concept elimination threshold (must match both sweeps)
ZERO_THRESHOLD = 1e-6

# Primary checkpoints (output caches)
HAM_PRIMARY_BASE = "results/HAM10000/primary_checkpoints"
CXR_PRIMARY_BASE = "results/chestxray/primary_checkpoints"

# Analysis output
OUT_DIR = "results/analysis/nonlinearity"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_active_concepts(
    model: NAMMulticlass,
    concept_names: list,
) -> tuple[np.ndarray, list]:
    """Return (active_concept_indices, active_concept_names) from model weights.

    A concept is active if its feature-group L2 norm exceeds ZERO_THRESHOLD.
    Mirrors the n_active counting in both sweeps.
    """
    norms = feature_group_norms(model)   # dict {name: norm_float}
    group_norms_arr = np.array([norms[name] for name in concept_names])
    active_idx = np.where(group_norms_arr > ZERO_THRESHOLD)[0]
    active_names = [concept_names[i] for i in active_idx]
    return active_idx, active_names


def compute_r2_per_concept_class(
    model: NAMMulticlass,
    X_train_sc: np.ndarray,             # (N, K) float32, scaled train_final features
    active_concept_indices: np.ndarray, # int indices into [0..K-1]
    n_classes: int,
) -> np.ndarray:
    """R²[i, c] = coefficient of determination for active concept i, class c.

    Only iterates over `active_concept_indices` — zeroed-out concepts are
    excluded entirely (they are not emitted to the output CSV at all).

    Fits an OLS line to (x_k, f_k(x_k)) where x_k = X_train_sc[:, k].
    All N train_final samples are used.  Model must already be in eval() mode.

    Returns array of shape (n_active, n_classes).
    """
    n_active = len(active_concept_indices)
    r2 = np.full((n_active, n_classes), fill_value=np.nan)
    model.eval()
    with torch.no_grad():
        for i, k in enumerate(active_concept_indices):
            x_k = X_train_sc[:, k]  # (N,)
            # concept_contributions handles unsqueeze internally → (N, C)
            out = model.concept_contributions(x_k, int(k)).cpu().numpy()  # (N, C)
            for c in range(n_classes):
                y_c = out[:, c]
                if np.std(y_c) < 1e-12:
                    # Effectively constant even though norm > threshold (edge case)
                    r2[i, c] = 1.0
                    continue
                reg = LinearRegression().fit(x_k.reshape(-1, 1), y_c)
                r2[i, c] = reg.score(x_k.reshape(-1, 1), y_c)
    return r2


# ─────────────────────────────────────────────────────────────────────────────
# HAM10000 — load checkpoint directly
# ─────────────────────────────────────────────────────────────────────────────

def find_ham_k10_checkpoint(seed: int) -> tuple[str, float, int, int]:
    """Return (ckpt_path, lambda_s, step, n_active) for K=10 operating point."""
    csv_path = os.path.join(HAM_COND_DIR, f"path_seed{seed}.csv")
    df = pd.read_csv(csv_path, comment="#")
    exact = df[df["n_active"] == 10]
    if len(exact) > 0:
        row = exact.iloc[0]
        fallback = False
    else:
        below = df[df["n_active"] < 10]
        if len(below) == 0:
            raise RuntimeError(f"HAM10000 seed {seed}: no step with n_active ≤ 10")
        row = below.iloc[0]
        fallback = True
    lam     = float(row["lambda_s"])
    step    = int(row["step"])
    n_active = int(row["n_active"])
    ckpt    = os.path.join(
        HAM_COND_DIR, f"seed_{seed}", "checkpoints",
        f"seed{seed}_lambda{lam:.6e}.pt"
    )
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"HAM10000 seed {seed} K=10 checkpoint not found:\n  {ckpt}"
        )
    return ckpt, lam, step, n_active, fallback


def process_ham10000(seeds: list, ham_winner: dict, raw: dict) -> pd.DataFrame:
    """Compute R² for all HAM10000 seeds.  Returns long-format DataFrame."""
    hidden_dims = tuple(ham_winner["hidden_dims"])
    dropout     = float(ham_winner["dropout"])
    # weight_decay not needed for inference

    concept_names = raw["concept_names"]
    class_names   = raw["class_names"]
    train_idx     = raw["train_idx"]
    scores        = raw["scores"]
    labels        = raw["labels"]
    lesion_ids    = raw["lesion_ids"]

    X_pool    = scores[train_idx]
    y_pool    = labels[train_idx]
    g_pool    = lesion_ids[train_idx]

    # Fixed val split (val_random_state=42 — same as all v7 scripts)
    split = make_fixed_val_split(X_pool, y_pool, g_pool, list(class_names))
    X_train_raw = split["X_train"]  # (N_train, 24)

    rows = []
    for seed in seeds:
        print(f"\n  [HAM10000] seed={seed}")
        ckpt_path, lam, step, n_active, fallback = find_ham_k10_checkpoint(seed)
        tag = "(fallback)" if fallback else "(exact)"
        print(f"    K=10 → step={step}, n_active={n_active} {tag}, λ={lam:.4f}")

        # Load scaler
        scaler_path = os.path.join(HAM_COND_DIR, f"seed_{seed}", "scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_train_sc = scaler.transform(X_train_raw).astype(np.float32)

        # Build model + load checkpoint
        model = NAMMulticlass(
            n_features=HAM_N_FEATURES,
            num_classes=HAM_N_CLASSES,
            hidden_dims=hidden_dims,
            dropout=dropout,
            concept_names=list(concept_names),
        ).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Fix 1: filter to active concepts only
        active_idx, active_names = get_active_concepts(model, list(concept_names))

        # Fix 3: sanity check active count vs path.csv n_active
        # HAM10000 loads a saved checkpoint directly — exact match required.
        print(f"    Active concepts: {len(active_idx)} (expected: {n_active})")
        if len(active_idx) != n_active:
            raise RuntimeError(
                f"HAM10000 seed {seed}: active concept count mismatch — "
                f"model weights give {len(active_idx)}, path.csv says {n_active}.  "
                f"(HAM10000 loads checkpoints directly; mismatch indicates a wrong "
                f"checkpoint or path.csv inconsistency.)"
            )

        # Save to primary_checkpoints (as reference copy)
        prim_dir = os.path.join(HAM_PRIMARY_BASE, f"seed_{seed}")
        os.makedirs(prim_dir, exist_ok=True)
        prim_model_pt   = os.path.join(prim_dir, "model.pt")
        prim_scaler_pkl = os.path.join(prim_dir, "scaler.pkl")
        prim_meta_json  = os.path.join(prim_dir, "meta.json")
        if not os.path.exists(prim_model_pt):
            torch.save(state, prim_model_pt)
        if not os.path.exists(prim_scaler_pkl):
            with open(prim_scaler_pkl, "wb") as fsc:
                pickle.dump(scaler, fsc)
        # Fix 2: always write meta (overwrites stale version if surviving_concepts missing)
        with open(prim_meta_json, "w", encoding="utf-8") as fm:
            json.dump({
                "dataset": "ham10000",
                "condition": "sparsity_concurvity",
                "K": 10,
                "seed": seed,
                "step": step,
                "lambda_s": lam,
                "n_active": n_active,
                "fallback": fallback,
                "source_ckpt": ckpt_path,
                "surviving_concepts": active_names,
                "surviving_concept_indices": active_idx.tolist(),
            }, fm, indent=2)
        print(f"    primary_checkpoints saved to {prim_dir}")

        # Fix 1 + Fix 4: compute R² over active concepts only
        r2 = compute_r2_per_concept_class(model, X_train_sc, active_idx, HAM_N_CLASSES)
        print(f"    R² computed: shape={r2.shape}, mean={r2.mean():.4f}, "
              f"min={r2.min():.4f}, max={r2.max():.4f}  "
              f"({len(active_idx)} active × {HAM_N_CLASSES} classes = "
              f"{len(active_idx) * HAM_N_CLASSES} shape functions)")

        for i, (k_idx, cname) in enumerate(zip(active_idx, active_names)):
            for c, cls in enumerate(class_names):
                rows.append({
                    "dataset":      "ham10000",
                    "seed":         seed,
                    "concept":      cname,
                    "concept_idx":  int(k_idx),
                    "class":        cls,
                    "class_idx":    c,
                    "r2":           float(r2[i, c]),
                    "step":         step,
                    "n_active_at_step": n_active,
                    "lambda_s":     lam,
                    "fallback":     fallback,
                })

        del model

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Chest X-ray — load data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cxr_label_mapping() -> dict:
    if os.path.exists(CXR_LABEL_MAP_PATH):
        with open(CXR_LABEL_MAP_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"normal": 0, "bacteria": 1, "virus": 2}


def load_cxr_data(subtype_to_int: dict) -> dict:
    """Load chest X-ray features and split.  Test set NOT loaded."""
    feat          = np.load(CXR_FEATURES_PATH, allow_pickle=True)
    scores        = feat["scores"]
    concept_names = feat["concept_names"].tolist()

    split          = np.load(CXR_SPLIT_PATH, allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]
    labels_subtype = split["labels_subtype"]
    patient_ids    = split["patient_ids"]

    labels_all = np.array(
        [subtype_to_int[s] for s in labels_subtype], dtype=np.int64
    )
    return {
        "scores":         scores,
        "concept_names":  concept_names,
        "labels_all":     labels_all,
        "train_pool_idx": train_pool_idx,
        "patient_ids":    patient_ids,
    }


def find_cxr_k10_step(seed: int) -> tuple[int, float, int, bool]:
    """Return (step, lambda_s, n_active, fallback) for chest X-ray K=10."""
    csv_path = os.path.join(CXR_SWEEP_BASE, CXR_COND, f"seed_{seed}", "path.csv")
    df = pd.read_csv(csv_path)
    exact = df[df["n_active"] == 10]
    if len(exact) > 0:
        row = exact.iloc[0]
        return int(row["step"]), float(row["lambda_s"]), int(row["n_active"]), False
    below = df[df["n_active"] < 10]
    if len(below) == 0:
        raise RuntimeError(
            f"Chest X-ray seed {seed}: no step with n_active ≤ 10 in path.csv"
        )
    row = below.iloc[0]
    return int(row["step"]), float(row["lambda_s"]), int(row["n_active"]), True


# ─────────────────────────────────────────────────────────────────────────────
# Chest X-ray — deterministic re-traversal to K=10 step
# ─────────────────────────────────────────────────────────────────────────────

def _retraverse_to_step(
    *,
    seed:              int,
    concurvity_lambda: float,
    hidden_dims:       tuple,
    dropout:           float,
    weight_decay:      float,
    concept_names:     list,
    X_train_final_raw: np.ndarray,
    X_val_raw:         np.ndarray,
    y_train_final:     np.ndarray,  # int64
    y_val:             np.ndarray,  # int64
    target_step:       int,
) -> tuple[NAMMulticlass, StandardScaler]:
    """Re-run dense training + warm-start path from scratch up to target_step.

    Mirrors run_one_seed() from run_sparsity_sweep.py exactly — same
    set_all_seeds(seed) call, same scaler fit, same class weights, same
    architecture, same hyperparameters.  Returns (model_at_target_step, scaler).

    Dense training is re-run from scratch (not loaded from .pt) so that the
    PyTorch RNG state entering the warm-start phase is identical to the sweep's.
    """
    from sklearn.metrics import balanced_accuracy_score as _balacc

    pin_memory = (DEVICE.type == "cuda")

    set_all_seeds(seed)

    # Scaler
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train_final_raw).astype(np.float32)
    X_val_sc = scaler.transform(X_val_raw).astype(np.float32)

    # Class weights
    counts    = np.bincount(y_train_final, minlength=CXR_NUM_CLASSES)
    n_tr      = len(y_train_final)
    weights   = n_tr / (CXR_NUM_CLASSES * counts.astype(np.float64))
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
            train_ds, batch_size=CXR_BATCH_SIZE, shuffle=shuffle,
            pin_memory=pin_memory
        )

    model = NAMMulticlass(
        n_features=CXR_N_FEATURES,
        num_classes=CXR_NUM_CLASSES,
        hidden_dims=hidden_dims,
        dropout=dropout,
        concept_names=concept_names,
    ).to(DEVICE)

    def _eval_val_full() -> tuple:
        """Returns (balacc, r_perp, val_loss_ce, val_loss_full).
        Mirrors _eval_val_full() in run_sparsity_sweep.py exactly."""
        model.eval()
        with torch.no_grad():
            logits, shape_outs = model(X_val_t, return_shape_outputs=True)
            val_loss_ce = criterion(logits, y_val_t).item()
            preds       = logits.argmax(dim=1).cpu().numpy()
            r_perp      = multiclass_concurvity(shape_outs).item()
        balacc = _balacc(y_val, preds)
        val_loss_full = val_loss_ce + concurvity_lambda * r_perp
        return float(balacc), float(r_perp), float(val_loss_ce), float(val_loss_full)

    # ── Dense phase (mirrors sweep exactly) ───────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CXR_LR, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=CXR_SCHED_FAC,
        patience=CXR_SCHED_PAT, min_lr=1e-6
    )
    best_val_balacc  = -1.0
    patience_ctr     = 0
    best_dense_state = None

    for epoch in range(CXR_MAX_DENSE_EPOCHS):
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

        val_balacc, _, _, _ = _eval_val_full()
        scheduler.step(val_balacc)
        if val_balacc > best_val_balacc + 1e-4:
            best_val_balacc  = val_balacc
            patience_ctr     = 0
            best_dense_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= CXR_DENSE_PATIENCE:
                break

    if best_dense_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_dense_state.items()})

    dense_balacc, _, _, _ = _eval_val_full()
    print(f"    [dense] done, val_balacc={dense_balacc:.4f}")

    # ── Warm-start path to target_step ────────────────────────────────────────
    # Build lambda schedule exactly as the sweep does
    lambda_schedule: list[float] = []
    t = 0
    while len(lambda_schedule) < target_step:
        lam = CXR_LAMBDA_0 * (1.0 + CXR_EPSILON) ** t
        if lam > CXR_MAX_LAMBDA:
            break
        lambda_schedule.append(lam)
        t += 1

    # prev_state_dict starts as the best dense state (Issue 4 fix)
    prev_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for step_idx, lambda_t in enumerate(lambda_schedule):
        # Issue 4 fix: warm-start from BEST previous-step state
        model.load_state_dict({k: v.to(DEVICE) for k, v in prev_state_dict.items()})

        # Fresh optimizer + scheduler per step
        optimizer_w = torch.optim.Adam(
            model.parameters(), lr=CXR_LR, weight_decay=weight_decay
        )
        scheduler_w = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_w, mode="min", factor=CXR_SCHED_FAC,
            patience=CXR_SCHED_PAT, min_lr=1e-6
        )

        best_step_loss  = float("inf")
        best_step_state = None
        no_improve_ctr  = 0

        for epoch in range(CXR_MAX_WARM_EPOCHS):
            model.train()
            for X_b, y_b in _make_loader(shuffle=True):
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer_w.zero_grad()
                if concurvity_lambda > 0:
                    logits, shape_outs = model(X_b, return_shape_outputs=True)
                    loss = (criterion(logits, y_b)
                            + concurvity_lambda * multiclass_concurvity(shape_outs))
                else:
                    loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer_w.step()
                # Proximal block soft-thresholding after EACH batch (mirrors sweep)
                apply_proximal_step(
                    model,
                    lr=optimizer_w.param_groups[0]["lr"],
                    sparsity_lambda=lambda_t,
                )

            _, _, _, val_loss_full = _eval_val_full()
            scheduler_w.step(val_loss_full)

            if val_loss_full < best_step_loss - CXR_WARM_MIN_DELTA:
                best_step_loss  = val_loss_full
                best_step_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve_ctr  = 0
            else:
                no_improve_ctr += 1
                if no_improve_ctr >= CXR_WARM_PATIENCE:
                    break

        # Issue 4 fix: restore best-within-step before advancing
        if best_step_state is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_step_state.items()})

        # Carry best state to next step
        prev_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Early exit if all zeroed
        norms = feature_group_norms(model)
        if all(v < CXR_ZERO_THRESHOLD for v in norms.values()):
            print(f"    [path] All concepts zeroed at step {step_idx + 1} — stopping early.")
            break

    model.eval()
    return model, scaler


def process_cxr(seeds: list, cxr_winner: dict, lam_c: float, data: dict,
                val_split: dict) -> pd.DataFrame:
    """Compute R² for all chest X-ray seeds.  Returns long-format DataFrame."""
    hidden_dims  = tuple(cxr_winner["hidden_dims"])
    dropout      = float(cxr_winner["dropout"])
    weight_decay = float(cxr_winner["weight_decay"])
    concept_names = data["concept_names"]

    X_pool         = data["scores"][data["train_pool_idx"]]
    y_pool_int     = data["labels_all"][data["train_pool_idx"]]
    patient_ids_pool = data["patient_ids"][data["train_pool_idx"]]

    X_train_raw  = X_pool[val_split["train_rel"]]
    X_val_raw    = X_pool[val_split["val_rel"]]
    y_train_final = y_pool_int[val_split["train_rel"]]
    y_val         = y_pool_int[val_split["val_rel"]]

    rows = []
    for seed in seeds:
        print(f"\n  [CXR] seed={seed}")

        # K=10 operating point
        k10_step, lambda_s, n_active, fallback = find_cxr_k10_step(seed)
        tag = "(fallback)" if fallback else "(exact)"
        print(f"    K=10 → step={k10_step}, n_active={n_active} {tag}, λ={lambda_s:.4f}")

        # Check primary_checkpoints cache
        prim_dir     = os.path.join(CXR_PRIMARY_BASE, f"seed_{seed}")
        prim_model   = os.path.join(prim_dir, "model.pt")
        prim_scaler  = os.path.join(prim_dir, "scaler.pkl")
        prim_meta    = os.path.join(prim_dir, "meta.json")
        cache_valid  = all(os.path.exists(p) for p in [prim_model, prim_scaler, prim_meta])

        if cache_valid:
            print(f"    Loading from cache: {prim_dir}")
            state = torch.load(prim_model, map_location=DEVICE, weights_only=True)
            model = NAMMulticlass(
                n_features=CXR_N_FEATURES,
                num_classes=CXR_NUM_CLASSES,
                hidden_dims=hidden_dims,
                dropout=dropout,
                concept_names=list(concept_names),
            ).to(DEVICE)
            model.load_state_dict(state)
            model.eval()
            with open(prim_scaler, "rb") as f:
                scaler = pickle.load(f)
        else:
            print(f"    Re-traversing path to step {k10_step} …")
            t0 = time.time()
            model, scaler = _retraverse_to_step(
                seed=seed,
                concurvity_lambda=lam_c,
                hidden_dims=hidden_dims,
                dropout=dropout,
                weight_decay=weight_decay,
                concept_names=list(concept_names),
                X_train_final_raw=X_train_raw,
                X_val_raw=X_val_raw,
                y_train_final=y_train_final,
                y_val=y_val,
                target_step=k10_step,
            )
            elapsed = time.time() - t0
            print(f"    Re-traversal done in {elapsed:.1f}s")

            # Save to primary_checkpoints
            os.makedirs(prim_dir, exist_ok=True)
            torch.save(model.state_dict(), prim_model)
            with open(prim_scaler, "wb") as f:
                pickle.dump(scaler, f)
            print(f"    Saved to {prim_dir}")

        X_train_sc = scaler.transform(X_train_raw).astype(np.float32)

        # Fix 1: filter to active concepts only
        active_idx, active_names = get_active_concepts(model, list(concept_names))

        # Fix 3: sanity check active count vs path.csv n_active.
        # CXR re-traverses from scratch; ±1 borderline tolerated with a warning.
        delta = abs(len(active_idx) - n_active)
        print(f"    Active concepts: {len(active_idx)} (expected: {n_active})")
        if delta > 1:
            raise RuntimeError(
                f"CXR seed {seed}: active concept count mismatch by {delta} — "
                f"model weights give {len(active_idx)}, path.csv says {n_active}.  "
                f"Re-traversal has diverged beyond floating-point tolerance."
            )
        if delta == 1:
            # Small floating-point drift over 30+ warm-start steps; ±1 tolerated.
            # Report the divergent concepts and proceed — model's actual active
            # set is authoritative for the R² computation.
            sweep_active_set = set()
            norms_csv = os.path.join(
                CXR_SWEEP_BASE, CXR_COND, f"seed_{seed}",
                "feature_group_norms_per_step.csv"
            )
            if os.path.exists(norms_csv):
                norms_df = pd.read_csv(norms_csv)
                row = norms_df[norms_df["step"] == k10_step]
                if len(row) > 0:
                    norm_cols = [c for c in norms_df.columns if c.startswith("norm_")]
                    sweep_active_set = {
                        c.replace("norm_", "") for c in norm_cols
                        if float(row.iloc[0][c]) > ZERO_THRESHOLD
                    }
            retrv_active_set = set(active_names)
            missing = sweep_active_set - retrv_active_set
            extra   = retrv_active_set - sweep_active_set
            print(f"    WARNING: ±1 re-traversal drift (floating-point over "
                  f"{k10_step} warm-start steps).")
            if missing:
                print(f"      In sweep but not re-traversal: {sorted(missing)}")
            if extra:
                print(f"      In re-traversal but not sweep: {sorted(extra)}")
            print(f"    Proceeding with model's actual active set "
                  f"({len(active_idx)} concepts).")

        # Fix 2: always write meta.json with surviving_concepts (overwrites stale version)
        # n_active_path_csv = value from path.csv; n_active = model's actual count
        os.makedirs(prim_dir, exist_ok=True)
        with open(prim_meta, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": "chestxray",
                "condition": CXR_COND,
                "K": 10,
                "seed": seed,
                "step": k10_step,
                "lambda_s": lambda_s,
                "n_active_path_csv": n_active,           # from path.csv
                "n_active": len(active_idx),              # model's actual active count
                "fallback": fallback,
                "concurvity_lambda": lam_c,
                "surviving_concepts": active_names,
                "surviving_concept_indices": active_idx.tolist(),
            }, f, indent=2)

        # Fix 1 + Fix 4: compute R² over active concepts only
        r2 = compute_r2_per_concept_class(
            model, X_train_sc, active_idx, CXR_NUM_CLASSES
        )
        print(f"    R² computed: shape={r2.shape}, mean={r2.mean():.4f}, "
              f"min={r2.min():.4f}, max={r2.max():.4f}  "
              f"({len(active_idx)} active × {CXR_NUM_CLASSES} classes = "
              f"{len(active_idx) * CXR_NUM_CLASSES} shape functions)")

        for i, (k_idx, cname) in enumerate(zip(active_idx, active_names)):
            for c, cls in enumerate(CXR_CLASS_NAMES):
                rows.append({
                    "dataset":          "chestxray",
                    "seed":             seed,
                    "concept":          cname,
                    "concept_idx":      int(k_idx),
                    "class":            cls,
                    "class_idx":        c,
                    "r2":               float(r2[i, c]),
                    "step":             k10_step,
                    "n_active_at_step": n_active,
                    "lambda_s":         lambda_s,
                    "fallback":         fallback,
                })

        del model

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis and plots
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(distribution_summary, per_concept) DataFrames."""
    dist_rows = []
    for dataset, grp in df.groupby("dataset"):
        r2_vals = grp["r2"].values
        dist_rows.append({
            "dataset":      dataset,
            "n_entries":    len(r2_vals),
            "mean_r2":      float(np.mean(r2_vals)),
            "median_r2":    float(np.median(r2_vals)),
            "std_r2":       float(np.std(r2_vals, ddof=1)),
            "pct10_r2":     float(np.percentile(r2_vals, 10)),
            "pct25_r2":     float(np.percentile(r2_vals, 25)),
            "pct75_r2":     float(np.percentile(r2_vals, 75)),
            "pct90_r2":     float(np.percentile(r2_vals, 90)),
            "frac_r2_gt09": float(np.mean(r2_vals > 0.9)),
            "frac_r2_lt05": float(np.mean(r2_vals < 0.5)),
        })
    dist_df = pd.DataFrame(dist_rows)

    concept_rows = []
    for (dataset, concept), grp in df.groupby(["dataset", "concept"]):
        concept_rows.append({
            "dataset":   dataset,
            "concept":   concept,
            "mean_r2":   float(grp["r2"].mean()),
            "std_r2":    float(grp["r2"].std(ddof=1)),
            "min_r2":    float(grp["r2"].min()),
            "max_r2":    float(grp["r2"].max()),
            "n_entries": len(grp),
        })
    concept_df = pd.DataFrame(concept_rows).sort_values(["dataset", "mean_r2"])
    return dist_df, concept_df


def make_plots(df: pd.DataFrame, out_dir: str) -> None:
    """R² histogram and per-class heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    datasets = df["dataset"].unique()

    # ── Histogram of all R² values by dataset ────────────────────────────────
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        r2_vals = df[df["dataset"] == ds]["r2"].values
        ax.hist(r2_vals, bins=30, edgecolor="black", alpha=0.8)
        ax.axvline(np.median(r2_vals), color="red", linestyle="--",
                   label=f"median={np.median(r2_vals):.2f}")
        ax.set_xlabel("R² (linear fit to shape function)")
        ax.set_ylabel("Count")
        ax.set_title(f"{ds} — all (concept × class × seed)")
        ax.legend()
    fig.tight_layout()
    hist_path = os.path.join(out_dir, "r2_histograms.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {hist_path}")

    # ── Per-class heatmap: concept × class, mean R² across seeds ─────────────
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        pivot = sub.groupby(["concept", "class"])["r2"].mean().unstack(fill_value=np.nan)
        # Sort concepts by mean R² (most nonlinear first)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

        fig2, ax2 = plt.subplots(figsize=(max(4, len(pivot.columns) * 1.5),
                                          max(5, len(pivot) * 0.35)))
        im = ax2.imshow(pivot.values, vmin=0.0, vmax=1.0, aspect="auto",
                        cmap="RdYlGn")
        ax2.set_xticks(range(len(pivot.columns)))
        ax2.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
        ax2.set_yticks(range(len(pivot.index)))
        ax2.set_yticklabels(pivot.index, fontsize=8)
        ax2.set_title(f"{ds} — mean R² per concept × class (seeds {SEEDS})")
        fig2.colorbar(im, ax=ax2, label="mean R²")
        fig2.tight_layout()
        hmap_path = os.path.join(out_dir, f"r2_per_class_heatmap_{ds}.png")
        fig2.savefig(hmap_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {hmap_path}")


def _df_to_md(df: pd.DataFrame, float_cols: list | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table without tabulate."""
    cols = df.columns.tolist()
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if float_cols and c in float_cols and isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append(cells)

    # Column widths
    widths = [max(len(c), max((len(r[i]) for r in rows), default=0))
              for i, c in enumerate(cols)]
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)) + " |"
    body = "\n".join(
        "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cols))) + " |"
        for cells in rows
    )
    return "\n".join([head, sep, body])


def write_summary_report(df: pd.DataFrame, dist_df: pd.DataFrame,
                         concept_df: pd.DataFrame, out_dir: str) -> None:
    float_dist    = ["mean_r2", "median_r2", "std_r2", "pct10_r2", "pct25_r2",
                     "pct75_r2", "pct90_r2", "frac_r2_gt09", "frac_r2_lt05"]
    float_concept = ["mean_r2", "std_r2", "min_r2", "max_r2"]

    lines = [
        "# Shape Function Nonlinearity Report",
        "",
        "Condition: sparsity_conc, K=10 operating point, seeds 42-46.",
        "R^2 = coefficient of determination for OLS fit to (x_k, f_k(x_k)) on train_final.",
        "R^2 ~= 1 -> near-linear; R^2 << 1 -> genuinely nonlinear.",
        "",
        "## Distribution Summary",
        "",
        _df_to_md(dist_df, float_cols=float_dist),
        "",
        "## Least-Linear Concepts (bottom-10 mean R^2, each dataset)",
        "",
    ]
    for ds in df["dataset"].unique():
        sub_concept = concept_df[concept_df["dataset"] == ds].head(10)
        lines += [
            f"### {ds}",
            "",
            _df_to_md(
                sub_concept[["concept", "mean_r2", "std_r2", "min_r2", "max_r2"]].reset_index(drop=True),
                float_cols=float_concept,
            ),
            "",
        ]
    lines += [
        "## Most-Linear Concepts (top-10 mean R^2, each dataset)",
        "",
    ]
    for ds in df["dataset"].unique():
        sub_concept = concept_df[concept_df["dataset"] == ds].tail(10).iloc[::-1]
        lines += [
            f"### {ds}",
            "",
            _df_to_md(
                sub_concept[["concept", "mean_r2", "std_r2", "min_r2", "max_r2"]].reset_index(drop=True),
                float_cols=float_concept,
            ),
            "",
        ]
    report = "\n".join(lines)
    rpath  = os.path.join(out_dir, "summary_report.md")
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {rpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-run sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks() -> bool:
    """Return True if all hard checks pass; print warnings for soft issues."""
    print("\n" + "=" * 70)
    print("PRE-RUN SANITY CHECKS — shape_function_nonlinearity.py")
    print("=" * 70)
    all_ok = True

    # ── [1] HAM10000 winner.json ──────────────────────────────────────────────
    if not os.path.exists(HAM_WINNER_JSON):
        print(f"  [1] FAIL: {HAM_WINNER_JSON} not found.")
        return False
    with open(HAM_WINNER_JSON, encoding="utf-8") as f:
        ham_winner = json.load(f)
    print(f"  [1] HAM10000 winner: hidden_dims={ham_winner['hidden_dims']}, "
          f"dropout={ham_winner['dropout']}, weight_decay={ham_winner['weight_decay']}, "
          f"config_id={ham_winner['config_id']}  ✓")

    # ── [2] HAM10000 K=10 checkpoints and scalers ────────────────────────────
    print(f"\n  [2] HAM10000 K=10 checkpoints and scalers:")
    for seed in SEEDS:
        try:
            ckpt_path, lam, step, n_active, fallback = find_ham_k10_checkpoint(seed)
            scaler_path = os.path.join(HAM_COND_DIR, f"seed_{seed}", "scaler.pkl")
            scaler_ok   = os.path.exists(scaler_path)
            tag = " (fallback)" if fallback else " (exact  )"
            scaler_tag = "✓" if scaler_ok else "MISSING"
            print(f"      seed{seed}: step={step:3d}, n_active={n_active:2d}{tag}, "
                  f"λ={lam:.4f}, ckpt=✓, scaler={scaler_tag}")
            if not scaler_ok:
                all_ok = False
        except (FileNotFoundError, RuntimeError) as e:
            print(f"      seed{seed}: FAIL — {e}")
            all_ok = False
    if all_ok:
        print(f"      All 5 HAM10000 checkpoints + scalers present  ✓")

    # ── [3] Chest X-ray winner.json ──────────────────────────────────────────
    print(f"\n  [3] Chest X-ray winner.json:")
    if not os.path.exists(CXR_WINNER_JSON):
        print(f"      FAIL: {CXR_WINNER_JSON} not found.")
        all_ok = False
    else:
        with open(CXR_WINNER_JSON, encoding="utf-8") as f:
            cxr_winner = json.load(f)
        print(f"      hidden_dims={cxr_winner.get('hidden_dims')}, "
              f"dropout={cxr_winner.get('dropout')}, "
              f"weight_decay={cxr_winner.get('weight_decay')}  ✓")

    if not os.path.exists(CXR_CONCURVITY_WINNER_JSON):
        print(f"      FAIL: {CXR_CONCURVITY_WINNER_JSON} not found.")
        all_ok = False
    else:
        with open(CXR_CONCURVITY_WINNER_JSON, encoding="utf-8") as f:
            cw = json.load(f)
        if cw.get("selection_pending", True):
            print(f"      FAIL: selection_pending=True in concurvity winner.json.")
            all_ok = False
        else:
            lam_c = float(cw["operative_lambda_c"])
            print(f"      concurvity winner: operative_lambda_c={lam_c}  ✓")

    # ── [4] Chest X-ray sweep artefacts ──────────────────────────────────────
    print(f"\n  [4] Chest X-ray sweep artefacts (STEP 5 must have completed):")
    cxr_sweep_ok = True
    for seed in SEEDS:
        seed_dir = os.path.join(CXR_SWEEP_BASE, CXR_COND, f"seed_{seed}")
        path_csv = os.path.join(seed_dir, "path.csv")
        dense_pt = os.path.join(seed_dir, f"dense_seed{seed}_conc3p0.pt")
        p_ok = os.path.exists(path_csv)
        d_ok = os.path.exists(dense_pt)
        if p_ok and d_ok:
            try:
                step, lam_s, n_act, fb = find_cxr_k10_step(seed)
                tag = "(fallback)" if fb else "(exact  )"
                print(f"      seed{seed}: K=10 → step={step:3d}, n_active={n_act:2d} {tag}, "
                      f"λ={lam_s:.4f}  ✓")
            except RuntimeError as e:
                print(f"      seed{seed}: path.csv exists but K=10 not reachable — {e}")
                cxr_sweep_ok = False
        else:
            missing = []
            if not p_ok: missing.append("path.csv")
            if not d_ok: missing.append(f"dense_seed{seed}_conc3p0.pt")
            print(f"      seed{seed}: MISSING {', '.join(missing)}")
            cxr_sweep_ok = False
    if not cxr_sweep_ok:
        print(f"\n      ⚠ Some chest X-ray sweep artefacts missing.")
        print(f"        Run Task 2 (run_sparsity_sweep.py) before proceeding.")
        all_ok = False
    else:
        print(f"      All 5 chest X-ray sweep artefacts present  ✓")

    # ── [5] Data files accessible ─────────────────────────────────────────────
    print(f"\n  [5] Data files:")
    for fpath in [CXR_FEATURES_PATH, CXR_SPLIT_PATH,
                  "data/features/biomedclip/ham10000_concept_scores_v6.npz"]:
        ok = os.path.exists(fpath)
        print(f"      {'✓' if ok else 'MISSING'}: {fpath}")
        if not ok:
            all_ok = False

    # ── [6] Output directory writeable ────────────────────────────────────────
    print(f"\n  [6] Output directory:")
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(HAM_PRIMARY_BASE, exist_ok=True)
        os.makedirs(CXR_PRIMARY_BASE, exist_ok=True)
        print(f"      {OUT_DIR}  ✓")
        print(f"      {HAM_PRIMARY_BASE}  ✓")
        print(f"      {CXR_PRIMARY_BASE}  ✓")
    except OSError as e:
        print(f"      FAIL: cannot create output dirs — {e}")
        all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("All sanity checks passed.\n")
    else:
        print("One or more sanity checks FAILED.  Resolve before running.\n")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shape function nonlinearity check — HAM10000 + chest X-ray."
    )
    parser.add_argument("--sanity_only", action="store_true",
                        help="Run sanity checks and exit without computing R².")
    parser.add_argument("--ham_only", action="store_true",
                        help="Skip chest X-ray (useful while STEP 5 sweep is pending).")
    parser.add_argument("--report_only", action="store_true",
                        help="Regenerate summary_report.md from existing CSVs; skip all training.")
    args = parser.parse_args()

    # ── Report-only shortcut (re-generate report from saved CSVs) ────────────
    if args.report_only:
        r2_path = os.path.join(OUT_DIR, "shape_function_r2.csv")
        if not os.path.exists(r2_path):
            print(f"ERROR: {r2_path} not found.  Run without --report_only first.")
            sys.exit(1)
        df = pd.read_csv(r2_path)
        dist_df, concept_df = build_summary_tables(df)
        write_summary_report(df, dist_df, concept_df, OUT_DIR)
        sys.exit(0)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    checks_ok = run_sanity_checks()
    if args.sanity_only:
        sys.exit(0 if checks_ok else 1)
    if not checks_ok:
        print("Aborting: sanity checks failed.")
        sys.exit(1)

    # ── Load arch winners ─────────────────────────────────────────────────────
    with open(HAM_WINNER_JSON, encoding="utf-8") as f:
        ham_winner = json.load(f)
    with open(CXR_WINNER_JSON, encoding="utf-8") as f:
        cxr_winner = json.load(f)
    with open(CXR_CONCURVITY_WINNER_JSON, encoding="utf-8") as f:
        cxr_conc_w = json.load(f)
    lam_c = float(cxr_conc_w["operative_lambda_c"])

    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows: list[pd.DataFrame] = []

    # ── HAM10000 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HAM10000 — sparsity_concurvity, K=10")
    print("=" * 60)
    raw = load_raw_data()
    ham_df = process_ham10000(SEEDS, ham_winner, raw)
    all_rows.append(ham_df)

    # ── Chest X-ray ───────────────────────────────────────────────────────────
    if not args.ham_only:
        print("\n" + "=" * 60)
        print("Chest X-ray — sparsity_conc, K=10")
        print("=" * 60)
        subtype_to_int = load_cxr_label_mapping()
        cxr_data = load_cxr_data(subtype_to_int)

        # Fixed val split — must match the sweep (val_random_state=42)
        X_pool     = cxr_data["scores"][cxr_data["train_pool_idx"]]
        y_pool_int = cxr_data["labels_all"][cxr_data["train_pool_idx"]]
        pids_pool  = cxr_data["patient_ids"][cxr_data["train_pool_idx"]]
        # make_fixed_val_split expects string labels; chest X-ray uses int labels
        # so encode as strings, pass class_names=["0","1","2"] to match sweep
        y_pool_str = y_pool_int.astype(str)
        val_split  = make_fixed_val_split(
            X_pool, y_pool_str, pids_pool, ["0", "1", "2"],
            val_random_state=CXR_VAL_RANDOM_STATE,
        )
        cxr_df = process_cxr(SEEDS, cxr_winner, lam_c, cxr_data, val_split)
        all_rows.append(cxr_df)

    # ── Combine and write outputs ─────────────────────────────────────────────
    df = pd.concat(all_rows, ignore_index=True)

    r2_path = os.path.join(OUT_DIR, "shape_function_r2.csv")
    df.to_csv(r2_path, index=False)
    print(f"\nSaved: {r2_path}  ({len(df)} rows)")

    dist_df, concept_df = build_summary_tables(df)

    dist_path = os.path.join(OUT_DIR, "r2_distribution_summary.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"Saved: {dist_path}")

    concept_path = os.path.join(OUT_DIR, "r2_per_concept.csv")
    concept_df.to_csv(concept_path, index=False)
    print(f"Saved: {concept_path}")

    make_plots(df, OUT_DIR)
    write_summary_report(df, dist_df, concept_df, OUT_DIR)

    # ── run_config.json ───────────────────────────────────────────────────────
    cfg = {
        "script":       "scripts/analysis/shape_function_nonlinearity.py",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "seeds":        SEEDS,
        "condition":    "sparsity_conc",
        "K_budget":     10,
        "datasets":     list(df["dataset"].unique()),
        "ham_winner":   {k: ham_winner[k] for k in ["hidden_dims", "dropout", "weight_decay", "config_id"]},
        "cxr_winner":   {k: cxr_winner.get(k) for k in ["hidden_dims", "dropout", "weight_decay"]},
        "cxr_lam_c":    lam_c,
        "r2_note": (
            "R² of OLS best-fit line to (x_k, f_k(x_k)) on train_final distribution. "
            "Constant outputs (zeroed concepts) assigned R²=1.0."
        ),
    }
    cfg_path = os.path.join(OUT_DIR, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved: {cfg_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NONLINEARITY SUMMARY")
    print("=" * 60)
    for _, row in dist_df.iterrows():
        print(f"  {row['dataset']:12s}: mean R²={row['mean_r2']:.4f}  "
              f"median={row['median_r2']:.4f}  "
              f"frac>0.9={row['frac_r2_gt09']:.2%}  "
              f"frac<0.5={row['frac_r2_lt05']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
