"""
Sparsity sweep SMOKE TEST — chest X-ray three-way task.

Mirrors scripts/v7/run_sparsity_sweep.py for the chest X-ray dataset but:
  - Single seed (42) only
  - 100 lambda steps (not 150)
  - Both conditions run sequentially: sparsity_only → sparsity_conc
  - Output to results/chestxray/sparsity_sweep_smoke/ (separate from full sweep)
  - Test set is NEVER loaded or touched

Purpose: confirm that λ_0=1.0, ε=0.04 is in the sparsity-active zone for
chest X-ray before committing to the full 5-seed × 2-condition × 150-step run.

Pass criteria (per condition):
  (a) First elimination ≤ step 50
  (b) n_active ≤ 10 at step 100

Audit fixes mirrored from v7
────────────────────────────
  Issue 4: best-within-step checkpoint restored before advancing to λ+1
  Issue 9: warmup_epochs=0 (concurvity active from epoch 1 when λ_c > 0)
  Issue 3: set_all_seeds() includes random.seed
  Issue 7: per-seed scaler saved
  Issue 8: CUDA determinism flags set

Usage (from project root):
    python scripts/chestxray/run_sparsity_sweep_smoke.py
    python scripts/chestxray/run_sparsity_sweep_smoke.py --sanity_only
    python scripts/chestxray/run_sparsity_sweep_smoke.py --dry_run   # 3 steps each
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts.v7._common import (
    set_all_seeds,
    make_fixed_val_split,
    make_optimizer_scheduler,
)
from src.models.nam_multiclass import NAMMulticlass
from src.models.concurvity import multiclass_concurvity
from src.models.sparsity import feature_group_norms, apply_proximal_step

# ── Constants ─────────────────────────────────────────────────────────────────
SEED        = 42
LR          = 1e-3
BATCH_SIZE  = 256
N_FEATURES  = 17
NUM_CLASSES = 3
VAL_RANDOM_STATE = 42
ZERO_THRESHOLD   = 1e-6   # concept considered eliminated when norm < this

# Dense phase
MAX_DENSE_EPOCHS = 100
DENSE_PATIENCE   = 15
SCHED_PAT        = 5
SCHED_FAC        = 0.5

# Warm-start phase
MAX_WARM_EPOCHS = 30
WARM_PATIENCE   = 6
WARM_MIN_DELTA  = 1e-4

# Lambda schedule (v3 from v7)
LAMBDA_0      = 1.0
EPSILON       = 0.04
MAX_LAMBDA    = 1e3
MAX_LAM_STEPS = 100   # smoke test: 100 (full sweep uses 150)

# Paths
FEATURES_PATH          = "data/features/biomedclip/chestxray_concept_scores_v4.npz"
SPLIT_PATH             = "data/splits/chestxray_outer_split.npz"
LABEL_MAP_PATH         = "results/chestxray/architecture_selection/label_mapping.json"
ARCH_WINNER_JSON       = "results/chestxray/architecture_selection/winning_config.json"
CONCURVITY_WINNER_JSON = "results/chestxray/concurvity_sweep/winner.json"
OUT_ROOT               = "results/chestxray/sparsity_sweep_smoke"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (mirrors train_final.py conventions exactly)
# ─────────────────────────────────────────────────────────────────────────────

def load_label_mapping() -> dict:
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"normal": 0, "bacteria": 1, "virus": 2}


def load_data(subtype_to_int: dict) -> dict:
    """Load scores and split indices.  Does NOT load test_idx into the model path."""
    feat          = np.load(FEATURES_PATH, allow_pickle=True)
    scores        = feat["scores"]
    concept_names = feat["concept_names"].tolist()

    split          = np.load(SPLIT_PATH, allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]
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
        "patient_ids":     patient_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pre-run sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(
    data:           dict,
    subtype_to_int: dict,
    val_split:      dict,
    hidden_dims:    tuple,
    dropout:        float,
    concept_names:  list,
) -> None:
    """Checks 1–5.  sys.exit(1) on hard failure; prints warning on soft."""
    print("\n" + "=" * 65)
    print("PRE-RUN SANITY CHECKS")
    print("=" * 65)

    # 1. Concurvity winner.json state
    if not os.path.exists(CONCURVITY_WINNER_JSON):
        print(f"  [1] FAIL: {CONCURVITY_WINNER_JSON} not found.")
        sys.exit(1)
    with open(CONCURVITY_WINNER_JSON, encoding="utf-8") as f:
        cw = json.load(f)
    if cw.get("selection_pending", True):
        print(f"  [1] FAIL: selection_pending=True in winner.json.")
        sys.exit(1)
    if "operative_lambda_c" not in cw:
        print(f"  [1] FAIL: operative_lambda_c missing from winner.json.")
        sys.exit(1)
    op_lam = float(cw["operative_lambda_c"])
    print(f"  [1] winner.json: selection_pending=false, operative_lambda_c={op_lam}  ✓")

    # 2. Val split reproducibility
    EXPECTED_TR10  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    EXPECTED_VAL10 = [29, 33, 47, 48, 52, 53, 55, 63, 64, 65]
    train_rel = val_split["train_rel"]
    val_rel   = val_split["val_rel"]
    got_tr10  = train_rel[:10].tolist()
    got_val10 = val_rel[:10].tolist()
    if got_tr10 != EXPECTED_TR10 or got_val10 != EXPECTED_VAL10:
        print(f"  [2] FAIL: val-split mismatch!")
        print(f"      expected train[:10] = {EXPECTED_TR10}")
        print(f"      got      train[:10] = {got_tr10}")
        print(f"      expected val[:10]   = {EXPECTED_VAL10}")
        print(f"      got      val[:10]   = {got_val10}")
        sys.exit(1)
    print(f"  [2] Val-split indices match STEP 2/3/4  ✓")
    print(f"      train_final[:10] = {got_tr10}")
    print(f"      val[:10]         = {got_val10}")
    print(f"      sizes: train_final={len(train_rel)}, val={len(val_rel)}")

    # 3. Class weights
    y_train_pool  = data["labels_all"][data["train_pool_idx"]]
    y_train_final = y_train_pool[train_rel]
    counts        = np.bincount(y_train_final, minlength=NUM_CLASSES)
    n_tr          = len(y_train_final)
    weights       = n_tr / (NUM_CLASSES * counts.astype(np.float64))
    int_to_name   = {v: k for k, v in subtype_to_int.items()}
    EXPECTED_W    = {"normal": 1.2172, "bacteria": 0.7019, "virus": 1.3269}
    tol           = 0.0002
    mismatch = any(
        abs(weights[i] - EXPECTED_W[int_to_name[i]]) > tol for i in range(NUM_CLASSES)
    )
    if mismatch:
        for i in range(NUM_CLASSES):
            print(f"  [3] {int_to_name[i]}: got {weights[i]:.4f}, "
                  f"expected ≈ {EXPECTED_W[int_to_name[i]]}")
        print(f"  [3] FAIL: class weights diverge from STEP 2/3/4 reference.")
        sys.exit(1)
    print(f"  [3] Class weights  ✓")
    for i in range(NUM_CLASSES):
        print(f"        {int_to_name[i]:10s}: n={counts[i]}, weight={weights[i]:.4f}")

    # 4. Proximal operator sanity: fresh-init model norms should be comparable
    set_all_seeds(SEED)
    _model_init = NAMMulticlass(
        n_features=N_FEATURES,
        num_classes=NUM_CLASSES,
        hidden_dims=hidden_dims,
        dropout=dropout,
        concept_names=concept_names,
    ).to(DEVICE)
    init_norms = feature_group_norms(_model_init)
    norm_vals  = list(init_norms.values())
    min_n, max_n, med_n = min(norm_vals), max(norm_vals), float(np.median(norm_vals))
    print(f"  [4] Init feature-group norms (Xavier):  "
          f"min={min_n:.4f}  med={med_n:.4f}  max={max_n:.4f}")
    phantom = sum(1 for v in norm_vals if v < 1e-4)
    if phantom > 0:
        print(f"  [4] ⚠ {phantom} concept(s) have init norm < 1e-4 — "
              f"proximal could phantom-eliminate them at step 0.")
    else:
        print(f"  [4] No phantom-zero norms at init  ✓")
    for nm, nv in sorted(init_norms.items(), key=lambda x: x[1]):
        print(f"        {nm:30s}: {nv:.4f}")

    # 5. Step-0 shrinkage diagnostic
    # Shrinkage factor = lr × λ_s / ‖θ‖  (fraction of norm removed per batch pass)
    shrinkage_med = LR * LAMBDA_0 / med_n
    shrinkage_min = LR * LAMBDA_0 / max_n   # hardest to shrink
    shrinkage_max = LR * LAMBDA_0 / min_n   # easiest to shrink
    print(f"  [5] Step-0 proximal shrinkage factor (lr × λ_0 / ‖θ‖):")
    print(f"        lr={LR:.3e}, λ_0={LAMBDA_0:.3e}")
    print(f"        median concept: {shrinkage_med:.4f}  "
          f"(range: {shrinkage_min:.4f} – {shrinkage_max:.4f})")
    if shrinkage_med < 0.001:
        print(f"        ⚠ shrinkage_med < 0.001 — proximal operator may be too weak. "
              f"Consider increasing λ_0.")
    elif shrinkage_med > 0.5:
        print(f"        ⚠ shrinkage_med > 0.5 — proximal operator may be too aggressive "
              f"(risk of phantom elimination at step 0).")
    else:
        print(f"        In viable range [0.001, 0.5]  ✓")
    # Compare to HAM10000 v0 failure case: shrinkage ≈ 1.7e-6
    print(f"        (HAM10000 v0 failure: ~1.7e-6; v3 fix target: ~0.2)")

    del _model_init
    print("=" * 65)
    print("All pre-run sanity checks passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Core: one condition run
# ─────────────────────────────────────────────────────────────────────────────

def run_one_condition(
    *,
    condition:          str,
    concurvity_lambda:  float,
    hidden_dims:        tuple,
    dropout:            float,
    weight_decay:       float,
    concept_names:      list,
    class_names:        list,
    X_train_final_raw:  np.ndarray,
    X_val_raw:          np.ndarray,
    y_train_final:      np.ndarray,   # int64
    y_val:              np.ndarray,   # int64
    out_dir:            str,
    lambda_0:           float = LAMBDA_0,
    epsilon:            float = EPSILON,
    max_lambda:         float = MAX_LAMBDA,
    max_lambda_steps:   int   = MAX_LAM_STEPS,
    max_dense_epochs:   int   = MAX_DENSE_EPOCHS,
    dense_patience:     int   = DENSE_PATIENCE,
    max_warm_epochs:    int   = MAX_WARM_EPOCHS,
    warm_patience:      int   = WARM_PATIENCE,
    warm_min_delta:     float = WARM_MIN_DELTA,
) -> dict:
    """Train dense checkpoint then traverse warm-start sparsity path.

    Issue 4 fix: best-within-step val_loss tracked; best state restored before
    advancing to λ+1.
    Issue 9 fix: warmup_epochs=0 (concurvity active from epoch 1 when λ_c > 0).
    Test set is NEVER touched anywhere in this function.
    """
    seed_dir = os.path.join(out_dir, f"seed_{SEED}")
    os.makedirs(seed_dir, exist_ok=True)

    set_all_seeds(SEED)
    pin_memory = (DEVICE.type == "cuda")

    # ── Per-condition scaler (fit on inner-train only) ────────────────────────
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train_final_raw).astype(np.float32)
    X_val_sc  = scaler.transform(X_val_raw).astype(np.float32)
    with open(os.path.join(seed_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # ── Class weights (inverse-frequency on inner-train) ─────────────────────
    counts   = np.bincount(y_train_final, minlength=NUM_CLASSES)
    n_tr     = len(y_train_final)
    weights  = n_tr / (NUM_CLASSES * counts.astype(np.float64))
    w_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)

    X_val_t = torch.tensor(X_val_sc,  dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val,     dtype=torch.long,    device=DEVICE)

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

    def _eval_val_full() -> tuple:
        """Returns (balacc, r_perp, val_loss_ce, val_loss_full)."""
        model.eval()
        with torch.no_grad():
            logits, shape_outs = model(X_val_t, return_shape_outputs=True)
            val_loss_ce = criterion(logits, y_val_t).item()
            preds       = logits.argmax(dim=1).cpu().numpy()
            r_perp      = multiclass_concurvity(shape_outs).item()
        balacc = balanced_accuracy_score(y_val, preds)
        val_loss_full = val_loss_ce + concurvity_lambda * r_perp
        return float(balacc), float(r_perp), float(val_loss_ce), float(val_loss_full)

    # ── Phase 1: Dense checkpoint (with per-condition caching) ────────────────
    conc_tag  = str(concurvity_lambda).replace(".", "p")
    ckpt_pt   = os.path.join(seed_dir, f"dense_seed{SEED}_conc{conc_tag}.pt")
    ckpt_json = os.path.join(seed_dir, f"dense_seed{SEED}_conc{conc_tag}.json")

    if os.path.exists(ckpt_pt) and os.path.exists(ckpt_json):
        model.load_state_dict(
            torch.load(ckpt_pt, map_location=DEVICE, weights_only=True)
        )
        with open(ckpt_json, encoding="utf-8") as fj:
            dense_meta = json.load(fj)
        dense_val_balacc = float(dense_meta["dense_val_balacc"])
        dense_r_perp     = float(dense_meta.get("dense_r_perp", float("nan")))
        print(f"  [dense] Loaded cached checkpoint: "
              f"val_balacc={dense_val_balacc:.4f}  r_perp={dense_r_perp:.4f}")
    else:
        print(f"  [dense] Training (condition={condition}, λ_c={concurvity_lambda}) ...")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=SCHED_FAC, patience=SCHED_PAT, min_lr=1e-6
        )
        best_val_balacc  = -1.0
        patience_ctr     = 0
        best_dense_state = None

        for epoch in range(max_dense_epochs):
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
                torch.save(model.state_dict(), ckpt_pt)
            else:
                patience_ctr += 1
                if patience_ctr >= dense_patience:
                    print(f"  [dense] Early stop epoch {epoch+1}  "
                          f"best={best_val_balacc:.4f}")
                    break

        if best_dense_state is not None:
            model.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_dense_state.items()}
            )
        elif os.path.exists(ckpt_pt):
            model.load_state_dict(
                torch.load(ckpt_pt, map_location=DEVICE, weights_only=True)
            )

        dense_val_balacc, dense_r_perp, _, _ = _eval_val_full()
        torch.save(model.state_dict(), ckpt_pt)
        with open(ckpt_json, "w", encoding="utf-8") as fj:
            json.dump({
                "condition":         condition,
                "concurvity_lambda": concurvity_lambda,
                "dense_val_balacc":  dense_val_balacc,
                "dense_r_perp":      dense_r_perp,
                "hidden_dims":       list(hidden_dims),
                "dropout":           dropout,
                "weight_decay":      weight_decay,
                "seed":              SEED,
                "timestamp":         datetime.now(timezone.utc).isoformat(),
            }, fj, indent=2)
        print(f"  [dense] Done.  val_balacc={dense_val_balacc:.4f}  "
              f"r_perp={dense_r_perp:.4f}")

    # ── Phase 2: Lambda schedule ──────────────────────────────────────────────
    lambda_schedule: list[float] = []
    t = 0
    while len(lambda_schedule) < max_lambda_steps:
        lam = lambda_0 * (1.0 + epsilon) ** t
        if lam > max_lambda:
            break
        lambda_schedule.append(lam)
        t += 1
    print(f"  [path]  {len(lambda_schedule)} steps: "
          f"{lambda_schedule[0]:.3e} → {lambda_schedule[-1]:.3e}")

    prev_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    prev_norms      = feature_group_norms(model)

    path_rows:             list[dict] = []
    norms_per_step_rows:   list[dict] = []
    elim_order:            list[tuple] = []   # (concept_name, step, lambda_s)
    elim_set:              set   = set()
    monotone_n_active:     bool  = True
    prev_n_active:         int   = sum(1 for v in prev_norms.values() if v > ZERO_THRESHOLD)
    t0_path = time.time()

    for step_idx, lambda_t in enumerate(lambda_schedule):
        t_step = time.time()

        # Issue 4 fix: warm-start from BEST previous-step state
        model.load_state_dict({k: v.to(DEVICE) for k, v in prev_state_dict.items()})

        # Fresh optimizer + scheduler per step
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHED_FAC, patience=SCHED_PAT, min_lr=1e-6
        )

        best_step_val_loss  = float("inf")
        best_step_state     = None
        no_improve_ctr      = 0
        actual_epochs       = 0

        for epoch in range(max_warm_epochs):
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
                # Proximal block soft-thresholding after each batch
                apply_proximal_step(
                    model,
                    lr=optimizer.param_groups[0]["lr"],
                    sparsity_lambda=lambda_t,
                )

            actual_epochs += 1
            _, _, _, val_loss_full = _eval_val_full()
            scheduler.step(val_loss_full)

            # Issue 4 fix: best-within-step tracking
            if val_loss_full < best_step_val_loss - warm_min_delta:
                best_step_val_loss = val_loss_full
                no_improve_ctr     = 0
                best_step_state    = {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}
            else:
                no_improve_ctr += 1
                if no_improve_ctr >= warm_patience:
                    break

        # Issue 4 fix: restore best-within-step before recording / advancing
        if best_step_state is not None:
            model.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_step_state.items()}
            )

        # Evaluate at best-within-step state
        val_balacc_best, val_r_perp_best, val_loss_ce_best, val_loss_full_best = _eval_val_full()
        norms    = feature_group_norms(model)
        n_active = sum(1 for v in norms.values() if v > ZERO_THRESHOLD)
        r_sparse = sum(norms.values())   # group-lasso regularization proxy
        step_sec = time.time() - t_step

        # Monotonicity check
        if n_active > prev_n_active:
            monotone_n_active = False
            print(f"  ⚠ step {step_idx+1}: n_active INCREASED "
                  f"({prev_n_active} → {n_active}) — re-activation detected.",
                  file=sys.stderr)

        # Detect new eliminations
        just_eliminated: list[str] = []
        for k in concept_names:
            if prev_norms[k] >= ZERO_THRESHOLD and norms[k] < ZERO_THRESHOLD:
                if k not in elim_set:
                    just_eliminated.append(k)
                    elim_order.append((k, step_idx + 1, lambda_t))
                    elim_set.add(k)

        # Path row
        path_rows.append({
            "step":                    step_idx + 1,
            "lambda_s":                lambda_t,
            "n_active":                n_active,
            "val_loss_best_in_step":   val_loss_full_best,
            "val_balacc_best_in_step": val_balacc_best,
            "val_r_perp_best_in_step": val_r_perp_best,
            "epochs_used_in_step":     actual_epochs,
            "r_sparse":                r_sparse,
        })

        # Norms per step row
        norms_row = {"step": step_idx + 1, "lambda_s": lambda_t}
        norms_row.update({f"norm_{k}": norms[k] for k in concept_names})
        norms_per_step_rows.append(norms_row)

        # Console progress (every 10 steps or on elimination)
        if just_eliminated or (step_idx + 1) % 10 == 0:
            elim_str = f"  ELIM: {just_eliminated}" if just_eliminated else ""
            print(
                f"  step {step_idx+1:3d}  λ={lambda_t:.4e}  "
                f"n_active={n_active:2d}  val_balacc={val_balacc_best:.4f}  "
                f"val_r_perp={val_r_perp_best:.4f}  ep={actual_epochs}  "
                f"t={step_sec:.1f}s{elim_str}"
            )

        # Advance: carry BEST state to next step
        prev_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        prev_norms      = norms
        prev_n_active   = n_active

        if n_active == 0:
            print(f"  [path] All {N_FEATURES} concepts zeroed at "
                  f"λ={lambda_t:.4e}. Stopping early.")
            break

    elapsed_path = time.time() - t0_path

    # ── Write per-condition artefacts ─────────────────────────────────────────
    path_df = pd.DataFrame(path_rows)
    path_df.to_csv(os.path.join(seed_dir, "path.csv"), index=False)

    norms_df = pd.DataFrame(norms_per_step_rows)
    norms_df.to_csv(
        os.path.join(seed_dir, "feature_group_norms_per_step.csv"), index=False
    )

    # Elimination table
    with open(os.path.join(seed_dir, "elimination_order.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"condition={condition}  seed={SEED}  lambda_c={concurvity_lambda}\n")
        f.write(f"dense_val_balacc={dense_val_balacc:.4f}\n\n")
        f.write(f"{'order':>5}  {'step':>5}  {'lambda_s':>12}  concept\n")
        f.write("-" * 50 + "\n")
        for i, (nm, st, lv) in enumerate(elim_order, 1):
            f.write(f"{i:>5}  {st:>5}  {lv:>12.4e}  {nm}\n")
        never = [k for k in concept_names if k not in elim_set]
        if never:
            f.write(f"\nNever zeroed ({len(never)} concepts):\n")
            for k in never:
                f.write(f"  {k}\n")

    print(f"  [condition] {len(path_rows)} steps in {elapsed_path/60:.1f} min  "
          f"(avg {elapsed_path/max(len(path_rows),1):.1f} s/step)")

    # ── Compute pass/fail ─────────────────────────────────────────────────────
    first_elim_step   = elim_order[0][1] if elim_order else None
    first_elim_lam    = elim_order[0][2] if elim_order else None
    first_elim_conc   = elim_order[0][0] if elim_order else None
    n_active_final    = path_rows[-1]["n_active"] if path_rows else None
    lambda_final      = path_rows[-1]["lambda_s"] if path_rows else None
    crit_a            = (first_elim_step is not None and first_elim_step <= 50)
    crit_b            = (n_active_final is not None and n_active_final <= 10)
    passed            = crit_a and crit_b

    return {
        "condition":            condition,
        "concurvity_lambda":    concurvity_lambda,
        "dense_val_balacc":     dense_val_balacc,
        "n_steps_run":          len(path_rows),
        "elapsed_min":          elapsed_path / 60.0,
        "first_elimination_step":    first_elim_step,
        "first_elimination_lambda":  first_elim_lam,
        "first_eliminated_concept":  first_elim_conc,
        "n_active_at_final_step":    n_active_final,
        "lambda_at_final_step":      lambda_final,
        "monotone_n_active":         monotone_n_active,
        "criterion_b_first_elim_le_50":   crit_a,
        "criterion_b_n_active_le_10_at_100": crit_b,
        "criterion_b_pass":          passed,
        "elim_order":                elim_order,
        "path_df":                   path_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sanity_only", action="store_true",
                        help="Run sanity checks then exit without training.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 3 lambda steps per condition (mechanics check).")
    parser.add_argument("--max_lambda_steps", type=int, default=MAX_LAM_STEPS,
                        help=f"Override number of lambda steps (default={MAX_LAM_STEPS}).")
    parser.add_argument("--max_dense_epochs", type=int, default=MAX_DENSE_EPOCHS)
    parser.add_argument("--max_warm_epochs",  type=int, default=MAX_WARM_EPOCHS)
    parser.add_argument("--arch_winner_json", type=str, default=None)
    parser.add_argument("--out_root",         type=str, default=None)
    args = parser.parse_args()

    max_lam_steps = 3 if args.dry_run else args.max_lambda_steps

    # ── Load architecture winner ───────────────────────────────────────────────
    arch_json = args.arch_winner_json or ARCH_WINNER_JSON
    if not os.path.exists(arch_json):
        raise FileNotFoundError(f"Architecture winner not found: {arch_json}")
    with open(arch_json, encoding="utf-8") as f:
        arch = json.load(f)
    hidden_dims  = tuple(arch["hidden_dims"])
    dropout      = float(arch["dropout"])
    weight_decay = float(arch["weight_decay"])

    # ── Load concurvity operative lambda ──────────────────────────────────────
    if not os.path.exists(CONCURVITY_WINNER_JSON):
        raise FileNotFoundError(f"Concurvity winner not found: {CONCURVITY_WINNER_JSON}")
    with open(CONCURVITY_WINNER_JSON, encoding="utf-8") as f:
        cw = json.load(f)
    if cw.get("selection_pending", True):
        raise RuntimeError("selection_pending=True in concurvity winner.json.")
    operative_lam_c = float(cw["operative_lambda_c"])

    out_root = args.out_root or OUT_ROOT
    os.makedirs(out_root, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    subtype_to_int = load_label_mapping()
    int_to_subtype = {v: k for k, v in subtype_to_int.items()}
    class_names    = [int_to_subtype[i] for i in range(NUM_CLASSES)]

    data          = load_data(subtype_to_int)
    scores        = data["scores"]
    concept_names = data["concept_names"]
    labels_all    = data["labels_all"]
    train_pool_idx = data["train_pool_idx"]
    patient_ids   = data["patient_ids"]

    X_train_pool = scores[train_pool_idx]
    y_train_pool = labels_all[train_pool_idx]
    groups_pool  = patient_ids[train_pool_idx]

    # ── Fixed val split (identical to all prior STEPS) ────────────────────────
    val_split = make_fixed_val_split(
        X_train_pool,
        y_train_pool.astype(str),
        groups_pool,
        ["0", "1", "2"],
        val_random_state=VAL_RANDOM_STATE,
    )
    train_rel = val_split["train_rel"]
    val_rel   = val_split["val_rel"]

    # ── Sanity checks ─────────────────────────────────────────────────────────
    run_sanity_checks(
        data, subtype_to_int, val_split,
        hidden_dims, dropout, concept_names,
    )

    if args.sanity_only:
        print("--sanity_only flag set. Exiting before sweep.")
        return

    # ── Slice arrays (no test set) ────────────────────────────────────────────
    X_train_final_raw = X_train_pool[train_rel]
    X_val_raw         = X_train_pool[val_rel]
    y_train_final     = y_train_pool[train_rel]
    y_val             = y_train_pool[val_rel]

    # ── Banner ─────────────────────────────────────────────────────────────────
    sched_preview = [LAMBDA_0 * (1.0 + EPSILON) ** t
                     for t in [0, 25, 50, 75, max_lam_steps - 1]]
    print(f"\n{'='*68}")
    print(f"Chest X-ray NAM — Sparsity sweep SMOKE TEST")
    print(f"  [Issue 4] best-within-step checkpoint restored before λ+1")
    print(f"  [Issue 9] warmup_epochs=0 (concurvity active from epoch 1 when λ_c>0)")
    print(f"  Config: hidden={list(hidden_dims)}, dropout={dropout}, wd={weight_decay:.0e}")
    print(f"  Conditions: sparsity_only (λ_c=0.0) → sparsity_conc (λ_c={operative_lam_c})")
    print(f"  Seed: {SEED}  |  max_lambda_steps={max_lam_steps}")
    print(f"  Schedule: λ_0={LAMBDA_0:.3e}, ε={EPSILON}, max_λ={MAX_LAMBDA:.1e}")
    print(f"  Preview λ at steps {[0,25,50,75,max_lam_steps-1]}: "
          f"{[f'{v:.3e}' for v in sched_preview]}")
    print(f"  Per-step budget: max_warm_epochs={args.max_warm_epochs}, "
          f"warm_patience={WARM_PATIENCE}")
    print(f"  val_random_state={VAL_RANDOM_STATE} (identical to STEP 2/3/4)")
    print(f"  Test set: NOT LOADED (smoke test, validation metrics only)")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {out_root}/")
    if args.dry_run:
        print(f"  [DRY RUN — {max_lam_steps} steps only]")
    print(f"{'='*68}\n")

    # ── Save run_config.json at startup ───────────────────────────────────────
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT), text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_hash = None

    run_cfg = {
        "smoke_test":          True,
        "seed":                SEED,
        "max_lambda_steps":    max_lam_steps,
        "lambda_0":            LAMBDA_0,
        "epsilon":             EPSILON,
        "max_lambda":          MAX_LAMBDA,
        "max_dense_epochs":    args.max_dense_epochs,
        "max_warm_epochs":     args.max_warm_epochs,
        "warm_patience":       WARM_PATIENCE,
        "warm_min_delta":      WARM_MIN_DELTA,
        "conditions":          ["sparsity_only", "sparsity_conc"],
        "sparsity_only_lam_c": 0.0,
        "sparsity_conc_lam_c": operative_lam_c,
        "concurvity_winner_source": CONCURVITY_WINNER_JSON,
        "val_random_state":    VAL_RANDOM_STATE,
        "val_split_note":      "GroupShuffleSplit(random_state=42), identical to STEP 2/3/4",
        "config_id":           arch["config_id"],
        "hidden_dims":         list(hidden_dims),
        "dropout":             dropout,
        "weight_decay":        weight_decay,
        "n_features":          N_FEATURES,
        "num_classes":         NUM_CLASSES,
        "feature_file":        FEATURES_PATH,
        "split_file":          SPLIT_PATH,
        "zero_threshold":      ZERO_THRESHOLD,
        "pass_criteria":       {
            "crit_a": "first_elimination_step <= 50",
            "crit_b": "n_active_at_step_100 <= 10",
        },
        "test_set_touched":    False,
        "dry_run":             args.dry_run,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "git_commit":          git_hash,
    }
    with open(os.path.join(out_root, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    # ── Run both conditions ───────────────────────────────────────────────────
    CONDITION_SPECS = [
        ("sparsity_only", 0.0),
        ("sparsity_conc",  operative_lam_c),
    ]
    all_results = {}
    t_total = time.time()

    for condition, lam_c in CONDITION_SPECS:
        print(f"\n{'─'*68}")
        print(f"[{condition}]  λ_c={lam_c}")
        print(f"{'─'*68}")

        cond_dir = os.path.join(out_root, condition)
        os.makedirs(cond_dir, exist_ok=True)

        t_cond = time.time()
        result = run_one_condition(
            condition=condition,
            concurvity_lambda=lam_c,
            hidden_dims=hidden_dims,
            dropout=dropout,
            weight_decay=weight_decay,
            concept_names=concept_names,
            class_names=class_names,
            X_train_final_raw=X_train_final_raw,
            X_val_raw=X_val_raw,
            y_train_final=y_train_final,
            y_val=y_val,
            out_dir=cond_dir,
            max_lambda_steps=max_lam_steps,
            max_dense_epochs=args.max_dense_epochs,
            max_warm_epochs=args.max_warm_epochs,
        )
        elapsed_cond = time.time() - t_cond
        result["elapsed_cond_min"] = elapsed_cond / 60.0
        all_results[condition] = result

        # Progress table for this condition (strategic checkpoints)
        path_df = result["path_df"]
        print(f"\n  Progress table [{condition}]:")
        print(f"  {'step':>5}  {'lambda_s':>10}  {'n_active':>8}  "
              f"{'val_balacc':>10}  {'val_r_perp':>10}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")
        checkpoints = {0, 25, 50, 75, len(path_df) - 1}
        for i, row in path_df.iterrows():
            if i in checkpoints or row["step"] == 1:
                print(f"  {int(row['step']):>5}  {row['lambda_s']:>10.4e}  "
                      f"{int(row['n_active']):>8}  "
                      f"{row['val_balacc_best_in_step']:>10.4f}  "
                      f"{row['val_r_perp_best_in_step']:>10.4f}")

        # Elimination sequence for this condition
        if result["elim_order"]:
            print(f"\n  Elimination sequence [{condition}]:")
            for i, (nm, st, lv) in enumerate(result["elim_order"], 1):
                print(f"    #{i:2d}  step={st:3d}  λ={lv:.4e}  {nm}")
        else:
            print(f"\n  No eliminations in {len(path_df)} steps [{condition}].")

        print(f"\n  [{condition}] Wall-clock: {elapsed_cond/60:.1f} min  |  "
              f"first_elim_step={result['first_elimination_step']}  |  "
              f"n_active_final={result['n_active_at_final_step']}  |  "
              f"PASS={result['criterion_b_pass']}")

    # ── Compute overall verdict ───────────────────────────────────────────────
    r_so   = all_results["sparsity_only"]
    r_sc   = all_results["sparsity_conc"]
    n_pass = sum(1 for r in [r_so, r_sc] if r["criterion_b_pass"])
    if n_pass == 2:
        verdict = "PASS"
    elif n_pass == 1:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    # Recommendation
    def _make_recommendation(r: dict) -> str:
        if r["criterion_b_pass"]:
            return "Schedule is viable for this condition. Proceed to full sweep."
        elif r["first_elimination_step"] is None:
            return (
                "No eliminations occurred in 100 steps. "
                "The λ schedule is not reaching the sparsity-active range. "
                "Consider increasing λ_0 (e.g., to 3.0) or ε (e.g., to 0.06)."
            )
        elif not r["criterion_b_first_elim_le_50"]:
            fe = r["first_elimination_step"]
            return (
                f"First elimination at step {fe} > 50. "
                f"Schedule is too slow to enter the active range early enough. "
                f"Consider increasing λ_0 or ε to shift the onset left."
            )
        else:
            naf = r["n_active_at_final_step"]
            return (
                f"First elimination ≤ 50 ✓, but n_active={naf} > 10 at step 100. "
                f"Schedule needs more steps or faster ramp to achieve sufficient sparsity. "
                f"Consider increasing max_lambda_steps to 150 or ε to 0.05."
            )

    rec_so = _make_recommendation(r_so)
    rec_sc = _make_recommendation(r_sc)
    if verdict == "PASS":
        recommendation = "Both conditions pass. Schedule λ_0=1.0, ε=0.04, 150 steps is viable for chest X-ray. Proceed to full 5-seed sweep."
    elif verdict == "PARTIAL":
        failing_cond = "sparsity_only" if not r_so["criterion_b_pass"] else "sparsity_conc"
        recommendation = f"PARTIAL: {failing_cond} failed. " + (rec_so if not r_so["criterion_b_pass"] else rec_sc)
    else:
        recommendation = f"FAIL: Both conditions failed. sparsity_only: {rec_so}  sparsity_conc: {rec_sc}"

    # ── smoke_test_summary.json ───────────────────────────────────────────────
    def _cond_summary(r: dict) -> dict:
        return {
            "first_elimination_step":        r["first_elimination_step"],
            "first_elimination_lambda":      r["first_elimination_lambda"],
            "first_eliminated_concept":      r["first_eliminated_concept"],
            "n_active_at_step_100":          r["n_active_at_final_step"],
            "lambda_at_step_100":            r["lambda_at_final_step"],
            "monotone_n_active":             r["monotone_n_active"],
            "criterion_b_first_elim_le_50":  r["criterion_b_first_elim_le_50"],
            "criterion_b_n_active_le_10_at_100": r["criterion_b_n_active_le_10_at_100"],
            "criterion_b_pass":              r["criterion_b_pass"],
            "dense_val_balacc":              r["dense_val_balacc"],
            "concurvity_lambda":             r["concurvity_lambda"],
            "n_steps_run":                   r["n_steps_run"],
            "elapsed_min":                   round(r["elapsed_cond_min"], 1),
        }

    summary = {
        "sparsity_only": _cond_summary(r_so),
        "sparsity_conc":  _cond_summary(r_sc),
        "overall_verdict": verdict,
        "recommendation":  recommendation,
        "schedule_params": {
            "lambda_0": LAMBDA_0, "epsilon": EPSILON,
            "max_lambda_steps": max_lam_steps,
            "max_warm_epochs":  args.max_warm_epochs,
            "warm_patience":    WARM_PATIENCE,
        },
        "total_elapsed_min": round((time.time() - t_total) / 60.0, 1),
        "dry_run": args.dry_run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = os.path.join(out_root, "smoke_test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Final printout ─────────────────────────────────────────────────────────
    total_min = (time.time() - t_total) / 60.0
    print(f"\n{'='*68}")
    print(f"Chest X-ray sparsity sweep SMOKE TEST — COMPLETE")
    print(f"")
    print(f"  {'Condition':<20}  {'first_elim_step':>15}  {'n_active@100':>12}  {'PASS':>6}")
    print(f"  {'-'*20}  {'-'*15}  {'-'*12}  {'-'*6}")
    for cond_name, r in [("sparsity_only", r_so), ("sparsity_conc", r_sc)]:
        fe  = str(r["first_elimination_step"]) if r["first_elimination_step"] else "none"
        naf = str(r["n_active_at_final_step"]) if r["n_active_at_final_step"] is not None else "?"
        print(f"  {cond_name:<20}  {fe:>15}  {naf:>12}  {'✓' if r['criterion_b_pass'] else '✗':>6}")
    print(f"")
    print(f"  Overall verdict : {verdict}")
    print(f"  Recommendation  : {recommendation}")
    print(f"")
    if r_sc["first_elimination_step"] is not None and r_so["first_elimination_step"] is not None:
        delta_steps = r_so["first_elimination_step"] - r_sc["first_elimination_step"]
        if delta_steps > 0:
            print(f"  sparsity_conc eliminates first concept {delta_steps} step(s) earlier than "
                  f"sparsity_only — consistent with concurvity reducing parameter norms.")
        elif delta_steps < 0:
            print(f"  sparsity_only eliminates first concept {-delta_steps} step(s) earlier — "
                  f"unexpected; concurvity may have increased parameter norms.")
        else:
            print(f"  Both conditions first eliminated at the same step.")
    print(f"")
    print(f"  Total elapsed   : {total_min:.1f} min  "
          f"({r_so['elapsed_cond_min']:.1f} min + {r_sc['elapsed_cond_min']:.1f} min)")
    print(f"  smoke_test_summary → {summary_path}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
