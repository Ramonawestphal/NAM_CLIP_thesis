"""
Sparsity sweep FULL — chest X-ray three-way task (5 seeds × 2 conditions × 150 steps).

Chest X-ray analogue of HAM10000 v7 STEP 5/6 sparsity sweep.
Smoke test (single seed, 100 steps) confirmed λ_0=1.0, ε=0.04 is live for both
conditions; this script runs the full 5-seed × 150-step sweep for downstream
ANEC evaluation at K ∈ {5, 8, 10, 15}.

Audit fixes mirrored from v7
────────────────────────────
  Issue 4: best-within-step checkpoint restored before advancing to λ+1
  Issue 9: warmup_epochs=0 (concurvity active from epoch 1 when λ_c > 0)
  Issue 3: set_all_seeds() includes random.seed
  Issue 7: per-seed scaler saved
  Issue 8: CUDA determinism flags set

Test set is NEVER loaded or touched. ANEC evaluation (STEP 6) loads test_idx.

Usage (from project root):
    python scripts/chestxray/run_sparsity_sweep.py
    python scripts/chestxray/run_sparsity_sweep.py --sanity_only
    python scripts/chestxray/run_sparsity_sweep.py --dry_run   # 3 steps × 1 seed each
    python scripts/chestxray/run_sparsity_sweep.py --seeds 42 43  # subset of seeds
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts.v7._common import (
    set_all_seeds,
    make_fixed_val_split,
)
from src.models.nam_multiclass import NAMMulticlass
from src.models.concurvity import multiclass_concurvity
from src.models.sparsity import feature_group_norms, apply_proximal_step

# ── Constants ─────────────────────────────────────────────────────────────────
SEEDS            = [42, 43, 44, 45, 46]
LR               = 1e-3
BATCH_SIZE       = 256
N_FEATURES       = 17
NUM_CLASSES      = 3
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

# Lambda schedule
LAMBDA_0      = 1.0
EPSILON       = 0.04
MAX_LAMBDA    = 1e3
MAX_LAM_STEPS = 150   # full sweep: 150

# Smoke test seed-42 reference values (for reproduction check)
SMOKE_SO_SEED42 = {
    "first_elim_step":    37,
    "first_elim_concept": "perihilar_infiltrates",
    "n_active_at_100":    5,
}
SMOKE_SC_SEED42 = {
    "first_elim_step":    32,
    "first_elim_concept": None,   # not checked by concept name; any is fine
    "n_active_at_100":    3,
}

# Paths
FEATURES_PATH          = "data/features/biomedclip/chestxray_concept_scores_v4.npz"
SPLIT_PATH             = "data/splits/chestxray_outer_split.npz"
LABEL_MAP_PATH         = "results/chestxray/architecture_selection/label_mapping.json"
ARCH_WINNER_JSON       = "results/chestxray/architecture_selection/winning_config.json"
CONCURVITY_WINNER_JSON = "results/chestxray/concurvity_sweep/winner.json"
OUT_ROOT               = "results/chestxray/sparsity_sweep"

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
# Pre-run sanity checks (checks 1–3; check 4 runs inline after seed 42)
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(
    data:           dict,
    subtype_to_int: dict,
    val_split:      dict,
    hidden_dims:    tuple,
    dropout:        float,
    concept_names:  list,
) -> None:
    """Checks 1–3.  sys.exit(1) on hard failure; prints warning on soft."""
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
    print(f"  [2] Val-split indices match STEP 2/3/4/smoke  ✓")
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
        print(f"  [3] FAIL: class weights diverge from STEP 2/3/4/smoke reference.")
        sys.exit(1)
    print(f"  [3] Class weights  ✓")
    for i in range(NUM_CLASSES):
        print(f"        {int_to_name[i]:10s}: n={counts[i]}, weight={weights[i]:.4f}")

    # Proximal operator note (informational, not a hard failure)
    set_all_seeds(42)
    _model_init = NAMMulticlass(
        n_features=N_FEATURES,
        num_classes=NUM_CLASSES,
        hidden_dims=hidden_dims,
        dropout=dropout,
        concept_names=concept_names,
    ).to(DEVICE)
    init_norms = feature_group_norms(_model_init)
    norm_vals  = list(init_norms.values())
    med_n      = float(np.median(norm_vals))
    shrinkage  = LR * LAMBDA_0 / med_n
    print(f"  [note] Step-0 proximal shrinkage: lr × λ_0 / ‖θ‖_med "
          f"= {LR:.0e} × {LAMBDA_0:.1f} / {med_n:.2f} = {shrinkage:.5f}")
    print(f"         (Smoke test confirmed live zone; see sparsity_sweep_smoke/ for full check)")
    del _model_init

    print("=" * 65)
    print("Pre-run checks 1–3 passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Core: one seed × one condition
# ─────────────────────────────────────────────────────────────────────────────

def run_one_seed(
    *,
    seed:               int,
    condition:          str,
    concurvity_lambda:  float,
    hidden_dims:        tuple,
    dropout:            float,
    weight_decay:       float,
    concept_names:      list,
    X_train_final_raw:  np.ndarray,
    X_val_raw:          np.ndarray,
    y_train_final:      np.ndarray,   # int64
    y_val:              np.ndarray,   # int64
    seed_dir:           str,
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
    """Train dense checkpoint then traverse warm-start sparsity path for one seed.

    Issue 4 fix: best-within-step val_loss tracked; best state restored before
    advancing to λ+1.
    Issue 9 fix: warmup_epochs=0 (concurvity active from epoch 1 when λ_c > 0).
    Test set is NEVER touched anywhere in this function.
    """
    os.makedirs(seed_dir, exist_ok=True)

    set_all_seeds(seed)
    pin_memory = (DEVICE.type == "cuda")

    # ── Per-seed scaler (fit on inner-train only) ─────────────────────────────
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

    # ── Phase 1: Dense checkpoint (with per-seed caching) ────────────────────
    conc_tag  = str(concurvity_lambda).replace(".", "p")
    ckpt_pt   = os.path.join(seed_dir, f"dense_seed{seed}_conc{conc_tag}.pt")
    ckpt_json = os.path.join(seed_dir, f"dense_seed{seed}_conc{conc_tag}.json")

    if os.path.exists(ckpt_pt) and os.path.exists(ckpt_json):
        model.load_state_dict(
            torch.load(ckpt_pt, map_location=DEVICE, weights_only=True)
        )
        with open(ckpt_json, encoding="utf-8") as fj:
            dense_meta = json.load(fj)
        dense_val_balacc = float(dense_meta["dense_val_balacc"])
        dense_r_perp     = float(dense_meta.get("dense_r_perp", float("nan")))
        print(f"  [dense] Loaded cached ckpt (seed={seed}): "
              f"val_balacc={dense_val_balacc:.4f}  r_perp={dense_r_perp:.4f}")
    else:
        print(f"  [dense] Training seed={seed}, condition={condition}, "
              f"λ_c={concurvity_lambda} ...")
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
                          f"best_val_balacc={best_val_balacc:.4f}")
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
                "seed":              seed,
                "timestamp":         datetime.now(timezone.utc).isoformat(),
            }, fj, indent=2)
        print(f"  [dense] Done (seed={seed}).  val_balacc={dense_val_balacc:.4f}  "
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
          f"{lambda_schedule[0]:.3e} → {lambda_schedule[-1]:.3e}  "
          f"(seed={seed})")

    prev_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    prev_norms      = feature_group_norms(model)

    path_rows:           list[dict] = []
    norms_per_step_rows: list[dict] = []
    elim_order:          list[tuple] = []   # (concept_name, step, lambda_s)
    elim_set:            set   = set()
    monotone_n_active:   bool  = True
    prev_n_active:       int   = sum(1 for v in prev_norms.values() if v > ZERO_THRESHOLD)
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

        best_step_val_loss = float("inf")
        best_step_state    = None
        no_improve_ctr     = 0
        actual_epochs      = 0

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
        val_balacc_best, val_r_perp_best, _, val_loss_full_best = _eval_val_full()
        norms    = feature_group_norms(model)
        n_active = sum(1 for v in norms.values() if v > ZERO_THRESHOLD)
        r_sparse = sum(norms.values())
        step_sec = time.time() - t_step

        # Monotonicity check
        if n_active > prev_n_active:
            monotone_n_active = False
            print(f"  ⚠ seed={seed} step {step_idx+1}: n_active INCREASED "
                  f"({prev_n_active} → {n_active}) — re-activation.",
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
                  f"λ={lambda_t:.4e} (seed={seed}). Stopping early.")
            break

    elapsed_path = time.time() - t0_path

    # ── Write per-seed artefacts ──────────────────────────────────────────────
    path_df = pd.DataFrame(path_rows)
    path_df.to_csv(os.path.join(seed_dir, "path.csv"), index=False)

    norms_df = pd.DataFrame(norms_per_step_rows)
    norms_df.to_csv(
        os.path.join(seed_dir, "feature_group_norms_per_step.csv"), index=False
    )

    # Elimination table
    with open(os.path.join(seed_dir, "elimination_order.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"condition={condition}  seed={seed}  lambda_c={concurvity_lambda}\n")
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

    n_steps_run   = len(path_rows)
    first_elim_step  = elim_order[0][1] if elim_order else None
    first_elim_lam   = elim_order[0][2] if elim_order else None
    first_elim_conc  = elim_order[0][0] if elim_order else None
    n_active_final   = path_rows[-1]["n_active"] if path_rows else None
    lambda_final     = path_rows[-1]["lambda_s"] if path_rows else None
    balacc_final     = path_rows[-1]["val_balacc_best_in_step"] if path_rows else None

    return {
        "seed":                      seed,
        "condition":                 condition,
        "concurvity_lambda":         concurvity_lambda,
        "dense_val_balacc":          dense_val_balacc,
        "n_steps_run":               n_steps_run,
        "elapsed_min":               elapsed_path / 60.0,
        "first_elimination_step":    first_elim_step,
        "first_elimination_lambda":  first_elim_lam,
        "first_eliminated_concept":  first_elim_conc,
        "n_active_at_final_step":    n_active_final,
        "n_active_at_step_100":      (path_rows[99]["n_active"]
                                       if len(path_rows) > 99 else n_active_final),
        "lambda_at_final_step":      lambda_final,
        "balacc_at_final_step":      balacc_final,
        "monotone_n_active":         monotone_n_active,
        "elim_order":                elim_order,
        "path_df":                   path_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_aggregate_path(seed_results: list[dict], condition: str) -> pd.DataFrame:
    """Stack per-seed path.csv into one DataFrame with a 'seed' column."""
    rows = []
    for r in seed_results:
        seed   = r["seed"]
        df     = r["path_df"]
        for _, row in df.iterrows():
            rows.append({
                "step":                    int(row["step"]),
                "seed":                    seed,
                "lambda_s":                row["lambda_s"],
                "n_active":                int(row["n_active"]),
                "val_balacc_best_in_step": row["val_balacc_best_in_step"],
                "val_r_perp_best_in_step": row["val_r_perp_best_in_step"],
            })
    return pd.DataFrame(rows).sort_values(["step", "seed"]).reset_index(drop=True)


def build_elimination_summary(
    seed_results: list[dict],
    concept_names: list[str],
    n_seeds: int,
) -> pd.DataFrame:
    """Per-concept elimination statistics across seeds."""
    # elim_steps[concept] = list of steps at which it was eliminated (one per seed)
    elim_steps: dict[str, list[int]] = {c: [] for c in concept_names}
    elim_lams:  dict[str, list[float]] = {c: [] for c in concept_names}

    for r in seed_results:
        eliminated_this_seed = {nm for nm, _, _ in r["elim_order"]}
        for nm, st, lv in r["elim_order"]:
            elim_steps[nm].append(st)
            elim_lams[nm].append(lv)

    rows = []
    for c in concept_names:
        steps = elim_steps[c]
        lams  = elim_lams[c]
        n_elim = len(steps)
        rows.append({
            "concept_name":       c,
            "n_seeds_eliminated": n_elim,
            "mean_elim_step":     float(np.mean(steps)) if steps else float("nan"),
            "std_elim_step":      float(np.std(steps))  if len(steps) > 1 else 0.0,
            "mean_elim_lambda":   float(np.mean(lams))  if lams  else float("nan"),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["n_seeds_eliminated", "mean_elim_step"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sanity_only", action="store_true",
                        help="Run sanity checks then exit without training.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 3 λ-steps × 1 seed (seed=42) per condition.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seed list (default: 42 43 44 45 46).")
    parser.add_argument("--max_lambda_steps", type=int, default=MAX_LAM_STEPS,
                        help=f"Override number of lambda steps (default={MAX_LAM_STEPS}).")
    parser.add_argument("--max_dense_epochs", type=int, default=MAX_DENSE_EPOCHS)
    parser.add_argument("--max_warm_epochs",  type=int, default=MAX_WARM_EPOCHS)
    parser.add_argument("--arch_winner_json", type=str, default=None)
    parser.add_argument("--out_root",         type=str, default=None)
    args = parser.parse_args()

    seeds         = [42] if args.dry_run else (args.seeds or SEEDS)
    max_lam_steps = 3    if args.dry_run else args.max_lambda_steps
    out_root      = args.out_root or OUT_ROOT

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

    os.makedirs(out_root, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    subtype_to_int = load_label_mapping()
    int_to_subtype = {v: k for k, v in subtype_to_int.items()}
    class_names_str = [int_to_subtype[i] for i in range(NUM_CLASSES)]

    data           = load_data(subtype_to_int)
    scores         = data["scores"]
    concept_names  = data["concept_names"]
    labels_all     = data["labels_all"]
    train_pool_idx = data["train_pool_idx"]
    patient_ids    = data["patient_ids"]

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

    # ── Sanity checks 1–3 ─────────────────────────────────────────────────────
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
    print(f"Chest X-ray NAM — Sparsity sweep FULL")
    print(f"  [Issue 4] best-within-step checkpoint restored before λ+1")
    print(f"  [Issue 9] warmup_epochs=0 (concurvity active from epoch 1 when λ_c>0)")
    print(f"  Config: hidden={list(hidden_dims)}, dropout={dropout}, "
          f"wd={weight_decay:.0e}")
    print(f"  Conditions: sparsity_only (λ_c=0.0) → sparsity_conc "
          f"(λ_c={operative_lam_c})")
    print(f"  Seeds: {seeds}  |  max_lambda_steps={max_lam_steps}")
    print(f"  Schedule: λ_0={LAMBDA_0:.3e}, ε={EPSILON}, max_λ={MAX_LAMBDA:.1e}")
    print(f"  Preview λ at steps {[0,25,50,75,max_lam_steps-1]}: "
          f"{[f'{v:.3e}' for v in sched_preview]}")
    print(f"  Per-step budget: max_warm_epochs={args.max_warm_epochs}, "
          f"warm_patience={WARM_PATIENCE}")
    print(f"  val_random_state={VAL_RANDOM_STATE} (identical to STEP 2/3/4/smoke)")
    print(f"  Test set: NOT LOADED")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {out_root}/")
    if args.dry_run:
        print(f"  [DRY RUN — {max_lam_steps} steps, seed={seeds} only]")
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
        "sweep_type":          "full",
        "seeds":               seeds,
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
        "val_split_note":      "GroupShuffleSplit(random_state=42), identical to STEP 2/3/4/smoke",
        "config_id":           arch["config_id"],
        "hidden_dims":         list(hidden_dims),
        "dropout":             dropout,
        "weight_decay":        weight_decay,
        "n_features":          N_FEATURES,
        "num_classes":         NUM_CLASSES,
        "feature_file":        FEATURES_PATH,
        "split_file":          SPLIT_PATH,
        "zero_threshold":      ZERO_THRESHOLD,
        "smoke_test_repro_reference": {
            "sparsity_only_seed42": SMOKE_SO_SEED42,
            "sparsity_conc_seed42": SMOKE_SC_SEED42,
        },
        "test_set_touched":    False,
        "dry_run":             args.dry_run,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "git_commit":          git_hash,
    }
    with open(os.path.join(out_root, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    # ── Main sweep loop ───────────────────────────────────────────────────────
    CONDITION_SPECS = [
        ("sparsity_only", 0.0),
        ("sparsity_conc",  operative_lam_c),
    ]
    all_cond_results: dict[str, list[dict]] = {}
    t_total = time.time()
    repro_mismatch = False   # set to True if seed-42 drift detected

    for condition, lam_c in CONDITION_SPECS:
        print(f"\n{'━'*68}")
        print(f"CONDITION: {condition}  |  λ_c={lam_c}  |  seeds={seeds}")
        print(f"{'━'*68}")

        cond_dir = os.path.join(out_root, condition)
        os.makedirs(cond_dir, exist_ok=True)

        seed_results: list[dict] = []

        for seed in seeds:
            print(f"\n{'─'*60}")
            print(f"  [{condition}  seed={seed}]")
            print(f"{'─'*60}")

            seed_dir = os.path.join(cond_dir, f"seed_{seed}")
            t_seed   = time.time()

            result = run_one_seed(
                seed=seed,
                condition=condition,
                concurvity_lambda=lam_c,
                hidden_dims=hidden_dims,
                dropout=dropout,
                weight_decay=weight_decay,
                concept_names=concept_names,
                X_train_final_raw=X_train_final_raw,
                X_val_raw=X_val_raw,
                y_train_final=y_train_final,
                y_val=y_val,
                seed_dir=seed_dir,
                max_lambda_steps=max_lam_steps,
                max_dense_epochs=args.max_dense_epochs,
                max_warm_epochs=args.max_warm_epochs,
            )
            elapsed_seed = time.time() - t_seed
            result["elapsed_seed_min"] = elapsed_seed / 60.0
            seed_results.append(result)

            # ── Per-seed progress line ─────────────────────────────────────────
            print(
                f"\n  [{condition} seed {seed}] "
                f"first_elim_step={result['first_elimination_step']}  "
                f"n_active@step{result['n_steps_run']}={result['n_active_at_final_step']}  "
                f"val_balacc@final={result['balacc_at_final_step']:.4f}  "
                f"wall_clock={elapsed_seed/60:.1f}min"
            )

            # ── Sanity check 4: smoke-test reproduction (seed 42 only) ────────
            if seed == 42 and not args.dry_run:
                smoke_ref = (SMOKE_SO_SEED42 if condition == "sparsity_only"
                             else SMOKE_SC_SEED42)
                full_fe   = result["first_elimination_step"]
                full_fc   = result["first_eliminated_concept"]
                full_n100 = result["n_active_at_step_100"]
                smoke_fe  = smoke_ref["first_elim_step"]
                smoke_n   = smoke_ref["n_active_at_100"]
                smoke_fc  = smoke_ref.get("first_elim_concept")
                match_fe   = (full_fe  == smoke_fe)
                match_n    = (full_n100 == smoke_n)
                match_fc   = (smoke_fc is None or full_fc == smoke_fc)
                overall_ok = match_fe and match_n and match_fc

                print(f"\n  [CHECK 4] {condition} seed 42 reproduction:")
                print(f"    Smoke test  : first_elim_step={smoke_fe}, "
                      f"first_elim_concept={smoke_fc or 'any'}, "
                      f"n_active@100={smoke_n}")
                print(f"    Full sweep  : first_elim_step={full_fe}, "
                      f"first_elim_concept={full_fc}, "
                      f"n_active@100={full_n100}")
                print(f"    Match       : {'✓' if overall_ok else '✗'}")

                if not overall_ok:
                    print(f"\n  [CHECK 4] FAIL — code drift between smoke and full sweep!")
                    print(f"    first_elim_step match : {'✓' if match_fe else f'✗  ({full_fe} vs {smoke_fe})'}")
                    print(f"    n_active@100    match : {'✓' if match_n  else f'✗  ({full_n100} vs {smoke_n})'}")
                    if smoke_fc:
                        print(f"    first_elim_conc match : {'✓' if match_fc else f'✗  ({full_fc} vs {smoke_fc})'}")
                    print(f"\n  Stopping sweep. Inspect for source of drift before re-running.")
                    repro_mismatch = True
                    break   # stop seeds for this condition
                else:
                    print(f"    [CHECK 4] ✓ Full sweep seed 42 reproduces smoke test exactly.")

            # Incremental aggregate_path flush (crash-safe)
            agg_df_partial = build_aggregate_path(seed_results, condition)
            agg_df_partial.to_csv(
                os.path.join(cond_dir, "aggregate_path.csv"), index=False
            )

        if repro_mismatch:
            print(f"\n  Halting at condition '{condition}' due to reproduction mismatch.")
            break

        # ── Per-condition aggregate artefacts ──────────────────────────────────
        all_cond_results[condition] = seed_results

        agg_df = build_aggregate_path(seed_results, condition)
        agg_df.to_csv(os.path.join(cond_dir, "aggregate_path.csv"), index=False)

        elim_sum_df = build_elimination_summary(
            seed_results, concept_names, n_seeds=len(seeds)
        )
        elim_sum_df.to_csv(
            os.path.join(cond_dir, "elimination_summary.csv"), index=False
        )

        # ── Print per-condition elimination summary ────────────────────────────
        print(f"\n  Elimination summary [{condition}]  "
              f"({len(seeds)} seeds, {max_lam_steps} steps each):")
        print(f"  {'concept':35s}  {'n_elim':>6}  {'mean_step':>9}  "
              f"{'std_step':>8}  {'mean_lam':>10}")
        print(f"  {'-'*35}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*10}")
        for _, row in elim_sum_df.iterrows():
            n_e = int(row["n_seeds_eliminated"])
            ms  = f"{row['mean_elim_step']:.1f}" if not np.isnan(row["mean_elim_step"]) else "—"
            ss  = f"{row['std_elim_step']:.1f}"  if not np.isnan(row["std_elim_step"])  else "—"
            ml  = f"{row['mean_elim_lambda']:.3e}" if not np.isnan(row["mean_elim_lambda"]) else "—"
            marker = "★" if n_e == len(seeds) else (" " if n_e > 0 else "○")
            print(f"  {marker} {row['concept_name']:33s}  {n_e:>6}  {ms:>9}  "
                  f"{ss:>8}  {ml:>10}")
        print(f"  (★ = eliminated in all {len(seeds)} seeds;  "
              f"○ = never eliminated)")

    # ── Bail if reproduction mismatch ─────────────────────────────────────────
    if repro_mismatch:
        print(f"\n{'='*68}")
        print(f"SWEEP ABORTED — smoke-test reproduction check failed.")
        print(f"No summary.json or STEP_5_COMPLETE.flag written.")
        print(f"{'='*68}")
        sys.exit(1)

    # ── summary.json ──────────────────────────────────────────────────────────
    def _cond_json(cond: str, seed_results: list[dict]) -> dict:
        na_final = [r["n_active_at_final_step"] for r in seed_results]
        na_100   = [r["n_active_at_step_100"]   for r in seed_results]
        fe_steps = [r["first_elimination_step"]  for r in seed_results]
        # replace None with nan for stats
        fe_arr   = np.array([s if s is not None else np.nan for s in fe_steps],
                            dtype=float)
        na_arr   = np.array([s if s is not None else np.nan for s in na_final],
                            dtype=float)
        return {
            "n_active_at_final_step_per_seed":  [int(v) if v is not None else None
                                                  for v in na_final],
            "n_active_at_step_100_per_seed":    [int(v) if v is not None else None
                                                  for v in na_100],
            "first_elim_step_per_seed":         fe_steps,
            "mean_n_active_at_final_step":      float(np.nanmean(na_arr)),
            "std_n_active_at_final_step":        float(np.nanstd(na_arr)),
            "mean_first_elim_step":             float(np.nanmean(fe_arr)),
            "std_first_elim_step":              float(np.nanstd(fe_arr)),
            "seeds_completed":                  [r["seed"] for r in seed_results],
            "dense_val_balacc_per_seed":        [round(r["dense_val_balacc"], 4)
                                                  for r in seed_results],
        }

    summary: dict = {}
    for cond, seed_results in all_cond_results.items():
        summary[cond] = _cond_json(cond, seed_results)
    summary["total_elapsed_min"] = round((time.time() - t_total) / 60.0, 1)
    summary["schedule_params"]   = {
        "lambda_0": LAMBDA_0, "epsilon": EPSILON,
        "max_lambda_steps": max_lam_steps,
        "max_warm_epochs": args.max_warm_epochs,
        "warm_patience": WARM_PATIENCE,
    }
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── STEP_5_COMPLETE.flag ──────────────────────────────────────────────────
    if not args.dry_run:
        with open(os.path.join(out_root, "STEP_5_COMPLETE.flag"), "w",
                  encoding="utf-8") as f:
            f.write(f"STEP_5 completed at {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"seeds={seeds}\nconditions={[s for s in all_cond_results]}\n")
            f.write(f"max_lambda_steps={max_lam_steps}\n")

    # ── Final report ──────────────────────────────────────────────────────────
    total_min = (time.time() - t_total) / 60.0
    print(f"\n{'='*68}")
    print(f"Chest X-ray NAM — Sparsity sweep FULL — COMPLETE")
    print(f"")
    print(f"  {'Condition':<16}  {'Seed':>4}  {'first_elim':>10}  "
          f"{'n_active@{}'.format(max_lam_steps):>13}  {'val_balacc@final':>16}")
    print(f"  {'-'*16}  {'-'*4}  {'-'*10}  {'-'*13}  {'-'*16}")
    for cond, seed_results in all_cond_results.items():
        for r in seed_results:
            fe  = str(r["first_elimination_step"]) if r["first_elimination_step"] else "none"
            naf = str(r["n_active_at_final_step"]) if r["n_active_at_final_step"] is not None else "?"
            ba  = f"{r['balacc_at_final_step']:.4f}" if r["balacc_at_final_step"] is not None else "?"
            print(f"  {cond:<16}  {r['seed']:>4}  {fe:>10}  {naf:>13}  {ba:>16}")
    print(f"")

    # Seed variance (meaningful signal for stability)
    for cond, seed_results in all_cond_results.items():
        fe_arr = np.array([r["first_elimination_step"] or np.nan for r in seed_results],
                          dtype=float)
        na_arr = np.array([r["n_active_at_final_step"] or np.nan for r in seed_results],
                          dtype=float)
        print(f"  {cond}: std(first_elim_step)={np.nanstd(fe_arr):.1f}  "
              f"std(n_active@{max_lam_steps})={np.nanstd(na_arr):.1f}")
    print(f"")
    print(f"  Total elapsed: {total_min:.1f} min")
    print(f"  Output dir   : {out_root}/")
    print(f"  summary.json → {os.path.join(out_root, 'summary.json')}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
