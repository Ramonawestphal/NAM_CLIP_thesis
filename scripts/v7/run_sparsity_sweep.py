"""
Warm-start Group LASSO regularization path — v7 corrected pipeline (STEP 5).
Runs one CONDITION at a time (sparsity_only or sparsity_concurvity) across all
seeds.  Invoke twice (once per condition) for the full overnight run.

Fixes applied
─────────────
Issue 4 : Best-within-step checkpoint: at each lambda step the best-val-loss
          state is tracked in memory and restored BEFORE advancing to lambda+1.
Issue 9 : warmup_epochs=0 in the dense phase (Setting A, diagnostic confirmed;
          see results/v7/diagnostic_warmup/comparison.md).
Issue 3 : set_all_seeds() includes random.seed.
Issue 7 : per-seed scaler saved to {condition}/seed_{N}/scaler.pkl.
Issue 8 : CUDA determinism flags set.

Lambda schedule (v3 — starts in the sparsity-active range)
────────────────────────────────────────────────────────────
  lambda_t = lambda_0 * (1 + epsilon)^t
  Default: lambda_0=1.0, epsilon=0.04, max_lambda_steps=150
  → Step   1: lambda = 1.0
  → Step  50: lambda ≈ 7.1
  → Step 100: lambda ≈ 50.5
  → Step 150: lambda ≈ 359

  Previous schedules (deprecated):
  v0: lambda_0=1e-3, epsilon=0.02,  300 steps → max lambda≈0.40 (no sparsity)
  v2: lambda_0=1e-3, epsilon=0.025, 500 steps → max lambda≈234  (no sparsity in
      practice because Adam growth dominates the proximal step at small lambda)
  Root cause: shrinkage = 1 - lr*lambda/norm ≈ 1 - 1.7e-6 per batch when
  lambda=0.012; the proximal step was mathematically unable to compete with Adam.

Conditions
──────────
  sparsity_only        : lambda_c=0.0 (no concurvity), lambda_s swept
  sparsity_concurvity  : lambda_c=3.0 (from concurvity_sweep/winner.json), lambda_s swept

Output tree
───────────
  results/v7/sparsity_sweep/
    config.json                        ← schedule params saved at startup
    {condition}/
      seed_{N}/
        scaler.pkl
        dense_seed{N}_conc{tag}_ep{M}.pt
        dense_seed{N}_conc{tag}_ep{M}.json
        checkpoints/
          seed{N}_lambda{val:.6e}.pt
          seed{N}_lambda{val:.6e}.json
      path_seed{N}.csv                 ← written after each seed
      path_seed{N}_elimination.txt
      condition_summary.md             ← written after all seeds for this condition

Usage (from project root)
──────────────────────────
  # Smoke test (1 seed, sparsity_only, 100 steps):
  python scripts/v7/run_sparsity_sweep.py --condition sparsity_only --seeds 42 --max_lambda_steps 100

  # Full overnight run (5 seeds, sparsity_only):
  python scripts/v7/run_sparsity_sweep.py --condition sparsity_only

  # Full overnight run (5 seeds, sparsity_concurvity):
  python scripts/v7/run_sparsity_sweep.py --condition sparsity_concurvity
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import pickle
import re
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
from torch.utils.data import DataLoader, TensorDataset

from scripts.v7._common import (
    FEATURES_PATH, SPLITS_PATH, N_FEATURES, N_CLASSES,
    set_all_seeds, load_raw_data, make_fixed_val_split, standardize,
    class_weight_tensor, make_model, write_step_flag,
)
from src.models.concurvity import multiclass_concurvity
from src.models.sparsity import (
    group_lasso_penalty, feature_group_norms, apply_proximal_step
)

# ── Constants ─────────────────────────────────────────────────────────────────
SEEDS = [42, 43, 44, 45, 46]

LR         = 1e-3
BATCH_SIZE = 256

MAX_DENSE_EPOCHS = 100
DENSE_PATIENCE   = 15
SCHED_PAT        = 5
SCHED_FAC        = 0.5

# Per-step fine-tuning budget (reduced to fit 12-hour overnight run)
MAX_WARM_EPOCHS = 30   # was 50 in broken v0 run
WARM_PATIENCE   = 6    # was 10; tighter early stopping per step
WARM_MIN_DELTA  = 1e-4

# Lambda schedule v3: start in the sparsity-active range.
# 1.0 * 1.04^149 ≈ 359   (previous: 1e-3 * 1.025^499 ≈ 234 but proximal was
# overwhelmed by Adam at small lambda — see DEPRECATED_short_schedule/README.md)
LAMBDA_0      = 1.0    # was 1e-3; start where sparsity is actually possible
EPSILON       = 0.04   # was 0.025; faster geometric ramp
MAX_LAMBDA    = 1e3
MAX_LAM_STEPS = 150    # was 500; budget concentrated in active range

CONVERGENCE_THRESHOLD = 0.50
CONVERGENCE_EPOCH     = 30

# Per-(condition, seed) budget — exceeding this stops that seed and reports
BUDGET_HOURS = 2.0

WINNER_JSON            = "results/v7/architecture_search_cv/winner.json"
CONCURVITY_WINNER_JSON = "results/v7/concurvity_sweep/winner.json"
OUT_ROOT               = "results/v7/sparsity_sweep"
RESULTS_V7             = "results/v7"
STEP_N                 = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin_memory)


# ── Core: one (seed, condition) run ──────────────────────────────────────────

def run_one_seed(
    *,
    seed:              int,
    condition:         str,
    hidden_dims:       tuple,
    dropout:           float,
    weight_decay:      float,
    concurvity_lambda: float,
    raw:               dict,
    out_dir:           str,
    lambda_0:          float = LAMBDA_0,
    epsilon:           float = EPSILON,
    max_lambda:        float = MAX_LAMBDA,
    max_lambda_steps:  int   = MAX_LAM_STEPS,
    max_dense_epochs:  int   = MAX_DENSE_EPOCHS,
    dense_patience:    int   = DENSE_PATIENCE,
    max_warm_epochs:   int   = MAX_WARM_EPOCHS,
    warm_patience:     int   = WARM_PATIENCE,
    warm_min_delta:    float = WARM_MIN_DELTA,
    skip_convergence_check: bool  = False,
    budget_hours:      float = BUDGET_HOURS,
) -> dict:
    """Train dense checkpoint + traverse warm-start sparsity path for one seed.

    Writes path_seed{seed}.csv immediately on completion (so a crash on a later
    seed does not lose earlier work).

    Issue 4 fix: within each lambda step, track best-within-step state dict and
    restore it BEFORE advancing to the next lambda.
    Issue 9 fix: warmup_dense=0 (Setting A confirmed by diagnostic experiment).
    """
    set_all_seeds(seed)

    # ── Unpack raw data ───────────────────────────────────────────────────────
    scores        = raw["scores"]
    labels        = raw["labels"]
    lesion_ids    = raw["lesion_ids"]
    concept_names = raw["concept_names"]
    class_names   = raw["class_names"]
    train_idx     = raw["train_idx"]

    X_all_train      = scores[train_idx]
    y_all_train      = labels[train_idx]
    lesion_ids_train = lesion_ids[train_idx]

    # Fixed val split: val_random_state=42 across ALL seeds for comparability
    # with STEP 2 (plain_nam) and STEP 4 (concurvity_only). Previous run used
    # val_random_state=seed — those results are in DEPRECATED_per_seed_val/.
    val_split = make_fixed_val_split(
        X_all_train, y_all_train, lesion_ids_train, class_names,
        val_random_state=42,
    )

    # ── Setup directories ─────────────────────────────────────────────────────
    seed_dir = os.path.join(out_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(os.path.join(seed_dir, "checkpoints"), exist_ok=True)

    # Per-seed status flags (crash visibility)
    running_flag  = os.path.join(seed_dir, "RUNNING.flag")
    complete_flag = os.path.join(seed_dir, "COMPLETE.flag")
    for _f in [running_flag, complete_flag]:
        if os.path.exists(_f):
            os.remove(_f)
    with open(running_flag, "w") as _f:
        _f.write(f"condition={condition} seed={seed} pid={os.getpid()}\n")

    pin_memory = (DEVICE.type == "cuda")

    # ── Standardize (per-seed scaler — Issue 7 fix) ───────────────────────────
    X_tr_sc, X_val_sc, _, scaler = standardize(
        val_split["X_train"], val_split["X_val"]
    )
    with open(os.path.join(seed_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    y_tr_str  = val_split["y_train_str"]
    y_tr_enc  = val_split["y_train_enc"]
    y_val_enc = val_split["y_val_enc"]

    w_tensor  = class_weight_tensor(y_tr_str, class_names, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)

    X_val_t = torch.tensor(X_val_sc, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val_enc, dtype=torch.long,   device=DEVICE)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc,  dtype=torch.float32),
        torch.tensor(y_tr_enc, dtype=torch.long),
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = make_model(hidden_dims, dropout, concept_names, DEVICE)

    def _eval_val() -> tuple[float, float, float]:
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        model.eval()
        with torch.no_grad():
            logits, shape_outs = model(X_val_t, return_shape_outputs=True)
            val_loss_ce = criterion(logits, y_val_t).item()
            preds       = logits.argmax(dim=1).cpu().numpy()
            r_perp      = multiclass_concurvity(shape_outs).item()
        balacc = balanced_accuracy_score(y_val_enc, preds)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
        try:
            auc = roc_auc_score(
                y_val_enc, proba, multi_class="ovr",
                average="weighted", labels=list(range(len(class_names)))
            )
        except ValueError:
            auc = float("nan")
        val_loss_full = val_loss_ce + concurvity_lambda * r_perp
        return balacc, auc, val_loss_full

    def _eval_train_loss() -> float:
        """CE loss on training set (no gradients; used for path CSV only)."""
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in _make_loader(train_ds, BATCH_SIZE * 4, False, pin_memory):
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                total += criterion(model(X_b), y_b).item() * len(y_b)
                n     += len(y_b)
        return total / n if n > 0 else float("nan")

    # ── Phase 1: Dense checkpoint (with caching) ──────────────────────────────
    conc_tag  = str(concurvity_lambda).replace(".", "p")
    ckpt_stem = f"dense_seed{seed}_conc{conc_tag}_ep{max_dense_epochs}"
    ckpt_pt   = os.path.join(seed_dir, f"{ckpt_stem}.pt")
    ckpt_json = os.path.join(seed_dir, f"{ckpt_stem}.json")

    if os.path.exists(ckpt_pt) and os.path.exists(ckpt_json):
        model.load_state_dict(
            torch.load(ckpt_pt, map_location=DEVICE, weights_only=True)
        )
        with open(ckpt_json) as fj:
            dense_meta = json.load(fj)
        dense_val_balacc = dense_meta["dense_val_balacc"]
        dense_val_auc    = dense_meta["dense_val_auc"]
        print(f"  [dense] Cached checkpoint: val_balacc={dense_val_balacc:.4f}  "
              f"val_auc={dense_val_auc:.4f}")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=SCHED_FAC,
            patience=SCHED_PAT, min_lr=1e-6,
        )
        best_val_balacc   = -1.0
        patience_ctr      = 0
        best_dense_state  = None
        reached_threshold = False

        print(f"  [dense] Training (seed={seed}, lambda_c={concurvity_lambda}) ...")
        for epoch in range(max_dense_epochs):
            model.train()
            for X_b, y_b in _make_loader(train_ds, BATCH_SIZE, True, pin_memory):
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

            val_balacc, _, _ = _eval_val()
            scheduler.step(val_balacc)

            if epoch < CONVERGENCE_EPOCH and val_balacc >= CONVERGENCE_THRESHOLD:
                reached_threshold = True

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

        if not skip_convergence_check and not reached_threshold:
            raise RuntimeError(
                f"Dense model (seed={seed}) did not reach val_balacc >= "
                f"{CONVERGENCE_THRESHOLD} within {CONVERGENCE_EPOCH} epochs."
            )

        if best_dense_state is not None:
            model.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_dense_state.items()}
            )
        elif os.path.exists(ckpt_pt):
            model.load_state_dict(
                torch.load(ckpt_pt, map_location=DEVICE, weights_only=True)
            )

        dense_val_balacc, dense_val_auc, _ = _eval_val()
        torch.save(model.state_dict(), ckpt_pt)

        json.dump(
            {
                "dense_val_balacc":  dense_val_balacc,
                "dense_val_auc":     dense_val_auc,
                "hidden_dims":       list(hidden_dims),
                "dropout":           dropout,
                "weight_decay":      weight_decay,
                "concurvity_lambda": concurvity_lambda,
                "warmup_dense":      0,
                "seed":              seed,
                "timestamp":         datetime.now(timezone.utc).isoformat(),
            },
            open(ckpt_json, "w"),
            indent=2,
        )
        print(f"  [dense] Done.  val_balacc={dense_val_balacc:.4f}  "
              f"val_auc={dense_val_auc:.4f}")

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
          f"{lambda_schedule[0]:.3e} --> {lambda_schedule[-1]:.3e}")

    prev_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    prev_norms      = feature_group_norms(model)

    rows:         list[dict]  = []
    elim_order:   list[tuple] = []
    selected_lams: dict       = {}
    t0_path = time.time()

    # ── Incremental CSV: open once, write one row per step for crash safety ────
    csv_path = os.path.join(out_dir, f"path_seed{seed}.csv")
    _csv_cols = (
        ["step", "lambda_s", "n_active", "val_balacc", "val_auc",
         "val_loss", "train_loss", "epochs_used", "time_sec",
         "concepts_just_eliminated"]
        + [f"norm_{k}" for k in concept_names]
    )
    _csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
    _csv_fh.write(f"# condition={condition}  seed={seed}  "
                  f"dense_val_balacc={dense_val_balacc:.6f}\n")
    _csv_writer = csv.DictWriter(_csv_fh, fieldnames=_csv_cols)
    _csv_writer.writeheader()

    # ── Phase 3: Path loop ────────────────────────────────────────────────────
    for step_idx, lambda_t in enumerate(lambda_schedule):
        t_step = time.time()

        # Budget guard
        elapsed_h = (t_step - t0_path) / 3600.0
        if elapsed_h > budget_hours:
            print(
                f"\n  [BUDGET] Seed {seed} exceeded {budget_hours:.1f}-hour budget "
                f"at step {step_idx+1} (lambda={lambda_t:.4e}). "
                f"Stopping this seed. Partial CSV already written."
            )
            break

        # Warm-start: load previous step's best weights (Issue 4 fix)
        model.load_state_dict({k: v.to(DEVICE) for k, v in prev_state_dict.items()})

        # Fresh optimizer + scheduler per step
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHED_FAC,
            patience=SCHED_PAT, min_lr=1e-6,
        )

        best_step_val_loss = float("inf")
        best_step_state    = None
        no_improve_ctr     = 0
        actual_epochs      = 0

        for epoch in range(max_warm_epochs):
            model.train()
            for X_b, y_b in _make_loader(train_ds, BATCH_SIZE, True, pin_memory):
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
                # Proximal operator after each batch (Issue 4 uses best-within-step)
                apply_proximal_step(
                    model,
                    lr=optimizer.param_groups[0]["lr"],
                    sparsity_lambda=lambda_t,
                )

            actual_epochs += 1
            val_balacc, val_auc, val_loss = _eval_val()
            scheduler.step(val_loss)

            # Issue 4 fix: track best-within-step state in memory
            if val_loss < best_step_val_loss - warm_min_delta:
                best_step_val_loss = val_loss
                no_improve_ctr     = 0
                best_step_state    = {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}
            else:
                no_improve_ctr += 1
                if no_improve_ctr >= warm_patience:
                    break

        # Issue 4 fix: restore best-within-step state before recording / advancing
        if best_step_state is not None:
            model.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_step_state.items()}
            )

        # Evaluate at best-within-step state
        val_balacc_best, val_auc_best, val_loss_best = _eval_val()
        train_loss_best = _eval_train_loss()
        norms    = feature_group_norms(model)
        n_active = sum(1 for v in norms.values() if v > 1e-8)
        step_sec = time.time() - t_step

        # Detect eliminations
        just_eliminated: list[str] = []
        for k in concept_names:
            if prev_norms[k] >= 1e-8 and norms[k] < 1e-8:
                just_eliminated.append(k)
                elim_order.append((k, lambda_t))
                selected_lams[k] = lambda_t

        # Warn on unexpected re-activations
        for k in concept_names:
            if prev_norms[k] < 1e-8 and norms[k] >= 1e-8:
                print(
                    f"  WARNING: concept '{k}' re-activated at "
                    f"lambda={lambda_t:.4e}",
                    file=sys.stderr,
                )

        # Per-step checkpoint
        lam_tag   = f"{lambda_t:.6e}"
        step_pt   = os.path.join(
            seed_dir, "checkpoints", f"seed{seed}_lambda{lam_tag}.pt"
        )
        step_json = os.path.join(
            seed_dir, "checkpoints", f"seed{seed}_lambda{lam_tag}.json"
        )
        torch.save(model.state_dict(), step_pt)
        json.dump(
            {
                "step":            step_idx + 1,
                "lambda_s":        lambda_t,
                "n_active":        n_active,
                "val_balacc":      val_balacc_best,
                "val_auc":         val_auc_best,
                "val_loss":        val_loss_best,
                "train_loss":      train_loss_best,
                "epochs_used":     actual_epochs,
                "time_sec":        step_sec,
                "just_eliminated": just_eliminated,
                "issue4_fix":      "best_within_step_checkpoint_restored",
                "norms":           norms,
            },
            open(step_json, "w"),
            indent=2,
        )

        rows.append({
            "step":                     step_idx + 1,
            "lambda_s":                 lambda_t,
            "n_active":                 n_active,
            "val_balacc":               val_balacc_best,
            "val_auc":                  val_auc_best,
            "val_loss":                 val_loss_best,
            "train_loss":               train_loss_best,
            "epochs_used":              actual_epochs,
            "time_sec":                 step_sec,
            "concepts_just_eliminated": ",".join(just_eliminated),
            **{f"norm_{k}": norms[k] for k in concept_names},
        })
        # Incremental CSV flush: survives crashes mid-seed
        _csv_writer.writerow(rows[-1])
        _csv_fh.flush()

        # Console progress: every 10 steps or on elimination
        if just_eliminated or (step_idx + 1) % 10 == 0:
            elim_str = (f"  ELIM: {just_eliminated}" if just_eliminated else "")
            print(
                f"  step {step_idx+1:3d}  lambda={lambda_t:.4e}  "
                f"n_active={n_active:2d}  val_balacc={val_balacc_best:.4f}  "
                f"val_auc={val_auc_best:.4f}  ep={actual_epochs}  "
                f"t={step_sec:.1f}s{elim_str}"
            )

        # Advance: carry BEST state to next step (Issue 4 fix)
        prev_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        prev_norms      = norms

        if n_active == 0:
            print(f"  [path]  All {N_FEATURES} subnets zeroed at "
                  f"lambda={lambda_t:.4e}. Stopping.")
            break

    # ── Close incremental CSV; do a clean final rewrite via DataFrame ──────────
    _csv_fh.close()
    path_df  = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"path_seed{seed}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# condition={condition}  seed={seed}  "
                f"dense_val_balacc={dense_val_balacc:.6f}\n")
        path_df.to_csv(f, index=False)

    # ── Write elimination table ───────────────────────────────────────────────
    elim_path = os.path.join(out_dir, f"path_seed{seed}_elimination.txt")
    with open(elim_path, "w", encoding="utf-8") as f:
        f.write(f"condition={condition}  seed={seed}\n")
        f.write(f"dense_val_balacc={dense_val_balacc:.4f}  "
                f"dense_val_auc={dense_val_auc:.4f}\n\n")
        f.write(f"{'order':>5}  {'lambda_s':>12}  concept\n")
        f.write("-" * 42 + "\n")
        for i, (name, lam) in enumerate(elim_order, 1):
            f.write(f"{i:>5}  {lam:>12.4e}  {name}\n")
        never_zeroed = [k for k in concept_names if k not in selected_lams]
        if never_zeroed:
            f.write(f"\nNever zeroed (lambda <= {max_lambda:.2e}):\n")
            for k in never_zeroed:
                f.write(f"  {k}\n")

    elapsed_min = (time.time() - t0_path) / 60.0
    avg_sec = np.mean([r["time_sec"] for r in rows]) if rows else 0.0
    print(f"  [seed {seed}] {len(rows)} steps in {elapsed_min:.1f} min "
          f"(avg {avg_sec:.1f} s/step).  CSV: {csv_path}")

    # Mark seed complete
    if os.path.exists(running_flag):
        os.remove(running_flag)
    with open(complete_flag, "w") as _f:
        _f.write(f"condition={condition} seed={seed} "
                 f"steps={len(rows)} elapsed_min={elapsed_min:.1f}\n")

    return {
        "seed":              seed,
        "n_steps_run":       len(rows),
        "dense_val_balacc":  dense_val_balacc,
        "dense_val_auc":     dense_val_auc,
        "path_csv":          csv_path,
        "elim_order":        elim_order,
        "elapsed_min":       elapsed_min,
        "avg_sec_per_step":  avg_sec,
    }


# ── PART 5: Post-run summary ──────────────────────────────────────────────────

def summarize_condition(
    condition: str,
    out_dir:   str,
    concept_names: list,
    seeds:     list,
) -> None:
    """Write condition_summary.md after all seeds complete (PART 5).

    Reports sparsity onset, mean/std of val_balacc and n_active at key lambda
    values, and candidate operating-point boundary.  Does NOT auto-select the
    operating point — that is STEP 6.
    """
    all_dfs: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        csv_path = os.path.join(out_dir, f"path_seed{seed}.csv")
        if not os.path.exists(csv_path):
            print(f"  [summary] WARNING: {csv_path} not found; skipping seed {seed}.")
            continue
        df = pd.read_csv(csv_path, comment="#")
        all_dfs[seed] = df

    if not all_dfs:
        print(f"  [summary] No CSV files for condition={condition}. Aborting summary.")
        return

    # ── Recover dense_val_balacc from CSV header ──────────────────────────────
    first_seed   = list(all_dfs.keys())[0]
    first_csv    = os.path.join(out_dir, f"path_seed{first_seed}.csv")
    dense_balacc = None
    with open(first_csv, encoding="utf-8") as f:
        m = re.search(r"dense_val_balacc=([0-9.]+)", f.readline())
    if m:
        dense_balacc = float(m.group(1))
    threshold = (dense_balacc - 0.02) if dense_balacc is not None else None

    # ── Sparsity onset: smallest lambda where any seed's n_active < N_FEATURES ─
    sparsity_onset_lambda = None
    for df in all_dfs.values():
        dropped = df[df["n_active"] < N_FEATURES]
        if not dropped.empty:
            lam = float(dropped["lambda_s"].min())
            if sparsity_onset_lambda is None or lam < sparsity_onset_lambda:
                sparsity_onset_lambda = lam

    # ── Align seeds by step index ─────────────────────────────────────────────
    min_steps = min(len(df) for df in all_dfs.values())
    summary_rows = []
    for i in range(min_steps):
        vals_n   = [df.iloc[i]["n_active"]    for df in all_dfs.values()]
        vals_acc = [df.iloc[i]["val_balacc"]  for df in all_dfs.values()]
        lam_val  = float(list(all_dfs.values())[0].iloc[i]["lambda_s"])
        summary_rows.append({
            "step":             i + 1,
            "lambda_s":         lam_val,
            "mean_n_active":    float(np.mean(vals_n)),
            "std_n_active":     float(np.std(vals_n,   ddof=1)) if len(vals_n) > 1 else 0.0,
            "mean_val_balacc":  float(np.mean(vals_acc)),
            "std_val_balacc":   float(np.std(vals_acc, ddof=1)) if len(vals_acc) > 1 else 0.0,
        })

    # ── Candidate operating-point boundary (Rule A) ───────────────────────────
    # Largest step where mean_val_balacc >= threshold
    last_above_threshold = None
    if threshold is not None:
        for row in summary_rows:
            if row["mean_val_balacc"] >= threshold:
                last_above_threshold = row

    # ── Write markdown ────────────────────────────────────────────────────────
    md_lines = [
        f"# Sparsity Sweep Condition Summary — {condition}",
        f"",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"**Seeds:** {list(all_dfs.keys())}",
        f"**Steps per seed (min):** {min_steps}",
        f"",
        f"## Dense baseline",
        f"",
        f"| metric | value |",
        f"|--------|-------|",
        f"| dense_val_balacc (seed={first_seed}) | {dense_balacc:.4f} |" if dense_balacc else "| dense_val_balacc | N/A |",
        f"| Rule A threshold (−0.02) | {threshold:.4f} |" if threshold else "| threshold | N/A |",
        f"",
        f"## Sparsity onset",
        f"",
    ]
    if sparsity_onset_lambda is not None:
        md_lines.append(
            f"First feature zeroed at **lambda_s ≈ {sparsity_onset_lambda:.4e}**"
        )
    else:
        md_lines += [
            "**No features zeroed.** The schedule did not reach the sparsity-inducing range.",
            "",
            "> Note: if this was a smoke test (100 steps, lambda_max ≈ 0.012), this is",
            "> expected — the sparsity-inducing range is lambda_s >> 1. Re-run with the",
            "> full 500-step schedule to traverse the relevant range.",
        ]

    md_lines += [
        f"",
        f"## Rule A candidate operating point",
        f"",
    ]
    if last_above_threshold is not None:
        md_lines += [
            f"Largest mean_val_balacc >= {threshold:.4f} at:",
            f"  step={last_above_threshold['step']},  "
            f"lambda_s={last_above_threshold['lambda_s']:.4e},  "
            f"mean_val_balacc={last_above_threshold['mean_val_balacc']:.4f},  "
            f"mean_n_active={last_above_threshold['mean_n_active']:.1f}",
            f"",
            f"**Do NOT use this as the operating point yet** — STEP 6 applies",
            f"Rule A formally and verifies the candidate.",
        ]
    else:
        md_lines.append(
            "Mean val_balacc stayed above threshold throughout — the performance "
            "boundary was not reached. If the sweep is complete (500 steps), "
            "there may be no meaningful sparsity-accuracy trade-off at this dataset."
        )

    md_lines += [
        f"",
        f"## Step-level summary (mean ± std across seeds, every 50 steps)",
        f"",
        f"| step | lambda_s | n_active (mean±std) | val_balacc (mean±std) |",
        f"|------|----------|---------------------|-----------------------|",
    ]
    for row in summary_rows:
        if row["step"] == 1 or row["step"] % 50 == 0:
            md_lines.append(
                f"| {row['step']:4d} "
                f"| {row['lambda_s']:.4e} "
                f"| {row['mean_n_active']:.1f} +/- {row['std_n_active']:.2f} "
                f"| {row['mean_val_balacc']:.4f} +/- {row['std_val_balacc']:.4f} |"
            )

    md_lines += [
        f"",
        f"---",
        f"*Full data in path_seedN.csv.  Operating point selection: STEP 6.*",
    ]

    md_path = os.path.join(out_dir, "condition_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  [summary] Written: {md_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--condition", type=str, default="sparsity_only",
        choices=["sparsity_only", "sparsity_concurvity"],
        help="Which condition to run. Default: sparsity_only.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Seeds to run. Default: all 5 (42 43 44 45 46).",
    )
    parser.add_argument(
        "--concurvity_lambda", type=float, default=None,
        help="Override concurvity lambda (sparsity_concurvity only). "
             "Default: read from concurvity_sweep/winner.json.",
    )
    parser.add_argument("--lambda_0",          type=float, default=LAMBDA_0)
    parser.add_argument("--epsilon",           type=float, default=EPSILON)
    parser.add_argument("--max_lambda",        type=float, default=MAX_LAMBDA)
    parser.add_argument("--max_lambda_steps",  type=int,   default=MAX_LAM_STEPS)
    parser.add_argument("--max_dense_epochs",  type=int,   default=MAX_DENSE_EPOCHS)
    parser.add_argument("--max_warm_epochs",   type=int,   default=MAX_WARM_EPOCHS)
    parser.add_argument("--warm_patience",     type=int,   default=WARM_PATIENCE)
    parser.add_argument("--budget_hours",      type=float, default=BUDGET_HOURS)
    parser.add_argument("--skip_convergence_check", action="store_true")
    parser.add_argument("--winner_json",       type=str,   default=None)
    parser.add_argument("--out_root",          type=str,   default=None)
    parser.add_argument("--warmup_epochs",     type=int,   default=0,
                        help="Dense-phase concurvity warm-up (default=0, Setting A).")
    args = parser.parse_args()

    seeds     = args.seeds if args.seeds is not None else SEEDS
    condition = args.condition

    # ── Resolve concurvity_lambda ─────────────────────────────────────────────
    if condition == "sparsity_only":
        concurvity_lambda = 0.0
        print(f"[config] condition=sparsity_only -> concurvity_lambda=0.0")
    elif args.concurvity_lambda is not None:
        concurvity_lambda = args.concurvity_lambda
        print(f"[config] concurvity_lambda={concurvity_lambda} (CLI override)")
    elif os.path.exists(CONCURVITY_WINNER_JSON):
        with open(CONCURVITY_WINNER_JSON) as _f:
            _cw = json.load(_f)
        concurvity_lambda = float(_cw["best_concurvity_lambda"])
        print(f"[config] concurvity_lambda={concurvity_lambda} "
              f"(auto-resolved from {CONCURVITY_WINNER_JSON})")
    else:
        concurvity_lambda = 1.0
        print(f"[config] WARNING: {CONCURVITY_WINNER_JSON} not found; "
              f"fallback concurvity_lambda={concurvity_lambda}")

    # ── Architecture winner ───────────────────────────────────────────────────
    winner_json_path = args.winner_json or WINNER_JSON
    if not os.path.exists(winner_json_path):
        raise FileNotFoundError(
            f"winner.json not found: {winner_json_path}. Run STEP 1 first."
        )
    with open(winner_json_path) as f:
        winner = json.load(f)
    hidden_dims  = tuple(winner["hidden_dims"])
    dropout      = float(winner["dropout"])
    weight_decay = float(winner["weight_decay"])

    # ── Output directories ────────────────────────────────────────────────────
    out_root     = args.out_root or OUT_ROOT
    cond_dir     = os.path.join(out_root, condition)
    os.makedirs(cond_dir,    exist_ok=True)
    os.makedirs(RESULTS_V7,  exist_ok=True)

    # ── Verify schedule math ──────────────────────────────────────────────────
    sched_preview = [args.lambda_0 * (1.0 + args.epsilon) ** t
                     for t in [0, 100, 200, 300, 400, args.max_lambda_steps - 1]
                     if args.lambda_0 * (1.0 + args.epsilon) ** t <= args.max_lambda]

    # ── Startup banner ────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"NAM v7 — Sparsity warm-start path (STEP {STEP_N})")
    print(f"  [audit-fix] Issue 4: best-within-step checkpoint restored before lambda+1")
    print(f"  [audit-fix] Issue 9: warmup_epochs={args.warmup_epochs} (Setting A)")
    print(f"  [audit-fix] Issue 3: set_all_seeds() includes random.seed")
    print(f"  [audit-fix] Issue 7: per-seed scaler in seed_N/scaler.pkl")
    print(f"  [audit-fix] Issue 8: CUDA determinism flags set")
    print(f"  Condition:        {condition}")
    print(f"  concurvity_lambda={concurvity_lambda}")
    print(f"  Seeds:            {seeds}")
    print(f"  Config:           hidden={list(hidden_dims)}, dropout={dropout}, "
          f"wd={weight_decay:.0e}")
    print(f"  Schedule:         lambda_0={args.lambda_0:.3e}, "
          f"epsilon={args.epsilon}, max_steps={args.max_lambda_steps}")
    print(f"    Preview lambda at t=[0,100,200,300,400,{args.max_lambda_steps-1}]: "
          f"{[f'{v:.3e}' for v in sched_preview]}")
    print(f"  Per-step budget:  max_warm_epochs={args.max_warm_epochs}, "
          f"warm_patience={args.warm_patience}")
    print(f"  Per-seed budget:  {args.budget_hours:.1f} h")
    print(f"  Val split:        val_random_state=42 (FIXED — same as STEP 2/4)")
    print(f"  Device:           {DEVICE}")
    print(f"  Output:           {cond_dir}/")
    print(f"  READY")
    print(f"{'='*68}\n")

    # ── Save config.json at startup (before any training) ─────────────────────
    config = {
        "condition":           condition,
        "concurvity_lambda":   concurvity_lambda,
        "seeds":               seeds,
        "hidden_dims":         list(hidden_dims),
        "dropout":             dropout,
        "weight_decay":        weight_decay,
        "lambda_0":            args.lambda_0,
        "epsilon":             args.epsilon,
        "max_lambda":          args.max_lambda,
        "max_lambda_steps":    args.max_lambda_steps,
        "max_dense_epochs":    args.max_dense_epochs,
        "max_warm_epochs":     args.max_warm_epochs,
        "warm_patience":       args.warm_patience,
        "budget_hours":        args.budget_hours,
        "warmup_epochs":       args.warmup_epochs,
        "device":              str(DEVICE),
        "winner_json":         winner_json_path,
        "started_at":          datetime.now(timezone.utc).isoformat(),
        "schedule_note": (
            f"lambda_t = {args.lambda_0:.3e} * {1+args.epsilon:.4f}^t; "
            f"at t={args.max_lambda_steps-1}: approx "
            f"{args.lambda_0 * (1+args.epsilon)**(args.max_lambda_steps-1):.3e}"
        ),
    }
    config_path = os.path.join(out_root, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[startup] Config saved: {config_path}")

    # ── Load data once (shared across seeds) ──────────────────────────────────
    raw = load_raw_data(FEATURES_PATH, SPLITS_PATH)

    # ── Seed loop ─────────────────────────────────────────────────────────────
    t_total_start  = time.time()
    all_results    = []
    budget_exceeded = False

    for i_seed, seed in enumerate(seeds):
        print(f"\n{'─'*68}")
        print(f"[{condition}]  Seed {seed}  ({i_seed+1}/{len(seeds)})")
        print(f"{'─'*68}")

        t_seed = time.time()
        try:
            result = run_one_seed(
                seed=seed,
                condition=condition,
                hidden_dims=hidden_dims,
                dropout=dropout,
                weight_decay=weight_decay,
                concurvity_lambda=concurvity_lambda,
                raw=raw,
                out_dir=cond_dir,
                lambda_0=args.lambda_0,
                epsilon=args.epsilon,
                max_lambda=args.max_lambda,
                max_lambda_steps=args.max_lambda_steps,
                max_dense_epochs=args.max_dense_epochs,
                dense_patience=DENSE_PATIENCE,
                max_warm_epochs=args.max_warm_epochs,
                warm_patience=args.warm_patience,
                skip_convergence_check=args.skip_convergence_check,
                budget_hours=args.budget_hours,
            )
            all_results.append(result)

            elapsed_seed = (time.time() - t_seed) / 3600.0
            if elapsed_seed > args.budget_hours:
                print(
                    f"\n[BUDGET] Seed {seed} took {elapsed_seed:.2f} h > "
                    f"{args.budget_hours:.1f} h budget. "
                    f"Consider reducing max_warm_epochs or skipping remaining seeds."
                )
                budget_exceeded = True

        except Exception as exc:
            print(f"\n[ERROR] Seed {seed} failed: {exc}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            continue

    # ── PART 5: Condition summary ─────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"[{condition}] All seeds done. Writing condition summary...")
    summarize_condition(
        condition=condition,
        out_dir=cond_dir,
        concept_names=raw["concept_names"],
        seeds=seeds,
    )

    # ── Final run config ──────────────────────────────────────────────────────
    elapsed_total = (time.time() - t_total_start) / 60.0
    run_cfg = {
        "condition":            condition,
        "seeds_run":            [r["seed"]            for r in all_results],
        "n_steps_per_seed":     [r["n_steps_run"]     for r in all_results],
        "dense_val_balacc":     [r["dense_val_balacc"] for r in all_results],
        "elapsed_min_per_seed": [r["elapsed_min"]     for r in all_results],
        "avg_sec_per_step":     [r["avg_sec_per_step"] for r in all_results],
        "total_elapsed_min":    elapsed_total,
        "budget_exceeded":      budget_exceeded,
        "completed_at":         datetime.now(timezone.utc).isoformat(),
    }
    run_cfg_path = os.path.join(cond_dir, "run_config.json")
    with open(run_cfg_path, "w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"  run_config: {run_cfg_path}")
    print(f"  Total elapsed: {elapsed_total:.1f} min")
    print(f"{'='*68}\n")

    # ── STEP flag: only if both conditions are complete and full seed set ─────
    # (Written only when all 5 seeds ran for this condition; STEP 6 can check
    #  for the flag of BOTH conditions before proceeding.)
    if not budget_exceeded and set(seeds) == set(SEEDS):
        # Check whether the OTHER condition also has a run_config.json
        cond_other = ("sparsity_concurvity" if condition == "sparsity_only"
                      else "sparsity_only")
        other_cfg = os.path.join(out_root, cond_other, "run_config.json")
        if os.path.exists(other_cfg):
            write_step_flag(RESULTS_V7, STEP_N)
            print(f"[flag] Both conditions complete — STEP_{STEP_N}_COMPLETE.flag written.")
        else:
            print(f"[flag] Waiting for {cond_other} to complete before writing flag.")


if __name__ == "__main__":
    main()
