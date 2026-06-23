"""
Warm-start concurvity verification.

Verifies that the lambda_c = 1.0 elbow found in the cold-start concurvity sweep
also appears when the path is walked warm-start style (LassoNet/glmnet convention).

Method:
  1. Train an unpenalized dense model (concurvity_lambda=0) to convergence.
     Checkpoint is cached keyed on (seed, max_dense_epochs).
  2. Walk an increasing geometric concurvity schedule (lambda_0=0.001,
     epsilon=0.15, max_lambda=100, ~83 steps).  At each step:
       - Warm-start from previous step's weights (optimizer NOT carried over).
       - Fine-tune for up to 50 epochs with patience=10 on val_loss.
       - val_loss = CE + lambda_c * R_perp  (no sparsity term ever).
  3. Apply elbow criterion (TOLERANCE=0.013) to the warm-start path.
  4. Compare warm-start elbow to cold-start elbow at lambda_c=1.0.

Produces:
  results/concurvity_warmstart/rperp_path_seed{N}.png  — regularization-path plot
  results/concurvity_warmstart/path_seed{N}.csv        — per-lambda metrics table
  Prints a 15-row sampled table and elbow statement to stdout.

Run from project root:
    python scripts/verify_concurvity_warm_start.py
    python scripts/verify_concurvity_warm_start.py --seed 42 --out_dir results/concurvity_warmstart
    python scripts/verify_concurvity_warm_start.py --max_dense_epochs 50 --skip_convergence_check
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
from datetime import datetime, timezone

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.models.nam_multiclass import NAMMulticlass
from src.models.concurvity import multiclass_concurvity

# ── Fixed paths (identical to run_sparsity_sweep.py) ─────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
SWEEP_CSV     = "reports/nam/v6_sweep/sweep_results.csv"

N_FEATURES = 24
N_CLASSES  = 7
SELECTED_CONFIG_ID = 9

CONVERGENCE_THRESHOLD = 0.50
CONVERGENCE_EPOCH     = 30

# Elbow criterion (plot_concurvity_tradeoff.py: TOLERANCE = 0.013)
ELBOW_TOLERANCE   = 0.013
COLD_START_ELBOW  = 1.0   # reference value from cold-start sweep

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities (verbatim from run_sparsity_sweep.py to avoid importing it)
# ─────────────────────────────────────────────────────────────────────────────

def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _get_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _make_loader(dataset: TensorDataset, batch_size: int,
                 shuffle: bool, pin_memory: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin_memory, drop_last=False)


def load_ham10000(
    features_path:    str = FEATURES_PATH,
    splits_path:      str = SPLITS_PATH,
    val_random_state: int = 42,
    device: torch.device  = DEVICE,
) -> dict:
    """Load BiomedCLIP concept scores, split, standardise, and tensorise.

    Mirrors run_sparsity_sweep.py lines 133-225 exactly.
    """
    feat          = np.load(features_path, allow_pickle=True)
    scores        = feat["scores"]
    labels        = feat["labels"]
    lesion_ids    = feat["lesion_ids"]
    concept_names = feat["concept_ids"].tolist()
    assert scores.shape == (10015, N_FEATURES), f"Unexpected shape: {scores.shape}"
    assert len(concept_names) == N_FEATURES

    split     = np.load(splits_path)
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

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20,
                            random_state=val_random_state)
    train_rel, val_rel = next(
        gss.split(X_all_train, y_all_train, groups=lesion_ids_train)
    )
    assert len(
        set(lesion_ids_train[train_rel]) & set(lesion_ids_train[val_rel])
    ) == 0, "Lesion leakage between train and val"

    X_train_raw = X_all_train[train_rel]
    y_train_enc = y_all_train_enc[train_rel]
    y_train_str = y_all_train[train_rel]
    X_val_raw   = X_all_train[val_rel]
    y_val_enc   = y_all_train_enc[val_rel]

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val_sc    = scaler.transform(X_val_raw).astype(np.float32)
    X_test_sc   = scaler.transform(X_test).astype(np.float32)

    weights       = compute_class_weight(
        "balanced", classes=np.array(class_names), y=y_train_str
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(
        torch.tensor(X_train_sc,  dtype=torch.float32),
        torch.tensor(y_train_enc, dtype=torch.long),
    )

    return {
        "X_train_t":     torch.tensor(X_train_sc, dtype=torch.float32, device=device),
        "y_train_t":     torch.tensor(y_train_enc, dtype=torch.long,   device=device),
        "X_val_t":       torch.tensor(X_val_sc,   dtype=torch.float32, device=device),
        "y_val_t":       torch.tensor(y_val_enc,  dtype=torch.long,    device=device),
        "X_test_t":      torch.tensor(X_test_sc,  dtype=torch.float32, device=device),
        "y_val_enc":     y_val_enc,
        "y_test_enc":    y_test_enc,
        "concept_names": concept_names,
        "class_names":   class_names,
        "weight_tensor": weight_tensor,
        "scaler":        scaler,
        "train_dataset": train_dataset,
    }


def load_sweep_hyperparams(
    sweep_csv: str = SWEEP_CSV,
    config_id: int = SELECTED_CONFIG_ID,
) -> tuple[tuple, float, float]:
    """Read hidden_dims, dropout, weight_decay for config_id from sweep CSV."""
    if not os.path.exists(sweep_csv):
        raise FileNotFoundError(
            f"Sweep results not found at {sweep_csv}. Run sweep_nam_v6.py first."
        )
    df  = pd.read_csv(sweep_csv)
    row = df[df["config_id"] == config_id]
    if row.empty:
        raise ValueError(
            f"Config {config_id} not found in {sweep_csv}. "
            f"Available IDs: {sorted(df['config_id'].tolist())}"
        )
    sel          = row.iloc[0]
    hidden_dims  = tuple(ast.literal_eval(sel["hidden"]))
    dropout      = float(sel["dropout"])
    weight_decay = float(sel["weight_decay"])
    return hidden_dims, dropout, weight_decay


# ─────────────────────────────────────────────────────────────────────────────
# Validation evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_val(
    model:             NAMMulticlass,
    X_val_t:           torch.Tensor,
    y_val_t:           torch.Tensor,
    y_val_enc:         np.ndarray,
    criterion:         nn.Module,
    concurvity_lambda: float,
) -> tuple[float, float, float, float]:
    """Return (val_balacc, val_auc, val_loss, r_perp_val).

    val_loss = CE + concurvity_lambda * R_perp.
    r_perp_val is always returned as a diagnostic regardless of concurvity_lambda.
    """
    model.eval()
    logits, shape_outs = model(X_val_t, return_shape_outputs=True)
    ce_loss   = criterion(logits, y_val_t).item()
    r_perp    = multiclass_concurvity(shape_outs).item()
    val_loss  = ce_loss + concurvity_lambda * r_perp

    preds  = logits.argmax(dim=1).cpu().numpy()
    proba  = torch.softmax(logits, dim=1).cpu().numpy()
    balacc = balanced_accuracy_score(y_val_enc, preds)
    auc    = roc_auc_score(
        y_val_enc, proba,
        multi_class="ovr", average="weighted",
        labels=list(range(N_CLASSES)),
    )
    return balacc, auc, val_loss, r_perp


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_concurvity_warm_start(
    data:   dict,
    hidden_dims:  tuple,
    dropout:      float,
    weight_decay: float,
    seed:         int   = 42,
    lr:           float = 1e-3,
    batch_size:   int   = 256,
    # Dense phase
    max_dense_epochs:       int   = 100,
    dense_patience:         int   = 15,
    sched_patience:         int   = 5,
    sched_factor:           float = 0.5,
    skip_convergence_check: bool  = False,
    # Concurvity path
    lambda_0:              float = 1e-3,
    epsilon:               float = 0.15,
    max_lambda:            float = 100.0,
    max_lambda_steps:      int   = 300,
    max_warm_start_epochs: int   = 50,
    warm_start_patience:   int   = 10,
    warm_start_min_delta:  float = 1e-4,
    # I/O
    out_dir: str          = "results/concurvity_warmstart",
    device:  torch.device = DEVICE,
) -> dict:
    """Dense-to-regularized warm-started concurvity path.

    Phase 1: train unpenalized (concurvity_lambda=0) dense model to convergence.
    Checkpoint cached as dense_seed{N}_ep{E}.pt to avoid re-training on re-runs.

    Phase 2: walk an increasing geometric lambda_c schedule. At each step:
      - Warm-start model weights from previous step (optimizer NOT carried over).
      - Fresh Adam + ReduceLROnPlateau(mode='min') on val_loss.
      - val_loss = CE + lambda_c * R_perp  (sparsity term is never included).
      - Fine-tune up to max_warm_start_epochs epochs, early-stop on val_loss.

    Returns dict with keys:
      baseline_val_balacc, baseline_val_auc, baseline_r_perp
      rows: list[dict] — one per lambda step (lambda_c, val_balacc, val_auc,
                          val_loss, r_perp_val, actual_epochs)
      warm_start_elbow: float | None
    """
    _set_seeds(seed)
    os.makedirs(out_dir, exist_ok=True)

    X_val_t       = data["X_val_t"]
    y_val_t       = data["y_val_t"]
    y_val_enc     = data["y_val_enc"]
    weight_tensor = data["weight_tensor"]
    train_dataset = data["train_dataset"]

    pin_memory = (device.type == "cuda")
    criterion  = nn.CrossEntropyLoss(weight=weight_tensor)

    model = NAMMulticlass(
        n_features=N_FEATURES,
        num_classes=N_CLASSES,
        hidden_dims=hidden_dims,
        dropout=dropout,
        concept_names=data["concept_names"],
    ).to(device)

    def _eval(lam: float) -> tuple[float, float, float, float]:
        return _evaluate_val(model, X_val_t, y_val_t, y_val_enc, criterion, lam)

    # ── Phase 1: dense model (concurvity_lambda = 0) ──────────────────────────
    ckpt_stem = f"dense_seed{seed}_ep{max_dense_epochs}"
    ckpt_pt   = os.path.join(out_dir, f"{ckpt_stem}.pt")
    ckpt_json = os.path.join(out_dir, f"{ckpt_stem}.json")

    if os.path.exists(ckpt_pt) and os.path.exists(ckpt_json):
        model.load_state_dict(
            torch.load(ckpt_pt, map_location=device, weights_only=True)
        )
        meta                = json.load(open(ckpt_json))
        baseline_balacc     = meta["baseline_val_balacc"]
        baseline_auc        = meta["baseline_val_auc"]
        baseline_r_perp     = meta["baseline_r_perp"]
        print(f"[dense] Loaded cached checkpoint: {ckpt_pt}")
        print(f"        val_balacc={baseline_balacc:.4f}  "
              f"val_auc={baseline_auc:.4f}  R_perp={baseline_r_perp:.4f}")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=sched_factor,
            patience=sched_patience, min_lr=1e-6,
        )
        best_val_balacc   = -1.0
        patience_ctr      = 0
        reached_threshold = False

        print(f"[dense] Training dense model (seed={seed}, concurvity_lambda=0) ...")
        for epoch in range(max_dense_epochs):
            model.train()
            for X_b, y_b in _make_loader(train_dataset, batch_size, True, pin_memory):
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()

            balacc, _, _, _ = _eval(0.0)
            scheduler.step(balacc)

            if epoch < CONVERGENCE_EPOCH and balacc >= CONVERGENCE_THRESHOLD:
                reached_threshold = True

            if balacc > best_val_balacc + 1e-4:
                best_val_balacc = balacc
                patience_ctr    = 0
                torch.save(model.state_dict(), ckpt_pt)
            else:
                patience_ctr += 1
                if patience_ctr >= dense_patience:
                    print(f"[dense] Early stop at epoch {epoch + 1}  "
                          f"best val_balacc={best_val_balacc:.4f}")
                    break

        if not skip_convergence_check and not reached_threshold:
            raise RuntimeError(
                f"Dense model did not reach val_balacc >= {CONVERGENCE_THRESHOLD} "
                f"within first {CONVERGENCE_EPOCH} epochs. "
                "Pass --skip_convergence_check to bypass."
            )

        model.load_state_dict(
            torch.load(ckpt_pt, map_location=device, weights_only=True)
        )
        baseline_balacc, baseline_auc, _, baseline_r_perp = _eval(0.0)

        json.dump(
            {
                "baseline_val_balacc": baseline_balacc,
                "baseline_val_auc":    baseline_auc,
                "baseline_r_perp":     baseline_r_perp,
                "hidden_dims":         list(hidden_dims),
                "dropout":             dropout,
                "weight_decay":        weight_decay,
                "lr":                  lr,
                "batch_size":          batch_size,
                "max_dense_epochs":    max_dense_epochs,
                "dense_patience":      dense_patience,
                "seed":                seed,
                "git_sha":             _get_git_sha(),
                "timestamp":           datetime.now(timezone.utc).isoformat(),
            },
            open(ckpt_json, "w"),
            indent=2,
        )
        print(f"[dense] Done.  val_balacc={baseline_balacc:.4f}  "
              f"val_auc={baseline_auc:.4f}  R_perp={baseline_r_perp:.4f}  "
              f"ckpt={ckpt_pt}")

    prev_state_dict = model.state_dict()

    # ── Phase 2: concurvity lambda schedule ───────────────────────────────────
    lambda_schedule: list[float] = []
    t = 0
    while len(lambda_schedule) < max_lambda_steps:
        lam = lambda_0 * (1.0 + epsilon) ** t
        if lam > max_lambda:
            break
        lambda_schedule.append(lam)
        t += 1
    print(f"[path]  Concurvity schedule: {len(lambda_schedule)} steps  "
          f"{lambda_schedule[0]:.3e} → {lambda_schedule[-1]:.3e}")

    rows: list[dict] = []

    for step_idx, lambda_c in enumerate(lambda_schedule):

        # (a) Warm-start from previous step
        model.load_state_dict(prev_state_dict)

        # (b) Fresh optimizer + scheduler (mode="min" on val_loss)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sched_factor,
            patience=sched_patience, min_lr=1e-6,
        )

        # (c) Fine-tuning with patience on val_loss
        best_step_val_loss = float("inf")
        no_improve_ctr     = 0
        actual_epochs      = 0
        last_balacc = last_auc = last_val_loss = last_r_perp = float("nan")

        for epoch in range(max_warm_start_epochs):
            model.train()
            for X_b, y_b in _make_loader(train_dataset, batch_size, True, pin_memory):
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits, shape_outs = model(X_b, return_shape_outputs=True)
                loss = (criterion(logits, y_b)
                        + lambda_c * multiclass_concurvity(shape_outs))
                loss.backward()
                optimizer.step()
                # No proximal step — no sparsity in this sweep.

            actual_epochs += 1
            last_balacc, last_auc, last_val_loss, last_r_perp = _eval(lambda_c)
            scheduler.step(last_val_loss)

            if last_val_loss < best_step_val_loss - warm_start_min_delta:
                best_step_val_loss = last_val_loss
                no_improve_ctr     = 0
            else:
                no_improve_ctr += 1
                if no_improve_ctr >= warm_start_patience:
                    break

        rows.append({
            "lambda_c":      lambda_c,
            "val_balacc":    last_balacc,
            "val_auc":       last_auc,
            "val_loss":      last_val_loss,
            "r_perp_val":    last_r_perp,
            "actual_epochs": actual_epochs,
        })

        prev_state_dict = model.state_dict()

        if (step_idx + 1) % 10 == 0 or step_idx == 0:
            print(f"  step {step_idx+1:3d}/{len(lambda_schedule)}  "
                  f"lambda_c={lambda_c:.3e}  "
                  f"val_balacc={last_balacc:.4f}  "
                  f"R_perp={last_r_perp:.4f}  "
                  f"epochs={actual_epochs}")

    # ── Elbow criterion ────────────────────────────────────────────────────────
    threshold     = baseline_balacc - ELBOW_TOLERANCE
    warm_start_elbow: float | None = None
    for row in rows:
        if row["val_balacc"] >= threshold:
            warm_start_elbow = row["lambda_c"]
    # warm_start_elbow is the last lambda satisfying the criterion

    return {
        "baseline_val_balacc": baseline_balacc,
        "baseline_val_auc":    baseline_auc,
        "baseline_r_perp":     baseline_r_perp,
        "rows":                rows,
        "warm_start_elbow":    warm_start_elbow,
        "lambda_schedule":     lambda_schedule,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_concurvity_path(results: dict, seed: int, out_dir: str) -> None:
    """R_perp vs lambda_c with val_balacc on twin axis; elbow marked."""
    rows      = results["rows"]
    elbow     = results["warm_start_elbow"]
    baseline  = results["baseline_val_balacc"]
    threshold = baseline - ELBOW_TOLERANCE

    lambdas   = [r["lambda_c"]   for r in rows]
    r_perps   = [r["r_perp_val"] for r in rows]
    balaccs   = [r["val_balacc"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.semilogx(lambdas, r_perps, color="steelblue", linewidth=1.8,
                 label="R_perp (val)")
    ax1.set_xlabel("Concurvity lambda")
    ax1.set_ylabel("R_perp (val)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.semilogx(lambdas, balaccs, color="darkorange", linewidth=1.8,
                 linestyle="--", label="val balanced acc")
    ax2.axhline(baseline,  color="gray",      linestyle=":",  linewidth=1.2,
                label=f"baseline = {baseline:.4f}")
    ax2.axhline(threshold, color="gray",      linestyle="-.", linewidth=1.0,
                label=f"threshold = {threshold:.4f}")
    ax2.set_ylabel("Val balanced accuracy", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    if elbow is not None:
        ax1.axvline(elbow, color="red", linestyle="--", linewidth=1.5,
                    label=f"warm-start elbow = {elbow:.3e}")
    ax1.axvline(COLD_START_ELBOW, color="purple", linestyle=":",
                linewidth=1.5, label=f"cold-start elbow = {COLD_START_ELBOW:.1f}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=8, framealpha=0.8)

    elbow_str = f"{elbow:.3e}" if elbow is not None else "None"
    ax1.set_title(f"Warm-start concurvity path — seed {seed}\n"
                  f"TOLERANCE={ELBOW_TOLERANCE}  "
                  f"warm-start elbow={elbow_str}  "
                  f"cold-start elbow={COLD_START_ELBOW}")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"rperp_path_seed{seed}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table / summary reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_sampled_table(results: dict, seed: int, n_rows: int = 15) -> None:
    """Print n_rows evenly sampled lambda steps."""
    rows    = results["rows"]
    elbow   = results["warm_start_elbow"]
    n_total = len(rows)

    if n_total <= n_rows:
        indices = list(range(n_total))
    else:
        indices = [round(i * (n_total - 1) / (n_rows - 1)) for i in range(n_rows)]

    header = (f"{'step':>5}  {'lambda_c':>10}  {'val_balacc':>10}  "
              f"{'val_auc':>8}  {'R_perp_val':>10}  {'epochs':>6}")
    sep    = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print(f"Warm-start concurvity path (seed={seed})  —  {n_rows} sampled rows")
    print(f"  Baseline val_balacc : {results['baseline_val_balacc']:.4f}")
    print(f"  Threshold (-{ELBOW_TOLERANCE}): "
          f"{results['baseline_val_balacc'] - ELBOW_TOLERANCE:.4f}")
    print(f"  Warm-start elbow    : {elbow:.3e}" if elbow else
          "  Warm-start elbow    : None (all below threshold)")
    print(f"  Cold-start elbow    : {COLD_START_ELBOW:.1f}")
    print(sep)
    print(header)
    print(sep)

    for i in indices:
        r = rows[i]
        marker = " <-- elbow" if elbow is not None and abs(r["lambda_c"] - elbow) < 1e-12 else ""
        print(f"{i+1:5d}  {r['lambda_c']:10.3e}  {r['val_balacc']:10.4f}  "
              f"{r['val_auc']:8.4f}  {r['r_perp_val']:10.4f}  "
              f"{r['actual_epochs']:6d}{marker}")
    print(sep)


def print_elbow_verdict(results: dict) -> None:
    """Print the elbow comparison statement."""
    elbow    = results["warm_start_elbow"]
    baseline = results["baseline_val_balacc"]
    print("\n" + "=" * 65)
    print("STEP 1 VERDICT — warm-start concurvity elbow")
    print("=" * 65)
    print(f"  Baseline val_balacc (lambda_c=0) : {baseline:.4f}")
    print(f"  Tolerance                        : {ELBOW_TOLERANCE}")
    print(f"  Threshold                        : {baseline - ELBOW_TOLERANCE:.4f}")
    if elbow is None:
        print("  Warm-start elbow : None")
        print("  Cold-start elbow : 1.0")
        print("\n  MISMATCH: warm-start path drops below threshold immediately.")
        print("  Check dense baseline quality before proceeding.")
    else:
        import math
        log_ratio = abs(math.log10(elbow) - math.log10(COLD_START_ELBOW))
        print(f"  Warm-start elbow : {elbow:.3e}")
        print(f"  Cold-start elbow : {COLD_START_ELBOW:.1f}")
        print(f"  |log10 ratio|    : {log_ratio:.2f}  (threshold for STOP: >0.5)")
        if log_ratio > 0.5:
            print("\n  STOP: elbows differ by more than 0.5 orders of magnitude.")
            print("  Investigate before proceeding to production sweep.")
        else:
            print("\n  PASS: elbows agree within 0.5 orders of magnitude.")
            print("  Proceed to Steps 2-3 (production 5-seed warm-start sparsity sweep).")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warm-start concurvity verification for NAM v6."
    )
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--out_dir",        type=str,   default="results/concurvity_warmstart")
    parser.add_argument("--lambda_0",       type=float, default=1e-3,
                        help="Starting concurvity lambda. Default 1e-3.")
    parser.add_argument("--epsilon",        type=float, default=0.15,
                        help="Geometric step size. Default 0.15 (~83 steps to lambda=100).")
    parser.add_argument("--max_lambda",     type=float, default=100.0,
                        help="Largest concurvity lambda. Default 100.")
    parser.add_argument("--max_dense_epochs", type=int, default=100,
                        help="Dense phase max epochs. Default 100.")
    parser.add_argument("--max_warm_start_epochs", type=int, default=50,
                        help="Per-lambda fine-tuning max epochs. Default 50.")
    parser.add_argument("--warm_start_patience", type=int, default=10,
                        help="Per-lambda patience (val_loss). Default 10.")
    parser.add_argument("--skip_convergence_check", action="store_true", default=False,
                        help="Skip dense convergence guard (for quick tests).")
    parser.add_argument("--device",         type=str,   default=None,
                        help="Torch device string. Default: auto-detect.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE
    if device.type == "cpu":
        print("WARNING: CUDA not available — training on CPU.")

    print("=" * 65)
    print("NAM v6 — warm-start concurvity verification (Step 1)")
    print(f"  seed              : {args.seed}")
    print(f"  lambda range      : [{args.lambda_0:.0e}, {args.max_lambda:.0f}]  "
          f"epsilon={args.epsilon}")
    print(f"  dense max_epochs  : {args.max_dense_epochs}")
    print(f"  fine-tune epochs  : {args.max_warm_start_epochs}")
    print(f"  fine-tune patience: {args.warm_start_patience}")
    print(f"  out_dir           : {args.out_dir}")
    print("=" * 65)

    print("\nLoading data...")
    data = load_ham10000(device=device)
    print(f"  train: {len(data['train_dataset'])}  "
          f"val: {len(data['y_val_enc'])}  "
          f"test: {len(data['y_test_enc'])}")

    print("\nLoading config 9 hyperparameters...")
    hidden_dims, dropout, weight_decay = load_sweep_hyperparams()
    print(f"  hidden_dims={list(hidden_dims)}  dropout={dropout}  "
          f"weight_decay={weight_decay:.0e}")

    results = run_concurvity_warm_start(
        data=data,
        hidden_dims=hidden_dims,
        dropout=dropout,
        weight_decay=weight_decay,
        seed=args.seed,
        max_dense_epochs=args.max_dense_epochs,
        skip_convergence_check=args.skip_convergence_check,
        lambda_0=args.lambda_0,
        epsilon=args.epsilon,
        max_lambda=args.max_lambda,
        max_warm_start_epochs=args.max_warm_start_epochs,
        warm_start_patience=args.warm_start_patience,
        out_dir=args.out_dir,
        device=device,
    )

    # Write CSV
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"path_seed{args.seed}.csv")
    pd.DataFrame(results["rows"]).to_csv(csv_path, index=False)
    print(f"[csv]  Saved: {csv_path}")

    plot_concurvity_path(results, args.seed, args.out_dir)
    print_sampled_table(results, args.seed, n_rows=15)
    print_elbow_verdict(results)


if __name__ == "__main__":
    main()
