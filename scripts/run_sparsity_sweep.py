"""
Sparsity lambda sweep for NAM v6 — cold-start and warm-start modes.

Cold-start (--sweep_mode=cold_start):
  Subprocess launcher: calls train_nam_v6_final.py once per sparsity lambda.
  Mirrors run_concurvity_sweep.py; does NOT modify train_nam_v6_final.py.
  Results go to results/cold_start_sparsity/lambda_{lam}/.

Warm-start (--sweep_mode=warm_start):
  In-process dense-to-sparse path following LassoNet / glmnet convention
  (Lemhadri et al. 2021; Friedman et al. 2010).
  Trains one unpenalized model to convergence, then fine-tunes along an
  increasing lambda schedule, carrying weights (NOT optimizer state) from
  step to step.
  Results go to results/warm_start/ (configurable via --out_dir).

Key design invariants (do not violate):
  - apply_proximal_step formula is unchanged from src/models/sparsity.py.
  - Proximal step fires after optimizer.step() in every minibatch, using the
    CURRENT learning rate (optimizer.param_groups[0]["lr"]), not the initial lr.
  - Val loss for patience / LR scheduler = CE + concurvity_lambda * R_perp.
    The sparsity penalty is NEVER included in val_loss.
  - Adam and ReduceLROnPlateau are re-initialized at every new lambda step.
    Model weights are carried over (warm start); optimizer state is not.
  - concurvity_lambda is fixed for the entire path (not swept jointly).

Usage (from project root):
    python scripts/run_sparsity_sweep.py --sweep_mode=warm_start --seed=42
    python scripts/run_sparsity_sweep.py --sweep_mode=cold_start
    python scripts/run_sparsity_sweep.py --sweep_mode=cold_start \\
        --sparsity_lambdas "0.01,0.1,1,10"
"""

from __future__ import annotations

import argparse
import ast
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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive; safe for headless / cluster runs
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
from src.models.sparsity import apply_proximal_step, feature_group_norms

# ── Fixed paths ────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
SWEEP_CSV     = "reports/nam/v6_sweep/sweep_results.csv"
TRAIN_SCRIPT  = "scripts/train_nam_v6_final.py"

N_FEATURES = 24
N_CLASSES  = 7

# Convergence guard — same thresholds as train_nam_v6_final.py.
CONVERGENCE_THRESHOLD = 0.50
CONVERGENCE_EPOCH     = 30

# Default cold-start lambda grid: log-spaced with finer resolution around
# the phase transition region (lambda = 1–100) observed in initial sweeps.
COLD_START_DEFAULT_LAMBDAS = [
    1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 3, 10, 30, 100, 300, 1e3,
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
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


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# Mirrors train_nam_v6_final.py lines 188–257.  train_nam_v6_final.py is NOT
# imported here (it executes at module level and cannot be safely imported).
# ─────────────────────────────────────────────────────────────────────────────

def load_ham10000(
    features_path:    str = FEATURES_PATH,
    splits_path:      str = SPLITS_PATH,
    val_random_state: int = 42,
    device: torch.device  = DEVICE,
) -> dict:
    """Load BiomedCLIP concept scores, split, standardise, and tensorise.

    Returns a dict:
      X_train_t, y_train_t  — training tensors on `device`
      X_val_t,   y_val_t    — validation tensors on `device`
      X_test_t              — test tensor on `device`
      y_val_enc, y_test_enc — integer-encoded labels (numpy int64)
      concept_names         — list[str], len=24
      class_names           — list[str], len=7, sorted
      weight_tensor         — balanced class weights on `device`
      scaler                — fitted StandardScaler (for downstream use)
      train_dataset         — TensorDataset on CPU (for DataLoader + pin_memory)

    The val split uses val_random_state=42 to match the existing trainer exactly.
    """
    feat          = np.load(features_path, allow_pickle=True)
    scores        = feat["scores"]
    labels        = feat["labels"]
    lesion_ids    = feat["lesion_ids"]
    concept_names = feat["concept_ids"].tolist()
    assert scores.shape == (10015, N_FEATURES), f"Unexpected feature shape: {scores.shape}"
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

    # Training DataLoader uses CPU tensors so pin_memory works correctly.
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
    config_id: int = 9,
) -> tuple[tuple, float, float]:
    """Read hidden_dims, dropout, weight_decay for config_id from sweep CSV."""
    if not os.path.exists(sweep_csv):
        raise FileNotFoundError(
            f"Sweep results not found at {sweep_csv}. Run sweep_nam_v6.py first."
        )
    df  = pd.read_csv(sweep_csv)
    row = df[df["config_id"] == config_id]
    if row.empty:
        raise ValueError(f"Config {config_id} not found in {sweep_csv}. "
                         f"Available IDs: {sorted(df['config_id'].tolist())}")
    sel          = row.iloc[0]
    hidden_dims  = tuple(ast.literal_eval(sel["hidden"]))
    dropout      = float(sel["dropout"])
    weight_decay = float(sel["weight_decay"])
    return hidden_dims, dropout, weight_decay


# ─────────────────────────────────────────────────────────────────────────────
# Validation evaluation (single definition used throughout this file)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_val(
    model:             NAMMulticlass,
    X_val_t:           torch.Tensor,
    y_val_t:           torch.Tensor,
    y_val_enc:         np.ndarray,
    criterion:         nn.Module,
    concurvity_lambda: float,
) -> tuple[float, float, float]:
    """Return (val_balacc, val_auc, val_loss).

    val_loss = CE + concurvity_lambda * R_perp.  The sparsity penalty is
    NEVER included — it is an optimization-side term, not a model-quality metric.

    val_balacc uses argmax predictions, matching balanced_accuracy_score usage
    in train_nam_v6_final.py.

    val_auc uses roc_auc_score(OvR, weighted average) on integer-encoded labels,
    consistent with how test_auc is computed in evaluate_on_test().
    """
    model.eval()
    logits, shape_outs = model(X_val_t, return_shape_outputs=True)
    ce_loss  = criterion(logits, y_val_t).item()
    r_perp   = multiclass_concurvity(shape_outs).item()
    val_loss = ce_loss + concurvity_lambda * r_perp

    preds   = logits.argmax(dim=1).cpu().numpy()
    proba   = torch.softmax(logits, dim=1).cpu().numpy()
    balacc  = balanced_accuracy_score(y_val_enc, preds)
    auc     = roc_auc_score(
        y_val_enc, proba,
        multi_class="ovr", average="weighted",
        labels=list(range(N_CLASSES)),
    )
    return balacc, auc, val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Cold-start sweep (subprocess launcher, mirrors run_concurvity_sweep.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_cold_start_sweep(args: argparse.Namespace) -> None:
    """Subprocess launcher: calls train_nam_v6_final.py once per sparsity lambda.

    train_nam_v6_final.py is called unchanged.  This function is a thin
    coordinator and contains no training logic of its own.
    """
    if args.sparsity_lambdas:
        lambdas = [
            float(x)
            for x in args.sparsity_lambdas.replace(",", " ").split()
            if x.strip()
        ]
    else:
        lambdas = COLD_START_DEFAULT_LAMBDAS

    sweep_root = pathlib.Path(args.cold_out_dir)
    done_file  = "metrics_summary.txt"

    env    = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    python = sys.executable

    print("=" * 65)
    print("Sparsity lambda sweep — cold-start mode")
    print(f"  trainer           : {TRAIN_SCRIPT}")
    print(f"  seed              : {args.seed}")
    print(f"  concurvity_lambda : {args.concurvity_lambda}")
    print(f"  lambdas ({len(lambdas):2d})      : {lambdas}")
    print(f"  output            : {sweep_root}/")
    print("=" * 65)
    print()

    n_run = n_skip = 0
    for lam in lambdas:
        out_dir   = sweep_root / f"lambda_{lam}"
        done_path = out_dir / done_file
        log_path  = out_dir / "run.log"

        if done_path.exists():
            print(f"[skip ] lambda={lam} — {done_file} already present in {out_dir}")
            n_skip += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[start] lambda={lam} → {out_dir}")
        print(f"        log: {log_path}")

        cmd = [
            python, TRAIN_SCRIPT,
            "--sparsity_lambda",   str(lam),
            "--concurvity_lambda", str(args.concurvity_lambda),
            "--out_dir",           str(out_dir),
            "--seed",              str(args.seed),
        ]
        if args.max_epochs is not None:
            cmd += ["--max_epochs", str(args.max_epochs)]
        if args.skip_convergence_check:
            cmd += ["--skip_convergence_check"]

        with log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_fh.write(line)
            proc.wait()

        if proc.returncode != 0:
            print(f"[ERROR] lambda={lam} — exit code {proc.returncode}. See {log_path}")
            sys.exit(proc.returncode)

        if not done_path.exists():
            print(f"[ERROR] lambda={lam} — {done_file} missing after run. See {log_path}")
            sys.exit(1)

        print(f"[done ] lambda={lam}")
        print()
        n_run += 1

    print("=" * 65)
    print(f"Cold-start sweep complete.  "
          f"Ran: {n_run}  Skipped: {n_skip}  Total: {len(lambdas)}")
    print(f"Results in: {sweep_root}/")


# ─────────────────────────────────────────────────────────────────────────────
# Warm-start sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_warm_started_sweep(
    data:                  dict,
    hidden_dims:           tuple,
    dropout:               float,
    weight_decay:          float,
    seed:                  int   = 42,
    # Dense-phase training (defaults match train_nam_v6_final.py)
    lr:                    float = 1e-3,
    batch_size:            int   = 256,
    max_dense_epochs:      int   = 100,
    dense_patience:        int   = 15,
    sched_patience:        int   = 5,
    sched_factor:          float = 0.5,
    skip_convergence_check: bool = False,
    # Warm-start path
    lambda_0:              float = 1e-3,
    epsilon:               float = 0.02,
    max_lambda:            float = 1e3,
    max_lambda_steps:      int   = 300,
    max_warm_start_epochs: int   = 50,
    warm_start_patience:   int   = 10,
    warm_start_min_delta:  float = 1e-4,
    # Fixed regularization (not swept)
    concurvity_lambda:     float = 1.0,
    # I/O
    out_dir:  str          = "results/warm_start",
    device:   torch.device = DEVICE,
) -> dict:
    """Dense-to-sparse warm-started Group LASSO regularization path.

    Phase 1: train unpenalized NAM (lambda=0) to convergence.  Checkpoint is
    cached and reused on re-runs with the same (seed, concurvity_lambda,
    max_dense_epochs).

    Phase 2: for each lambda_t in an increasing geometric schedule, load the
    previous step's weights, re-initialize optimizer+scheduler, and fine-tune
    for up to max_warm_start_epochs epochs with early stopping on val_loss.

    Returns a dict; see bottom of function for full key list.

    Output files written under out_dir/:
      dense_seed{seed}_conc{conc}_ep{max_dense_epochs}.pt    — dense checkpoint
      dense_seed{seed}_conc{conc}_ep{max_dense_epochs}.json  — dense config+metrics
      path_seed{seed}.csv           — per-lambda metrics table (# seed=N header)
      path_seed{seed}_elimination.txt — elimination order
      checkpoints/seed{seed}_lambda{lam:.6e}.pt   — state_dict per lambda step
      checkpoints/seed{seed}_lambda{lam:.6e}.json — metrics per lambda step
    """
    # ── 0. Setup ───────────────────────────────────────────────────────────────
    _set_seeds(seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    X_val_t       = data["X_val_t"]
    y_val_t       = data["y_val_t"]
    y_val_enc     = data["y_val_enc"]
    concept_names = data["concept_names"]
    weight_tensor = data["weight_tensor"]
    train_dataset = data["train_dataset"]

    pin_memory = (device.type == "cuda")
    criterion  = nn.CrossEntropyLoss(weight=weight_tensor)

    # Single model instance.  Weights are overwritten by load_state_dict() at
    # each lambda step; the object itself is reused to avoid repeated allocation.
    model = NAMMulticlass(
        n_features=N_FEATURES,
        num_classes=N_CLASSES,
        hidden_dims=hidden_dims,
        dropout=dropout,
        concept_names=concept_names,
    ).to(device)

    def _eval() -> tuple[float, float, float]:
        return _evaluate_val(
            model, X_val_t, y_val_t, y_val_enc,
            criterion, concurvity_lambda,
        )

    # ── 1. Dense model (sparsity_lambda = 0) ──────────────────────────────────
    # Checkpoint key encodes (seed, concurvity_lambda, max_dense_epochs) so that
    # changes to any of these three trigger a re-train rather than a cache hit.
    conc_tag  = str(concurvity_lambda).replace(".", "p")
    ckpt_stem = f"dense_seed{seed}_conc{conc_tag}_ep{max_dense_epochs}"
    ckpt_pt   = os.path.join(out_dir, f"{ckpt_stem}.pt")
    ckpt_json = os.path.join(out_dir, f"{ckpt_stem}.json")

    if os.path.exists(ckpt_pt) and os.path.exists(ckpt_json):
        model.load_state_dict(
            torch.load(ckpt_pt, map_location=device, weights_only=True)
        )
        meta             = json.load(open(ckpt_json))
        dense_val_balacc = meta["dense_val_balacc"]
        dense_val_auc    = meta["dense_val_auc"]
        print(f"[dense] Loaded cached checkpoint: {ckpt_pt}")
        print(f"        val_balacc={dense_val_balacc:.4f}  val_auc={dense_val_auc:.4f}")
    else:
        # Train dense model.  Uses mode="max" on val_balacc, matching the
        # existing trainer's early-stopping criterion.
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

        print(f"[dense] Training dense model (seed={seed}, "
              f"concurvity_lambda={concurvity_lambda}) ...")
        for epoch in range(max_dense_epochs):
            model.train()
            for X_b, y_b in _make_loader(train_dataset, batch_size, True, pin_memory):
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                if concurvity_lambda > 0:
                    logits, shape_outs = model(X_b, return_shape_outputs=True)
                    loss = (criterion(logits, y_b)
                            + concurvity_lambda * multiclass_concurvity(shape_outs))
                else:
                    loss = criterion(model(X_b), y_b)
                # No proximal step and no sparsity penalty for dense phase.
                loss.backward()
                optimizer.step()

            val_balacc, _, _ = _eval()
            scheduler.step(val_balacc)

            if epoch < CONVERGENCE_EPOCH and val_balacc >= CONVERGENCE_THRESHOLD:
                reached_threshold = True

            if val_balacc > best_val_balacc + 1e-4:
                best_val_balacc = val_balacc
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
                f"within the first {CONVERGENCE_EPOCH} epochs. "
                "Pass --skip_convergence_check to bypass."
            )

        model.load_state_dict(
            torch.load(ckpt_pt, map_location=device, weights_only=True)
        )
        dense_val_balacc, dense_val_auc, _ = _eval()

        json.dump(
            {
                "dense_val_balacc":  dense_val_balacc,
                "dense_val_auc":     dense_val_auc,
                "hidden_dims":       list(hidden_dims),
                "dropout":           dropout,
                "weight_decay":      weight_decay,
                "lr":                lr,
                "batch_size":        batch_size,
                "max_dense_epochs":  max_dense_epochs,
                "dense_patience":    dense_patience,
                "sched_patience":    sched_patience,
                "sched_factor":      sched_factor,
                "concurvity_lambda": concurvity_lambda,
                "seed":              seed,
                "git_sha":           _get_git_sha(),
                "timestamp":         datetime.now(timezone.utc).isoformat(),
            },
            open(ckpt_json, "w"),
            indent=2,
        )
        print(f"[dense] Done.  val_balacc={dense_val_balacc:.4f}  "
              f"val_auc={dense_val_auc:.4f}  ckpt={ckpt_pt}")

    # All dense params are written above.  Carry the best-val state into the path.
    prev_state_dict = model.state_dict()          # .state_dict() returns a copy
    prev_norms      = feature_group_norms(model)  # all >> 1e-8 at lambda=0

    # ── 2. Lambda schedule ─────────────────────────────────────────────────────
    # lambda_t = lambda_0 * (1 + epsilon)^t  for t = 0, 1, 2, ...
    # Capped by both max_lambda and max_lambda_steps.
    lambda_schedule: list[float] = []
    t = 0
    while len(lambda_schedule) < max_lambda_steps:
        lam = lambda_0 * (1.0 + epsilon) ** t
        if lam > max_lambda:
            break
        lambda_schedule.append(lam)
        t += 1
    print(f"[path]  Lambda schedule: {len(lambda_schedule)} steps  "
          f"{lambda_schedule[0]:.3e} → {lambda_schedule[-1]:.3e}")

    # ── 3. Accumulators ────────────────────────────────────────────────────────
    rows:        list[dict]  = []   # one dict per completed lambda step
    state_dicts: list[dict]  = []   # in-memory state_dicts (~232 KB each)
    elim_order:  list[tuple] = []   # [(concept_name, lambda_t), ...] in order
    selected_lams: dict      = {}   # concept_name → lambda_t of first elimination

    # ── 4. Lambda path loop ───────────────────────────────────────────────────
    for step_idx, lambda_t in enumerate(lambda_schedule):

        # (a) WARM START — overwrite model weights with previous step's result.
        #     Optimizer and scheduler are NOT carried over (stale Adam moments
        #     from the previous lambda would fight the proximal operator).
        model.load_state_dict(prev_state_dict)

        # (b) FRESH optimizer + scheduler for this lambda step.
        #     weight_decay is kept at the config-9 value (1e-5); zeroing it here
        #     would change gradient dynamics relative to the dense phase.
        #     mode="min" because the patience criterion watches val_loss; both
        #     the scheduler and patience watch the same signal for consistency.
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sched_factor,
            patience=sched_patience, min_lr=1e-6,
        )

        # (c) Per-lambda fine-tuning with patience on val_loss.
        best_step_val_loss = float("inf")
        no_improve_ctr     = 0
        actual_epochs      = 0
        # Track last-epoch metrics to avoid a redundant _eval() after the loop.
        last_balacc = last_auc = last_val_loss = float("nan")

        for epoch in range(max_warm_start_epochs):
            # ── Training pass ─────────────────────────────────────────────────
            model.train()
            for X_b, y_b in _make_loader(train_dataset, batch_size, True, pin_memory):
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()

                if concurvity_lambda > 0:
                    logits, shape_outs = model(X_b, return_shape_outputs=True)
                    loss = (criterion(logits, y_b)
                            + concurvity_lambda * multiclass_concurvity(shape_outs))
                else:
                    loss = criterion(model(X_b), y_b)
                # Sparsity penalty is NOT added to the loss.  Regularization
                # happens exclusively through apply_proximal_step() below.

                loss.backward()
                optimizer.step()

                # PROXIMAL STEP — fires immediately after optimizer.step().
                # lr must be the CURRENT value; ReduceLROnPlateau may have
                # decayed it during previous epochs of this lambda step.
                apply_proximal_step(
                    model,
                    lr=optimizer.param_groups[0]["lr"],
                    sparsity_lambda=lambda_t,
                )

            actual_epochs += 1

            # ── Validation pass ────────────────────────────────────────────────
            last_balacc, last_auc, last_val_loss = _eval()
            scheduler.step(last_val_loss)

            # ── Patience check on val_loss ─────────────────────────────────────
            if last_val_loss < best_step_val_loss - warm_start_min_delta:
                best_step_val_loss = last_val_loss
                no_improve_ctr     = 0
            else:
                no_improve_ctr += 1
                if no_improve_ctr >= warm_start_patience:
                    break   # done with this lambda step

        # (d) Final metrics — reuse last-epoch values from the loop above.
        val_balacc = last_balacc
        val_auc    = last_auc
        val_loss   = last_val_loss

        norms    = feature_group_norms(model)
        # theta_k = flattened concatenation of all trainable params of subnet k,
        # which is exactly the vector used by apply_proximal_step.  See sparsity.py.
        n_active = sum(1 for v in norms.values() if v > 1e-8)

        # (e) Detect eliminations: active → zeroed this step.
        just_eliminated: list[str] = []
        for k in concept_names:
            if prev_norms[k] >= 1e-8 and norms[k] < 1e-8:
                just_eliminated.append(k)
                elim_order.append((k, lambda_t))
                selected_lams[k] = lambda_t

        # (f) Detect re-activations: should not occur with proximal operator;
        #     print a diagnostic warning if it does.
        for k in concept_names:
            if prev_norms[k] < 1e-8 and norms[k] >= 1e-8:
                print(
                    f"WARNING: concept '{k}' re-activated at lambda={lambda_t:.4e} "
                    f"(prev={prev_norms[k]:.2e} → curr={norms[k]:.2e})",
                    file=sys.stderr,
                )

        # (g) Write per-lambda checkpoint + metrics to disk.
        lam_tag   = f"{lambda_t:.6e}"
        step_pt   = os.path.join(out_dir, "checkpoints", f"seed{seed}_lambda{lam_tag}.pt")
        step_json = os.path.join(out_dir, "checkpoints", f"seed{seed}_lambda{lam_tag}.json")
        torch.save(model.state_dict(), step_pt)
        json.dump(
            {
                "lambda":          lambda_t,
                "n_active":        n_active,
                "val_balacc":      val_balacc,
                "val_auc":         val_auc,
                "val_loss":        val_loss,
                "actual_epochs":   actual_epochs,
                "norms":           norms,
                "just_eliminated": just_eliminated,
            },
            open(step_json, "w"),
            indent=2,
        )

        # (h) Accumulate in-memory.
        state_dicts.append(model.state_dict())   # .state_dict() returns a copy
        rows.append({
            "lambda":                   lambda_t,
            "n_active":                 n_active,
            "val_loss":                 val_loss,
            "val_balacc":               val_balacc,
            "val_auc":                  val_auc,
            "actual_epochs":            actual_epochs,
            "concepts_just_eliminated": ",".join(just_eliminated),
            **{f"norm_{k}": norms[k] for k in concept_names},
        })

        # Progress line: always print on eliminations; else every 10 steps.
        if just_eliminated or (step_idx + 1) % 10 == 0:
            elim_str = f"  ELIMINATED: {just_eliminated}" if just_eliminated else ""
            print(f"  step {step_idx + 1:3d}  lambda={lambda_t:.4e}  "
                  f"n_active={n_active:2d}  val_balacc={val_balacc:.4f}  "
                  f"val_auc={val_auc:.4f}  epochs={actual_epochs}{elim_str}")

        # (i) Advance state.
        prev_state_dict = model.state_dict()
        prev_norms      = norms

        # (j) Stop if all subnets zeroed.
        if n_active == 0:
            print(f"[path]  All {N_FEATURES} subnets zeroed at "
                  f"lambda={lambda_t:.4e}. Stopping.")
            break

    # ── 5. Write path CSV ──────────────────────────────────────────────────────
    # First line is a comment with the seed so the file is self-documenting.
    # Read back with: pd.read_csv(path, comment="#")
    path_df  = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"path_seed{seed}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# seed={seed}\n")
        path_df.to_csv(f, index=False)

    # ── 6. Write elimination table ─────────────────────────────────────────────
    elim_path = os.path.join(out_dir, f"path_seed{seed}_elimination.txt")
    with open(elim_path, "w", encoding="utf-8") as f:
        f.write(f"seed={seed}\n")
        f.write(f"dense_val_balacc={dense_val_balacc:.4f}  "
                f"dense_val_auc={dense_val_auc:.4f}\n\n")
        f.write(f"{'order':>5}  {'lambda':>12}  concept\n")
        f.write("-" * 35 + "\n")
        for i, (name, lam) in enumerate(elim_order, 1):
            f.write(f"{i:>5}  {lam:>12.4e}  {name}\n")
        never_zeroed = [k for k in concept_names if k not in selected_lams]
        if never_zeroed:
            f.write(f"\nNever zeroed within lambda <= {max_lambda:.2e}:\n")
            for k in never_zeroed:
                f.write(f"  {k}\n")

    # ── 7. Return ──────────────────────────────────────────────────────────────
    norms_array = np.array(
        [[row[f"norm_{k}"] for k in concept_names] for row in rows],
        dtype=np.float32,
    )  # shape: (n_steps_run, 24)

    return {
        "seed":                        seed,
        "lambda_schedule":             [r["lambda"]        for r in rows],
        "active_count_per_lambda":     [r["n_active"]      for r in rows],
        "theta_norms_per_lambda":      norms_array,
        "val_balanced_acc_per_lambda": [r["val_balacc"]    for r in rows],
        "val_auc_per_lambda":          [r["val_auc"]       for r in rows],
        "val_loss_per_lambda":         [r["val_loss"]      for r in rows],
        "actual_epochs_per_lambda":    [r["actual_epochs"] for r in rows],
        "state_dicts":                 state_dicts,
        "elimination_order":           elim_order,
        "selected_lambdas":            selected_lams,
        "dense_val_balacc":            dense_val_balacc,
        "dense_val_auc":               dense_val_auc,
        "concept_names":               concept_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_regularization_path(
    results:       dict,
    seed:          int,
    concept_names: list,
    out_dir:       str,
) -> None:
    """2-panel regularization path figure.

    Panel 1 (top): ||theta_k||_2 vs lambda for each of the 24 concepts.
      Log x-axis.  Horizontal dotted line at 1e-8 (zero threshold).
    Panel 2 (bottom): val balanced acc and val AUC (left y-axis),
      number of active subnets (right y-axis, step plot).

    Saved to out_dir/path_seed{seed}.png.
    """
    lambdas    = results["lambda_schedule"]
    norms      = results["theta_norms_per_lambda"]   # (n_steps, 24)
    n_active   = results["active_count_per_lambda"]
    val_balacc = results["val_balanced_acc_per_lambda"]
    val_auc    = results["val_auc_per_lambda"]
    dense_balacc = results["dense_val_balacc"]
    dense_auc    = results["dense_val_auc"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    cmap = plt.cm.tab20
    for i, name in enumerate(concept_names):
        ax1.plot(lambdas, norms[:, i],
                 color=cmap(i / len(concept_names)),
                 linewidth=1.0, label=name)
    ax1.axhline(1e-8, color="gray", linewidth=0.5, linestyle=":",
                label="_zero threshold")
    ax1.set_xscale("log")
    ax1.set_ylabel("||θ_k||₂")
    ax1.set_title(f"Group LASSO regularization path — warm start (seed {seed})")
    ax1.legend(fontsize=6, ncol=3, loc="upper right")

    ax2.axhline(dense_balacc, color="steelblue", linewidth=0.8,
                linestyle="--", alpha=0.6, label=f"dense balacc ({dense_balacc:.3f})")
    ax2.axhline(dense_auc, color="darkorange", linewidth=0.8,
                linestyle="--", alpha=0.6, label=f"dense AUC ({dense_auc:.3f})")
    ax2.plot(lambdas, val_balacc, color="steelblue",  linewidth=1.5,
             label="val balanced acc")
    ax2.plot(lambdas, val_auc,    color="darkorange", linewidth=1.5,
             label="val AUC (OvR wtd)")
    ax2.set_xlabel("λ (log scale)")
    ax2.set_ylabel("metric")
    ax2.set_xscale("log")

    ax2r = ax2.twinx()
    ax2r.step(lambdas, n_active, color="black", linewidth=1.5,
              alpha=0.7, where="mid", label="n_active")
    ax2r.set_ylabel("# active subnets")
    ax2r.set_ylim(-0.5, len(concept_names) + 0.5)

    lines  = ax2.get_lines() + ax2r.get_lines()
    labels = [l.get_label() for l in lines if not l.get_label().startswith("_")]
    ax2.legend(lines[:len(labels)], labels, fontsize=8, loc="center right")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"path_seed{seed}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot]  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(
    results:       dict,
    seed:          int,
    n_rows:        int = 20,
) -> None:
    """Print a ~20-row subsample of the path, spread uniformly."""
    lambdas      = results["lambda_schedule"]
    n_active     = results["active_count_per_lambda"]
    val_bacc     = results["val_balanced_acc_per_lambda"]
    val_auc      = results["val_auc_per_lambda"]
    concept_names = results["concept_names"]

    # lambda → list of concepts eliminated at that exact lambda value
    lam_to_elim: dict[float, list] = {}
    for name, lam in results["elimination_order"]:
        lam_to_elim.setdefault(lam, []).append(name)

    total = len(lambdas)
    idxs  = sorted({
        0,
        *[int(round(i * (total - 1) / (n_rows - 1))) for i in range(1, n_rows - 1)],
        total - 1,
    })

    sep    = "-" * 80
    header = (f"{'lambda':>12}  {'n_active':>8}  {'val_bal_acc':>11}  "
              f"{'val_auc':>8}  concepts_just_eliminated")
    print(f"\n{sep}")
    print(f"  Regularization path — subsample (seed={seed}, "
          f"{total} total steps)")
    print(f"  Dense baseline: val_balacc={results['dense_val_balacc']:.4f}  "
          f"val_auc={results['dense_val_auc']:.4f}")
    print(sep)
    print(header)
    print(sep)
    for i in idxs:
        lam  = lambdas[i]
        elim = ", ".join(lam_to_elim.get(lam, []))
        print(f"{lam:>12.4e}  {n_active[i]:>8d}  {val_bacc[i]:>11.4f}  "
              f"{val_auc[i]:>8.4f}  {elim}")
    print(sep)


def print_elimination_order(results: dict) -> None:
    elim_order    = results["elimination_order"]
    concept_names = results["concept_names"]
    print(f"\nElimination order ({len(elim_order)} of {len(concept_names)} "
          f"concepts dropped):")
    print(f"{'order':>5}  {'lambda':>12}  concept")
    print("-" * 40)
    for i, (name, lam) in enumerate(elim_order, 1):
        print(f"{i:>5}  {lam:>12.4e}  {name}")
    never = [k for k in concept_names if k not in results["selected_lambdas"]]
    if never:
        print(f"\nNever zeroed within the swept lambda range:")
        for k in never:
            print(f"  {k}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sweep_mode", choices=["cold_start", "warm_start"], default="cold_start",
        help="cold_start (default): subprocess launcher. warm_start: in-process path.",
    )
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--concurvity_lambda", type=float, default=1.0,
                        help="Concurvity regularization weight, fixed for entire path. "
                             "Default 1.0 matches the elbow from concurvity-only sweep. "
                             "Override to 0.0 to disable concurvity reg for ablations.")
    parser.add_argument("--max_epochs",        type=int,   default=None,
                        help="Max epochs for dense phase (warm) or per run (cold). "
                             "Default 100 for warm, unlimited for cold.")
    parser.add_argument("--skip_convergence_check", action="store_true")

    cold_grp = parser.add_argument_group("cold_start options")
    cold_grp.add_argument(
        "--sparsity_lambdas", type=str, default=None,
        help="Comma- or space-separated lambda values. "
             f"Default: {COLD_START_DEFAULT_LAMBDAS}",
    )
    cold_grp.add_argument(
        "--cold_out_dir", type=str, default="results/cold_start_sparsity",
    )

    warm_grp = parser.add_argument_group("warm_start options")
    warm_grp.add_argument("--lambda_0",              type=float, default=1e-3)
    warm_grp.add_argument("--epsilon",               type=float, default=0.02,
                          help="Path multiplier: lambda_t = lambda_0*(1+epsilon)^t")
    warm_grp.add_argument("--max_lambda",            type=float, default=1e3)
    warm_grp.add_argument("--max_lambda_steps",      type=int,   default=300)
    warm_grp.add_argument("--max_warm_start_epochs", type=int,   default=50)
    warm_grp.add_argument("--warm_start_patience",   type=int,   default=10)
    warm_grp.add_argument("--warm_start_min_delta",  type=float, default=1e-4)
    warm_grp.add_argument("--out_dir", type=str, default="results/warm_start")

    args = parser.parse_args()

    if args.sweep_mode == "warm_start" and args.sparsity_lambdas is not None:
        print(
            "WARNING: --sparsity_lambdas is ignored in warm_start mode. "
            "The schedule is generated from --lambda_0, --epsilon, "
            "--max_lambda, --max_lambda_steps.",
            file=sys.stderr,
        )

    if args.sweep_mode == "cold_start":
        run_cold_start_sweep(args)
        return

    # ── warm_start ────────────────────────────────────────────────────────────
    if DEVICE.type == "cpu":
        print("WARNING: CUDA not available — running on CPU. "
              "Expect ~10–30 min total on this hardware.")
    else:
        print(f"Using device: {DEVICE}")

    print("Loading data...")
    data = load_ham10000(device=DEVICE)
    print(f"  train: {data['X_train_t'].shape[0]:5d}  "
          f"val: {data['X_val_t'].shape[0]:5d}  "
          f"test: {data['X_test_t'].shape[0]:5d}")

    print("Loading sweep hyperparams (config 9)...")
    hidden_dims, dropout, weight_decay = load_sweep_hyperparams()
    print(f"  hidden={list(hidden_dims)}  dropout={dropout}  wd={weight_decay:.0e}")

    max_dense_epochs = args.max_epochs if args.max_epochs is not None else 100

    results = run_warm_started_sweep(
        data=data,
        hidden_dims=hidden_dims,
        dropout=dropout,
        weight_decay=weight_decay,
        seed=args.seed,
        max_dense_epochs=max_dense_epochs,
        skip_convergence_check=args.skip_convergence_check,
        lambda_0=args.lambda_0,
        epsilon=args.epsilon,
        max_lambda=args.max_lambda,
        max_lambda_steps=args.max_lambda_steps,
        max_warm_start_epochs=args.max_warm_start_epochs,
        warm_start_patience=args.warm_start_patience,
        warm_start_min_delta=args.warm_start_min_delta,
        concurvity_lambda=args.concurvity_lambda,
        out_dir=args.out_dir,
        device=DEVICE,
    )

    plot_regularization_path(results, args.seed, data["concept_names"], args.out_dir)
    print_summary_table(results, seed=args.seed)
    print_elimination_order(results)

    print(f"\nDone. All outputs under {args.out_dir}/")


if __name__ == "__main__":
    main()
