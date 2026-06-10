"""
Concurvity lambda sweep — chest X-ray three-way task (STEP 3).

Mirrors scripts/v7/run_concurvity_sweep.py for the chest X-ray dataset
(Normal / Bacteria / Virus, 3 classes, 17 features, Config 10 architecture).
Single seed (42) across 10 concurvity lambdas.  Val-only selection; test
metrics logged per lambda for downstream reporting (test_set_touched=True).

Records TWO candidate winners:
  best_val_balacc_lambda  — highest val_balacc, tiebreak lowest R_perp
  rule_a_lambda           — largest lambda where val_balacc >= baseline - 0.02

The operative lambda for STEP 4 is chosen by the user after reviewing both.

Hard isolation rules
────────────────────
- Does NOT import _common.load_raw_data (HAM10000-specific).
- Does NOT touch any HAM10000 artefact, v1–v3 features, or architecture_selection/
  or plain_nam/ output.
- Test indices loaded upfront for logging but NEVER enter the selection rule.
- Val split identical to train_final.py (GroupShuffleSplit, random_state=42).
- Concurvity active from epoch 1 (warmup_epochs=0, per HAM10000 v7 diagnostic finding).

Usage (from project root):
    python scripts/chestxray/run_concurvity_sweep.py
    python scripts/chestxray/run_concurvity_sweep.py --smoke_test   # lambda=0.0 only
    python scripts/chestxray/run_concurvity_sweep.py --sanity_only
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
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from scripts.v7._common import (
    set_all_seeds,
    make_fixed_val_split,
    make_optimizer_scheduler,
    train_one_run,
)
from src.models.nam_multiclass import NAMMulticlass
from src.models.sparsity import feature_group_norms

# ── Constants ──────────────────────────────────────────────────────────────────
SEED       = 42
LR         = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE   = 15
SCHED_PAT  = 5
SCHED_FAC  = 0.5
N_FEATURES  = 17
NUM_CLASSES = 3
VAL_RANDOM_STATE = 42
ZERO_THRESHOLD   = 1e-4
RULE_A_TOLERANCE = 0.02   # val_balacc may drop by at most this vs lambda=0 baseline

LAMBDAS = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]

FEATURES_PATH  = "data/features/biomedclip/chestxray_concept_scores_v4.npz"
SPLIT_PATH     = "data/splits/chestxray_outer_split.npz"
LABEL_MAP_PATH = "results/chestxray/architecture_selection/label_mapping.json"
WINNER_JSON    = "results/chestxray/architecture_selection/winning_config.json"
PLAIN_NAM_CSV  = "results/chestxray/plain_nam/aggregated_metrics.csv"
OUT_ROOT       = "results/chestxray/concurvity_sweep"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (chest-X-ray-specific; mirrors train_final.py conventions)
# ─────────────────────────────────────────────────────────────────────────────

def load_label_mapping() -> dict:
    """Load SUBTYPE_TO_INT from label_mapping.json; fall back to hardcoded."""
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"normal": 0, "bacteria": 1, "virus": 2}


def load_data(subtype_to_int: dict) -> dict:
    """Load feature matrix and split indices.

    Mirrors train_final.load_data().  Returns test_idx alongside train_pool
    so that per-lambda test logging is possible — but test_idx is NEVER used
    for selection decisions.
    """
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


def evaluate_on_test(
    model:        NAMMulticlass,
    X_test_t:     torch.Tensor,
    y_test:       np.ndarray,   # int64, values 0/1/2
    class_names:  list,         # ["normal", "bacteria", "virus"]
) -> dict:
    """Integer-label test evaluation matching train_final.evaluate_on_test."""
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()

    labels_list  = list(range(NUM_CLASSES))
    bal_acc      = float(balanced_accuracy_score(y_test, preds))
    macro_f1     = float(f1_score(y_test, preds, average="macro",    zero_division=0))
    weighted_f1  = float(f1_score(y_test, preds, average="weighted", zero_division=0))
    top1_acc     = float(accuracy_score(y_test, preds))
    macro_auc    = float(roc_auc_score(
        y_test, proba, multi_class="ovr", average="macro",    labels=labels_list
    ))
    weighted_auc = float(roc_auc_score(
        y_test, proba, multi_class="ovr", average="weighted", labels=labels_list
    ))
    return {
        "balanced_accuracy": bal_acc,
        "macro_f1":          macro_f1,
        "weighted_f1":       weighted_f1,
        "top1_accuracy":     top1_acc,
        "macro_auc_ovr":     macro_auc,
        "weighted_auc_ovr":  weighted_auc,
    }


# ─────────────────────────────────────────────────────────────────────────────

def lam_tag(lam_c: float) -> str:
    """Folder-safe lambda string, matching v7 convention."""
    return f"{lam_c:.6f}".rstrip("0").rstrip(".")


# ─────────────────────────────────────────────────────────────────────────────

def run_one_lambda(
    *,
    lam_c:              float,
    hidden_dims:        tuple,
    dropout:            float,
    weight_decay:       float,
    X_train_final_raw:  np.ndarray,   # inner-train features (unscaled)
    X_val_raw:          np.ndarray,   # val features (unscaled)
    y_train_final:      np.ndarray,   # int64 labels for inner-train
    y_val:              np.ndarray,   # int64 labels for val
    X_test_raw:         np.ndarray,   # test features (unscaled)
    y_test:             np.ndarray,   # int64 labels for test
    concept_names:      list,
    class_names:        list,         # ["normal", "bacteria", "virus"]
    out_dir:            str,
) -> dict:
    """Train seed=42 at a single concurvity lambda.  Returns metrics dict."""
    os.makedirs(out_dir, exist_ok=True)
    seed_dir = os.path.join(out_dir, f"seed_{SEED}")
    os.makedirs(seed_dir, exist_ok=True)

    set_all_seeds(SEED)

    # ── Per-lambda scaler (fit on inner-train only) ────────────────────────────
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train_final_raw).astype(np.float32)
    X_val_sc  = scaler.transform(X_val_raw).astype(np.float32)
    X_test_sc = scaler.transform(X_test_raw).astype(np.float32)
    with open(os.path.join(seed_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # ── Class weights (inverse-frequency on inner-train) ──────────────────────
    counts   = np.bincount(y_train_final, minlength=NUM_CLASSES)
    n_tr     = len(y_train_final)
    weights  = n_tr / (NUM_CLASSES * counts.astype(np.float64))
    w_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    # ── Model, optimiser, criterion ───────────────────────────────────────────
    model = NAMMulticlass(
        n_features=N_FEATURES,
        num_classes=NUM_CLASSES,
        hidden_dims=hidden_dims,
        dropout=dropout,
        concept_names=concept_names,
    ).to(DEVICE)
    optimizer, scheduler = make_optimizer_scheduler(
        model, LR, weight_decay, SCHED_PAT, SCHED_FAC, scheduler_mode="max"
    )
    criterion = nn.CrossEntropyLoss(weight=w_tensor)
    ckpt_path = os.path.join(seed_dir, "best_model.pt")

    X_val_t  = torch.tensor(X_val_sc,  dtype=torch.float32, device=DEVICE)
    y_val_t  = torch.tensor(y_val,     dtype=torch.long,    device=DEVICE)
    X_test_t = torch.tensor(X_test_sc, dtype=torch.float32, device=DEVICE)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc,       dtype=torch.float32),
        torch.tensor(y_train_final, dtype=torch.long),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    result = train_one_run(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_dataset=train_ds,
        X_val_t=X_val_t,
        y_val_t=y_val_t,
        y_val_enc=y_val,            # int64 for balanced_accuracy_score
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        concurvity_lambda=lam_c,
        warmup_epochs=0,            # no warm-up: HAM10000 v7 diagnostic finding
        sparsity_lambda=0.0,
        proximal_sparsity=True,
        save_path=ckpt_path,
        verbose_every=10,
        scheduler_watches="val_balacc",
    )

    # Save training log
    result["log_df"].to_csv(os.path.join(seed_dir, "training_log.csv"), index=False)

    # ── Feature group norms at best checkpoint ────────────────────────────────
    norms = feature_group_norms(model)
    norms_rows = [
        {"concept_name": nm, "norm": nv}
        for nm, nv in norms.items()
    ]
    pd.DataFrame(norms_rows).to_csv(
        os.path.join(seed_dir, "feature_group_norms.csv"), index=False
    )

    # ── R_perp at best val epoch (from training log) ──────────────────────────
    log_df   = result["log_df"]
    best_idx = log_df["val_balanced_acc"].idxmax()
    r_perp_val_at_best  = float(log_df.loc[best_idx, "r_perp_val"])
    r_perp_tr_at_best   = float(log_df.loc[best_idx, "r_perp_train"])

    # ── Test evaluation (logged but NOT used for selection) ───────────────────
    test_metrics = evaluate_on_test(model, X_test_t, y_test, class_names)

    # ── metrics.json ──────────────────────────────────────────────────────────
    row = {
        "concurvity_lambda":    lam_c,
        "best_val_balacc":      float(result["best_val_balacc"]),
        "best_epoch":           int(result["best_epoch"]),
        "total_epochs":         int(result["total_epochs"]),
        "r_perp_val_at_best":   r_perp_val_at_best,
        "r_perp_train_at_best": r_perp_tr_at_best,
        "test_balacc":          test_metrics["balanced_accuracy"],
        "test_macro_f1":        test_metrics["macro_f1"],
        "test_macro_auc":       test_metrics["macro_auc_ovr"],
        "test_weighted_auc":    test_metrics["weighted_auc_ovr"],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(
    data:           dict,
    subtype_to_int: dict,
    val_split:      dict,
) -> None:
    """Run and print pre-sweep sanity checks 1–4.  sys.exit on failure."""
    print("\n" + "=" * 65)
    print("PRE-SWEEP SANITY CHECKS")
    print("=" * 65)

    # 1. Plain NAM reference loads correctly
    if os.path.exists(PLAIN_NAM_CSV):
        plain_df = pd.read_csv(PLAIN_NAM_CSV)
        mean_row = plain_df[plain_df["seed"] == "mean"]
        plain_mean_bal = float(mean_row["balanced_accuracy"].values[0]) if len(mean_row) else float("nan")
        plain_std_bal  = float(plain_df[plain_df["seed"] == "std"]["balanced_accuracy"].values[0]) if len(plain_df[plain_df["seed"] == "std"]) else float("nan")
        print(f"  [1] Plain NAM 5-seed test bal_acc: {plain_mean_bal:.4f} ± {plain_std_bal:.4f}")
        print(f"      (Reference for comparison; NOT the Rule A baseline.)")
        print(f"      Rule A baseline = sweep's lambda=0 val_balacc (single-seed, val-set).")
    else:
        print(f"  [1] WARN: {PLAIN_NAM_CSV} not found — skipping reference load.")

    # 2. Architecture sanity
    with open(WINNER_JSON, encoding="utf-8") as f:
        wc = json.load(f)
    ok = (wc["hidden_dims"] == [64, 32] and
          abs(wc["dropout"] - 0.1) < 1e-9 and
          abs(wc["weight_decay"] - 1e-4) < 1e-9)
    if not ok:
        print(f"  [2] FAIL: winning_config mismatch: {wc['hidden_dims']}, "
              f"drop={wc['dropout']}, wd={wc['weight_decay']}")
        sys.exit(1)
    print(f"  [2] Winning config (id={wc['config_id']}): hidden={wc['hidden_dims']}, "
          f"dropout={wc['dropout']}, wd={wc['weight_decay']}  ✓")

    # 3. Val split reproducibility — must match plain_nam run
    EXPECTED_TRAIN10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    EXPECTED_VAL10   = [29, 33, 47, 48, 52, 53, 55, 63, 64, 65]
    train_rel = val_split["train_rel"]
    val_rel   = val_split["val_rel"]
    got_tr10 = train_rel[:10].tolist()
    got_val10 = val_rel[:10].tolist()
    if got_tr10 != EXPECTED_TRAIN10 or got_val10 != EXPECTED_VAL10:
        print(f"  [3] FAIL: val-split mismatch!")
        print(f"      expected train[:10] = {EXPECTED_TRAIN10}")
        print(f"      got      train[:10] = {got_tr10}")
        print(f"      expected val[:10]   = {EXPECTED_VAL10}")
        print(f"      got      val[:10]   = {got_val10}")
        print(f"      Apples-to-apples comparison requires identical val split.")
        sys.exit(1)
    print(f"  [3] Val-split indices match plain_nam run  ✓")
    print(f"      train_final[:10] = {got_tr10}")
    print(f"      val[:10]         = {got_val10}")
    print(f"      train_final size = {len(train_rel)}, val size = {len(val_rel)}")

    # 4. Class weight values — must match plain_nam run seed-42 weights
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
            print(f"  [4] {int_to_name[i]}: got {weights[i]:.4f}, expected ≈ {EXPECTED_W[int_to_name[i]]}")
        print(f"  [4] FAIL: class weights diverge from plain_nam reference.")
        sys.exit(1)
    print(f"  [4] Class weights (inner-train, n={n_tr}):")
    for i in range(NUM_CLASSES):
        print(f"        {int_to_name[i]:10s}: n={counts[i]}, weight={weights[i]:.4f}  ✓")

    print("=" * 65)
    print("All pre-sweep sanity checks passed.\n")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run lambda=0.0 only to verify mechanics.")
    parser.add_argument("--sanity_only", action="store_true",
                        help="Run sanity checks then exit without training.")
    parser.add_argument("--winner_json", type=str, default=None)
    parser.add_argument("--out_root",   type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience",   type=int, default=None)
    args = parser.parse_args()

    max_ep = args.max_epochs if args.max_epochs is not None else MAX_EPOCHS
    pat    = args.patience   if args.patience   is not None else PATIENCE

    # ── Load winning config ────────────────────────────────────────────────────
    winner_json_path = args.winner_json or WINNER_JSON
    if not os.path.exists(winner_json_path):
        raise FileNotFoundError(
            f"winning_config.json not found at {winner_json_path}. "
            "Run select_architecture.py (STEP 1) first."
        )
    with open(winner_json_path, encoding="utf-8") as f:
        winner = json.load(f)
    hidden_dims  = tuple(winner["hidden_dims"])
    dropout      = float(winner["dropout"])
    weight_decay = float(winner["weight_decay"])

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
    test_idx      = data["test_idx"]
    patient_ids   = data["patient_ids"]

    X_train_pool = scores[train_pool_idx]
    y_train_pool = labels_all[train_pool_idx]
    groups_pool  = patient_ids[train_pool_idx]

    # ── Fixed val split (identical to train_final.py; fixed across all lambdas) ──
    val_split = make_fixed_val_split(
        X_train_pool,
        y_train_pool.astype(str),
        groups_pool,
        ["0", "1", "2"],
        val_random_state=VAL_RANDOM_STATE,
    )
    train_rel = val_split["train_rel"]
    val_rel   = val_split["val_rel"]

    # ── Pre-sweep sanity checks ────────────────────────────────────────────────
    run_sanity_checks(data, subtype_to_int, val_split)

    if args.sanity_only:
        print("--sanity_only flag set.  Exiting before sweep.")
        return

    # ── Slice arrays (used by every lambda) ───────────────────────────────────
    X_train_final_raw = X_train_pool[train_rel]
    X_val_raw         = X_train_pool[val_rel]
    y_train_final     = y_train_pool[train_rel]   # int64
    y_val             = y_train_pool[val_rel]      # int64
    X_test_raw        = scores[test_idx]
    y_test            = labels_all[test_idx]       # int64 (test, NOT used for selection)

    lambdas = [0.0] if args.smoke_test else LAMBDAS

    # ── Banner ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Chest X-ray NAM — Concurvity sweep (STEP 3)")
    print(f"  Concurvity active from epoch 1 (warmup_epochs=0, per HAM10000 v7 diagnostic finding)")
    print(f"  Config (id={winner['config_id']}): hidden={list(hidden_dims)}, "
          f"dropout={dropout}, wd={weight_decay:.0e}")
    print(f"  Lambdas ({len(lambdas)}): {lambdas}")
    print(f"  seed={SEED}, max_epochs={max_ep}, patience={pat}, batch={BATCH_SIZE}")
    print(f"  val_random_state={VAL_RANDOM_STATE}  (identical to plain_nam val split)")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {out_root}/")
    if args.smoke_test:
        print("  [SMOKE TEST — lambda=0.0 only]")
    print(f"{'='*65}\n")

    # ── Sweep loop ─────────────────────────────────────────────────────────────
    all_rows = []
    t0       = time.time()

    for lam_c in lambdas:
        tag     = lam_tag(lam_c)
        lam_dir = os.path.join(out_root, f"lambda_{tag}")
        t_lam   = time.time()
        print(f"\n── lambda_c = {lam_c} " + "─" * 45)

        row = run_one_lambda(
            lam_c=lam_c,
            hidden_dims=hidden_dims,
            dropout=dropout,
            weight_decay=weight_decay,
            X_train_final_raw=X_train_final_raw,
            X_val_raw=X_val_raw,
            y_train_final=y_train_final,
            y_val=y_val,
            X_test_raw=X_test_raw,
            y_test=y_test,
            concept_names=concept_names,
            class_names=class_names,
            out_dir=lam_dir,
        )
        elapsed_lam = time.time() - t_lam
        all_rows.append(row)
        print(
            f"  Done ({elapsed_lam:.0f}s): val_balacc={row['best_val_balacc']:.4f}  "
            f"val_R_perp={row['r_perp_val_at_best']:.4f}  "
            f"test_balacc={row['test_balacc']:.4f}"
        )

        # ── Sanity check 5: R_perp at lambda=0 convergence ────────────────────
        if lam_c == 0.0:
            rp = row["r_perp_val_at_best"]
            print(f"\n  [Sanity 5] lambda=0 convergence R_perp_val = {rp:.4f}")
            if 0.20 <= rp <= 0.45:
                print(f"             In expected range [0.20, 0.45]  ✓")
            else:
                print(f"             ⚠ Outside expected range [0.20, 0.45] — "
                      f"investigate if unexpected (HAM10000 chest X-ray reference ~0.30–0.33)")

    # ── Summary CSV ────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(os.path.join(out_root, "summary.csv"), index=False)

    # ── Selection rules ────────────────────────────────────────────────────────
    # Rule best_val_balacc: highest val_balacc, tiebreak lowest R_perp (mirrors v7 lines 293-311)
    best_row = summary_df.sort_values(
        ["best_val_balacc", "r_perp_val_at_best"],
        ascending=[False, True],
    ).iloc[0]
    best_val_lam = float(best_row["concurvity_lambda"])

    # Rule A: largest lambda where val_balacc >= baseline(lambda=0) - RULE_A_TOLERANCE
    lam0_row   = summary_df[summary_df["concurvity_lambda"] == 0.0].iloc[0]
    lam0_balacc = float(lam0_row["best_val_balacc"])
    threshold   = lam0_balacc - RULE_A_TOLERANCE

    rule_a_candidates = summary_df[
        summary_df["best_val_balacc"] >= threshold
    ].sort_values("concurvity_lambda", ascending=False)

    rule_a_degenerate = False
    if len(rule_a_candidates) == 0:
        rule_a_lam = 0.0
        rule_a_degenerate = True
    else:
        rule_a_lam = float(rule_a_candidates.iloc[0]["concurvity_lambda"])
        if rule_a_lam == 0.0:
            rule_a_degenerate = True

    # Compute delta vs lambda=0 for each row
    deltas = summary_df["best_val_balacc"] - lam0_balacc

    # ── winner.json (records both candidates, no operative lambda chosen) ──────
    winner_out: dict = {
        "best_val_balacc_lambda":   best_val_lam,
        "best_val_balacc_value":    float(best_row["best_val_balacc"]),
        "best_val_balacc_r_perp":   float(best_row["r_perp_val_at_best"]),
        "best_val_balacc_criterion": "highest val_balacc at seed=42; tie-break lowest R_perp",
        "rule_a_lambda":            rule_a_lam,
        "rule_a_threshold":         RULE_A_TOLERANCE,
        "lambda_zero_val_balacc":   lam0_balacc,
        "test_set_touched":         True,
        "selection_pending":        True,
        "selection_note": (
            "Two candidate lambda_c values recorded. "
            "Operative lambda_c for STEP 4 (concurvity_only 5-seed final) "
            "to be selected manually after reviewing both candidates and the "
            "summary table."
        ),
    }
    if rule_a_degenerate:
        winner_out["rule_a_degenerate"] = True
        winner_out["rule_a_degenerate_note"] = (
            "Rule A degenerates to lambda=0: even the smallest non-zero lambda "
            f"drops val_balacc by more than {RULE_A_TOLERANCE} relative to the "
            "lambda=0 baseline.  This is itself a reportable finding — concurvity "
            "regularisation has a meaningful cost on this dataset."
        )

    if not args.smoke_test:
        with open(os.path.join(out_root, "winner.json"), "w", encoding="utf-8") as f:
            json.dump(winner_out, f, indent=2)

    # ── run_config.json ────────────────────────────────────────────────────────
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT), text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_hash = None

    meta = {
        "lambdas":       lambdas,
        "seed":          SEED,
        "config_id":     winner["config_id"],
        "hidden_dims":   list(hidden_dims),
        "dropout":       dropout,
        "weight_decay":  weight_decay,
        "lr":            LR,
        "batch_size":    BATCH_SIZE,
        "max_epochs":    max_ep,
        "patience":      pat,
        "val_random_state": VAL_RANDOM_STATE,
        "warmup_epochs": 0,
        "sparsity_lambda": 0.0,
        "num_classes":   NUM_CLASSES,
        "n_features":    N_FEATURES,
        "feature_file":  FEATURES_PATH,
        "split_file":    SPLIT_PATH,
        "smoke_test":    args.smoke_test,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "git_commit":    git_hash,
        "winner_json":   winner_json_path,
    }
    with open(os.path.join(out_root, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ── STEP_3_COMPLETE flag ───────────────────────────────────────────────────
    if not args.smoke_test:
        flag_path = os.path.join(out_root, "STEP_3_COMPLETE.flag")
        with open(flag_path, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat() + "\n")

    # ── Summary printout ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"Chest X-ray NAM — Concurvity sweep complete ({len(all_rows)} lambdas)")
    print(f"")
    print(f"  {'lambda_c':>10}  {'val_balacc':>10}  {'val_R_perp':>10}  "
          f"{'test_balacc':>11}  {'Δ_vs_lam0':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*10}")
    for idx, row in summary_df.iterrows():
        delta = row["best_val_balacc"] - lam0_balacc
        rule_a_flag = " ← Rule A" if row["concurvity_lambda"] == rule_a_lam else ""
        bv_flag     = " ← best-val" if row["concurvity_lambda"] == best_val_lam else ""
        flags = (rule_a_flag + bv_flag).strip()
        print(
            f"  {row['concurvity_lambda']:>10}  "
            f"{row['best_val_balacc']:>10.4f}  "
            f"{row['r_perp_val_at_best']:>10.4f}  "
            f"{row['test_balacc']:>11.4f}  "
            f"{delta:>+10.4f}  {flags}"
        )
    print(f"")
    print(f"  Best-val-balacc winner : lambda_c = {best_val_lam}")
    print(f"    val_balacc = {float(best_row['best_val_balacc']):.4f}  "
          f"val_R_perp = {float(best_row['r_perp_val_at_best']):.4f}  "
          f"test_balacc = {float(best_row['test_balacc']):.4f}")
    print(f"")
    print(f"  Rule A winner           : lambda_c = {rule_a_lam}")
    if not rule_a_degenerate:
        ra_row = summary_df[summary_df["concurvity_lambda"] == rule_a_lam].iloc[0]
        ra_delta = float(ra_row["best_val_balacc"]) - lam0_balacc
        print(f"    val_balacc = {float(ra_row['best_val_balacc']):.4f}  "
              f"Δ = {ra_delta:+.4f} vs lambda=0  "
              f"(threshold: {threshold:.4f} = {lam0_balacc:.4f} - {RULE_A_TOLERANCE})")
    else:
        print(f"    ⚠ Rule A degenerate: even the smallest non-zero lambda "
              f"drops val_balacc by > {RULE_A_TOLERANCE}.")
        print(f"    This is a reportable finding.")
    print(f"")
    print(f"  NOTE: selection_pending=True in winner.json.")
    print(f"        User must choose the operative lambda_c for STEP 4 after review.")
    print(f"")
    print(f"  Total elapsed: {elapsed/60:.1f} min  ({elapsed/len(all_rows)/60:.1f} min/lambda)")
    print(f"  Outputs → {out_root}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
