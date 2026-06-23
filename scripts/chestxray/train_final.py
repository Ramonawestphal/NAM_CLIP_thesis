"""
Final 5-seed NAM training — chest X-ray three-way task.

Mirrors scripts/HAM10000/train_final.py exactly for the chest X-ray dataset
(Normal / Bacteria / Virus, 3 classes, 17 features, Config 10 architecture).
Structured for all four experimental conditions; plain_nam and concurvity_only
are implemented.  sparsity_only and sparsity_conc are stubs for future work.

Usage (from project root):
    python scripts/chestxray/train_final.py --condition plain_nam
    python scripts/chestxray/train_final.py --condition concurvity_only

Hard isolation rules
────────────────────
- Does NOT import _common.load_raw_data (HAM10000-specific).
- Does NOT touch any HAM10000 artefact, v1–v3 features, or architecture_selection output.
- Test set loaded ONLY in the per-seed evaluation block at the end of the seed loop.
- Val split (GroupShuffleSplit, random_state=42) is identical across all 5 seeds.
- StandardScaler fit on inner-train fold only, per seed.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix as sklearn_confusion_matrix,
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

# ── Fixed training settings ────────────────────────────────────────────────────
SEEDS            = [42, 43, 44, 45, 46]
LR               = 1e-3
BATCH_SIZE       = 256
MAX_EPOCHS       = 100
PATIENCE         = 15
SCHED_PATIENCE   = 5
SCHED_FACTOR     = 0.5
N_FEATURES       = 17
NUM_CLASSES      = 3
VAL_RANDOM_STATE = 42
ZERO_THRESHOLD   = 1e-4

FEATURES_PATH         = "data/features/biomedclip/chestxray_concept_scores_v4.npz"
SPLIT_PATH            = "data/splits/chestxray_outer_split.npz"
LABEL_MAP_PATH        = "results/chestxray/architecture_selection/label_mapping.json"
WINNER_JSON           = "results/chestxray/architecture_selection/winning_config.json"
CONCURVITY_WINNER_JSON = "results/chestxray/concurvity_sweep/winner.json"

# STEP 3 ground-truth reference for seed-42 reproduction check (concurvity_only)
# Loaded at runtime from the sweep metrics.json; these are fallback values.
_STEP3_LAM3_METRICS_PATH = "results/chestxray/concurvity_sweep/lambda_3/metrics.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
def load_label_mapping() -> dict:
    """Load SUBTYPE_TO_INT from label_mapping.json; fall back to hardcoded."""
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, encoding="utf-8") as f:
            mapping = json.load(f)
        return mapping
    return {"normal": 0, "bacteria": 1, "virus": 2}


def load_data(subtype_to_int: dict) -> dict:
    """Chest-X-ray-specific data loader.  Does NOT touch test_idx.

    Returns a dict with all arrays needed by pre-run sanity checks and the
    seed loop.  test_idx is returned but MUST NOT be used before the
    per-seed evaluation block.
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
        "labels_subtype":  labels_subtype,
        "train_pool_idx":  train_pool_idx,
        "test_idx":        test_idx,
        "patient_ids":     patient_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_test(
    model:        NAMMulticlass,
    X_test_t:     torch.Tensor,
    y_test:       np.ndarray,   # int64, 0/1/2
    class_names:  list,         # ["normal", "bacteria", "virus"]
) -> dict:
    """Compute full metric suite on the held-out test set.

    Separate from _common.evaluate_on_test because that function expects
    string labels and does a 7-class roc_auc call.  Here y_test is int64
    with exactly 3 classes.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()

    labels_list = list(range(NUM_CLASSES))
    bal_acc     = float(balanced_accuracy_score(y_test, preds))
    macro_f1    = float(f1_score(y_test, preds, average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
    top1_acc    = float(accuracy_score(y_test, preds))
    macro_auc   = float(roc_auc_score(
        y_test, proba, multi_class="ovr", average="macro",    labels=labels_list
    ))
    weighted_auc = float(roc_auc_score(
        y_test, proba, multi_class="ovr", average="weighted", labels=labels_list
    ))

    per_cls_auc = {
        name: float(roc_auc_score((y_test == i).astype(int), proba[:, i]))
        for i, name in enumerate(class_names)
    }

    report_dict = classification_report(
        y_test, preds,
        labels=labels_list,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T.loc[class_names].copy()
    report_df["support"] = report_df["support"].astype(int)
    report_df["auc"] = [per_cls_auc[c] for c in report_df.index]

    cm = sklearn_confusion_matrix(y_test, preds, labels=labels_list)

    return {
        "balanced_accuracy": bal_acc,
        "macro_f1":          macro_f1,
        "weighted_f1":       weighted_f1,
        "top1_accuracy":     top1_acc,
        "macro_auc_ovr":     macro_auc,
        "weighted_auc_ovr":  weighted_auc,
        "report_df":         report_df,
        "confusion_matrix":  cm,
        "proba":             proba,
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_sanity_checks(data: dict, subtype_to_int: dict, val_split: dict) -> None:
    """Print and assert pre-run sanity checks.  Raises on any failure."""
    scores        = data["scores"]
    train_pool_idx = data["train_pool_idx"]
    test_idx      = data["test_idx"]
    patient_ids   = data["patient_ids"]
    labels_all    = data["labels_all"]

    print("\n" + "=" * 65)
    print("PRE-RUN SANITY CHECKS")
    print("=" * 65)

    # 1. Feature file shape
    assert scores.shape == (5856, N_FEATURES), \
        f"CHECK 1 FAILED: scores.shape={scores.shape}, expected (5856, {N_FEATURES})"
    assert scores.dtype == np.float32, \
        f"CHECK 1 FAILED: dtype={scores.dtype}, expected float32"
    print(f"  [1] Feature shape: {scores.shape} dtype={scores.dtype}  ✓")

    # 2. Split file integrity
    n_train = len(train_pool_idx)
    n_test  = len(test_idx)
    overlap = np.intersect1d(train_pool_idx, test_idx)
    assert overlap.size == 0, f"CHECK 2 FAILED: {overlap.size} overlapping indices"
    assert n_train == 4658, f"CHECK 2 FAILED: train_pool len={n_train}, expected 4658"
    assert n_test  == 1198, f"CHECK 2 FAILED: test len={n_test}, expected 1198"
    # Patient-level non-overlap
    train_patients = set(patient_ids[train_pool_idx].tolist())
    test_patients  = set(patient_ids[test_idx].tolist())
    pat_overlap    = train_patients & test_patients
    assert len(pat_overlap) == 0, \
        f"CHECK 2 FAILED: {len(pat_overlap)} patients shared between train/test"
    print(f"  [2] Split integrity: train_pool={n_train}, test={n_test}, "
          f"index_overlap=0, patient_overlap=0  ✓")

    # 3. Label mapping consistency
    expected = {"normal": 0, "bacteria": 1, "virus": 2}
    assert subtype_to_int == expected, \
        f"CHECK 3 FAILED: label mapping {subtype_to_int} != {expected}"
    print(f"  [3] Label mapping: {subtype_to_int}  ✓")

    # 4. Three-way class counts on train pool and val split
    y_train_pool = labels_all[train_pool_idx]
    int_to_name  = {v: k for k, v in subtype_to_int.items()}
    counts_pool  = np.bincount(y_train_pool, minlength=NUM_CLASSES)
    props_pool   = counts_pool / counts_pool.sum()
    print(f"  [4] Train-pool class counts (N={n_train}):")
    for i in range(NUM_CLASSES):
        print(f"        {int_to_name[i]}: {counts_pool[i]} ({props_pool[i]*100:.1f}%)")

    y_val = y_train_pool[val_split["val_rel"]]
    counts_val = np.bincount(y_val, minlength=NUM_CLASSES)
    props_val  = counts_val / counts_val.sum()
    print(f"  [4] Val split class counts (N={len(y_val)}):")
    for i in range(NUM_CLASSES):
        diff_pp = abs(props_val[i] - props_pool[i]) * 100
        flag = "  ⚠ >2pp" if diff_pp > 2 else ""
        print(f"        {int_to_name[i]}: {counts_val[i]} ({props_val[i]*100:.1f}%)  "
              f"[Δ{diff_pp:.1f}pp vs pool{flag}]")

    # 5. Winning config sanity
    with open(WINNER_JSON, encoding="utf-8") as f:
        wc = json.load(f)
    assert wc["hidden_dims"] == [64, 32], \
        f"CHECK 5 FAILED: hidden_dims={wc['hidden_dims']}, expected [64, 32]"
    assert abs(wc["dropout"] - 0.1) < 1e-9, \
        f"CHECK 5 FAILED: dropout={wc['dropout']}, expected 0.1"
    assert abs(wc["weight_decay"] - 1e-4) < 1e-9, \
        f"CHECK 5 FAILED: weight_decay={wc['weight_decay']}, expected 1e-4"
    print(f"  [5] Winning config (config {wc['config_id']}): "
          f"hidden={wc['hidden_dims']}, dropout={wc['dropout']}, "
          f"wd={wc['weight_decay']}  ✓")

    # 6. Deterministic val split — print first 10 indices for seed 42
    train_rel = val_split["train_rel"]
    val_rel   = val_split["val_rel"]
    print(f"  [6] Val-split indices (GroupShuffleSplit, random_state=42):")
    print(f"        train_final[:10] = {train_rel[:10].tolist()}")
    print(f"        val[:10]         = {val_rel[:10].tolist()}")
    print(f"        train_final size = {len(train_rel)}, val size = {len(val_rel)}")

    # 7. Class weight values on seed 42 train_final fold
    y_train_final = y_train_pool[train_rel]
    counts_tf = np.bincount(y_train_final, minlength=NUM_CLASSES)
    n_tf = len(y_train_final)
    weights_tf = n_tf / (NUM_CLASSES * counts_tf)
    print(f"  [7] Inverse-frequency weights (seed-42 train_final, n={n_tf}):")
    for i in range(NUM_CLASSES):
        print(f"        {int_to_name[i]}: n={counts_tf[i]}, weight={weights_tf[i]:.4f}")

    print("=" * 65)
    print("All sanity checks passed.  Ready to train.\n")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True,
        choices=["plain_nam", "concurvity_only", "sparsity_only", "sparsity_conc"],
        help="Experimental condition.  plain_nam and concurvity_only are implemented.")
    parser.add_argument("--concurvity_lambda", type=float, default=None)
    parser.add_argument("--sparsity_lambda",   type=float, default=None)
    parser.add_argument("--out_dir",     type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience",   type=int, default=None)
    parser.add_argument("--seed",       type=int, default=None,
                        help="Run a single seed only (for testing).")
    parser.add_argument("--sanity_only", action="store_true",
                        help="Run sanity checks then exit without training.")
    parser.add_argument("--winner_json", type=str, default=None)
    parser.add_argument("--concurvity_winner_json", type=str, default=None,
                        help="Override path to concurvity_sweep/winner.json "
                             "(used for concurvity_only and sparsity_conc).")
    args = parser.parse_args()

    cond = args.condition
    if cond in ("sparsity_only", "sparsity_conc"):
        raise NotImplementedError(
            f"Condition '{cond}' is not yet implemented for chest X-ray. "
            "Run the sparsity sweep first."
        )

    # ── Resolve hyperparameters (mirrors v7/train_final.py lines 124–151) ──────
    # Concurvity lambda: explicit CLI > concurvity_sweep/winner.json > 0.0
    if args.concurvity_lambda is not None:
        lam_c = args.concurvity_lambda
    elif cond in ("concurvity_only",):
        conc_winner_path = args.concurvity_winner_json or CONCURVITY_WINNER_JSON
        if not os.path.exists(conc_winner_path):
            raise FileNotFoundError(
                f"Concurvity sweep winner.json not found at {conc_winner_path}. "
                "Run run_concurvity_sweep.py (STEP 3) first."
            )
        with open(conc_winner_path, encoding="utf-8") as _f:
            _cw = json.load(_f)
        if _cw.get("selection_pending", True):
            raise RuntimeError(
                f"selection_pending=True in {conc_winner_path}. "
                "Resolve the operative lambda_c before running STEP 4."
            )
        if "operative_lambda_c" not in _cw:
            raise KeyError(
                f"operative_lambda_c missing from {conc_winner_path}. "
                "Edit winner.json to add the operative lambda before running STEP 4."
            )
        lam_c = float(_cw["operative_lambda_c"])
        print(f"  [lambda_c] loaded {lam_c} from {conc_winner_path} "
              f"(rule: {_cw.get('selection_rule', 'n/a')})")
    else:
        lam_c = 0.0   # plain_nam

    lam_s     = args.sparsity_lambda if args.sparsity_lambda is not None else 0.0
    warmup_ep = 0   # HAM10000 v7 diagnostic confirmed warmup=0 is optimal
    max_ep   = args.max_epochs if args.max_epochs is not None else MAX_EPOCHS
    pat      = args.patience   if args.patience   is not None else PATIENCE
    seeds    = [args.seed] if args.seed is not None else SEEDS

    # ── Read winning config ────────────────────────────────────────────────────
    winner_json_path = args.winner_json or WINNER_JSON
    if not os.path.exists(winner_json_path):
        raise FileNotFoundError(
            f"winning_config.json not found at {winner_json_path}. "
            "Run select_architecture.py first."
        )
    with open(winner_json_path, encoding="utf-8") as f:
        winner = json.load(f)
    hidden_dims  = tuple(winner["hidden_dims"])
    dropout      = float(winner["dropout"])
    weight_decay = float(winner["weight_decay"])

    # ── Output directory ───────────────────────────────────────────────────────
    default_out = f"results/chestxray/{cond}"
    out_dir     = args.out_dir if args.out_dir is not None else default_out
    os.makedirs(out_dir, exist_ok=True)

    # ── Load label mapping and data ────────────────────────────────────────────
    subtype_to_int = load_label_mapping()
    int_to_subtype = {v: k for k, v in subtype_to_int.items()}
    class_names    = [int_to_subtype[i] for i in range(NUM_CLASSES)]  # ["normal","bacteria","virus"]

    data = load_data(subtype_to_int)
    scores        = data["scores"]
    concept_names = data["concept_names"]
    labels_all    = data["labels_all"]
    train_pool_idx = data["train_pool_idx"]
    test_idx      = data["test_idx"]
    patient_ids   = data["patient_ids"]

    X_train_pool = scores[train_pool_idx]
    y_train_pool = labels_all[train_pool_idx]
    groups_pool  = patient_ids[train_pool_idx]

    # ── Fixed val split (identical across seeds) ───────────────────────────────
    # Pass int-as-str labels and ["0","1","2"] as class_names, mirroring
    # final_evaluation.py.  GroupShuffleSplit ignores y; class_names only
    # affect the returned encoding, which we discard in favour of y_train_pool.
    val_split = make_fixed_val_split(
        X_train_pool,
        y_train_pool.astype(str),
        groups_pool,
        ["0", "1", "2"],
        val_random_state=VAL_RANDOM_STATE,
    )
    train_rel = val_split["train_rel"]
    val_rel   = val_split["val_rel"]

    # ── Pre-run sanity checks ──────────────────────────────────────────────────
    run_sanity_checks(data, subtype_to_int, val_split)

    if args.sanity_only:
        print("--sanity_only flag set.  Exiting before training.")
        return

    # ── Print banner ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Chest X-ray NAM — Final training [{cond}]")
    print(f"  Config (from {winner_json_path}):")
    print(f"    config_id={winner['config_id']}, hidden={list(hidden_dims)}, "
          f"dropout={dropout}, wd={weight_decay:.0e}")
    print(f"  lambda_c={lam_c}, lambda_s={lam_s}, warmup_epochs={warmup_ep}")
    print(f"  seeds={seeds}, max_epochs={max_ep}, patience={pat}")
    print(f"  val_random_state={VAL_RANDOM_STATE} (fixed across all seeds)")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {out_dir}/")
    print(f"{'='*65}\n")

    # ── Slices used throughout ─────────────────────────────────────────────────
    X_train_final_raw = X_train_pool[train_rel]   # inner train (pre-scaling)
    X_val_raw         = X_train_pool[val_rel]
    y_train_final     = y_train_pool[train_rel]   # int64
    y_val             = y_train_pool[val_rel]      # int64

    # X_test_raw loaded here to avoid repeated disk I/O, but ONLY passed to the
    # evaluation block inside the seed loop — never used for fitting/val.
    X_test_raw = scores[test_idx]
    y_test     = labels_all[test_idx]

    all_results = []
    t0 = time.time()

    for seed in seeds:
        print(f"── Seed {seed} " + "─" * 50)
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        set_all_seeds(seed)

        # ── Per-seed standardisation (fit on inner train only) ─────────────────
        scaler    = StandardScaler()
        X_tr_sc   = scaler.fit_transform(X_train_final_raw).astype(np.float32)
        X_val_sc  = scaler.transform(X_val_raw).astype(np.float32)
        X_test_sc = scaler.transform(X_test_raw).astype(np.float32)

        with open(os.path.join(seed_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        # ── Class weights (inverse-frequency, matching select_architecture.py) ──
        counts   = np.bincount(y_train_final, minlength=NUM_CLASSES)
        n_train  = len(y_train_final)
        weights  = n_train / (NUM_CLASSES * counts.astype(np.float64))
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

        # ── Model and optimiser ────────────────────────────────────────────────
        model = NAMMulticlass(
            n_features=N_FEATURES,
            num_classes=NUM_CLASSES,
            hidden_dims=hidden_dims,
            dropout=dropout,
            concept_names=concept_names,
        ).to(DEVICE)
        optimizer, scheduler = make_optimizer_scheduler(
            model, LR, weight_decay, SCHED_PATIENCE, SCHED_FACTOR
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

        # ── Training ───────────────────────────────────────────────────────────
        result = train_one_run(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_dataset=train_ds,
            X_val_t=X_val_t,
            y_val_t=y_val_t,
            y_val_enc=y_val,         # int64 array for balanced_accuracy_score
            max_epochs=max_ep,
            patience=pat,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            concurvity_lambda=lam_c,
            warmup_epochs=warmup_ep,
            sparsity_lambda=lam_s,
            proximal_sparsity=True,
            save_path=ckpt_path,
            verbose_every=10,
        )

        result["log_df"].to_csv(os.path.join(seed_dir, "training_log.csv"), index=False)

        # ── R_perp at best val epoch (from training log) ───────────────────────
        log_df   = result["log_df"]
        best_idx = log_df["val_balanced_acc"].idxmax()
        r_perp_val_best = float(log_df.loc[best_idx, "r_perp_val"])
        r_perp_tr_best  = float(log_df.loc[best_idx, "r_perp_train"])

        # ── Feature group norms ────────────────────────────────────────────────
        norms = feature_group_norms(model)
        norms_rows = [
            {"concept_name": nm, "norm": nv}
            for nm, nv in norms.items()
        ]
        pd.DataFrame(norms_rows).to_csv(
            os.path.join(seed_dir, "feature_group_norms.csv"), index=False
        )
        n_active = sum(1 for r in norms_rows if r["norm"] >= ZERO_THRESHOLD)

        # ── Test evaluation (once per seed, at the very end) ──────────────────
        metrics = evaluate_on_test(model, X_test_t, y_test, class_names)

        best_ep  = result["best_epoch"]
        best_val = result["best_val_balacc"]
        stop_str = "early_stop" if result["early_stopped"] else "max_epochs"
        print(
            f"  Test: bal_acc={metrics['balanced_accuracy']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}  "
            f"macro_auc={metrics['macro_auc_ovr']:.4f}  "
            f"n_active={n_active}/{N_FEATURES}"
        )
        print(
            f"  best_epoch={best_ep}  val_balacc={best_val:.4f}  "
            f"val_R_perp={r_perp_val_best:.4f}  "
            f"total_epochs={result['total_epochs']}  stop={stop_str}"
        )

        # ── STEP 3 reproduction check (concurvity_only seed 42 only) ──────────
        if cond == "concurvity_only" and seed == 42:
            _ref: dict = {}
            if os.path.exists(_STEP3_LAM3_METRICS_PATH):
                with open(_STEP3_LAM3_METRICS_PATH, encoding="utf-8") as _mf:
                    _s3 = json.load(_mf)
                _ref = {
                    "val_balacc":  float(_s3["best_val_balacc"]),
                    "val_R_perp":  float(_s3["r_perp_val_at_best"]),
                    "test_balacc": float(_s3["test_balacc"]),
                    "best_epoch":  int(_s3["best_epoch"]),
                }
            else:
                # Hard fallback from task spec
                _ref = {"val_balacc": 0.7331, "val_R_perp": 0.1074,
                        "test_balacc": 0.7281, "best_epoch": 26}
                print(f"  [repro] STEP 3 metrics.json not found; using hard-coded reference.")
            _tol = 0.0001
            _match = (
                abs(best_val - _ref["val_balacc"])               <= _tol and
                abs(r_perp_val_best - _ref["val_R_perp"])        <= _tol * 20 and
                abs(metrics["balanced_accuracy"] - _ref["test_balacc"]) <= _tol and
                best_ep == _ref["best_epoch"]
            )
            _status = "✓" if _match else "✗"
            print(f"\n  ── STEP 3 reproduction check (seed 42, λ_c=3.0) ──────────")
            print(f"  STEP 3 sweep   : val_balacc={_ref['val_balacc']:.4f}  "
                  f"val_R_perp={_ref['val_R_perp']:.4f}  "
                  f"test_balacc={_ref['test_balacc']:.4f}  "
                  f"best_epoch={_ref['best_epoch']}")
            print(f"  STEP 4 seed 42 : val_balacc={best_val:.4f}  "
                  f"val_R_perp={r_perp_val_best:.4f}  "
                  f"test_balacc={metrics['balanced_accuracy']:.4f}  "
                  f"best_epoch={best_ep}")
            print(f"  Match          : {_status}")
            if not _match:
                print(f"  ⚠ MISMATCH — stopping before remaining seeds. Investigate.")
                sys.exit(1)
            print()

        # ── Suspicious early-stopping check ───────────────────────────────────
        if best_ep <= 3:
            print(f"  ⚠ best_epoch={best_ep} ≤ 3 — suspiciously early checkpoint. "
                  f"Possible warm-up-era artefact.")

        all_results.append({
            "seed":               seed,
            "best_val_balacc":    best_val,
            "best_epoch":         best_ep,
            "n_active":           n_active,
            "r_perp_val_at_best": r_perp_val_best,
            "r_perp_tr_at_best":  r_perp_tr_best,
            "early_stopped":      result["early_stopped"],
            "total_epochs":       result["total_epochs"],
            "log_df":             result["log_df"],
            **{k: v for k, v in metrics.items()
               if k not in ("report_df", "confusion_matrix", "proba")},
            "report_df":        metrics["report_df"],
            "confusion_matrix": metrics["confusion_matrix"],
        })

    # ── Aggregated metrics ─────────────────────────────────────────────────────
    agg_keys = [
        "balanced_accuracy", "macro_f1", "weighted_f1",
        "top1_accuracy", "macro_auc_ovr", "weighted_auc_ovr",
    ]
    agg_rows = [
        {"seed": r["seed"], **{k: r[k] for k in agg_keys}}
        for r in all_results
    ]
    agg_df   = pd.DataFrame(agg_rows)
    mean_row = {**agg_df[agg_keys].mean().to_dict(), "seed": "mean"}
    std_row  = {**agg_df[agg_keys].std().to_dict(),  "seed": "std"}   # ddof=1
    agg_out  = pd.concat(
        [agg_df, pd.DataFrame([mean_row, std_row])], ignore_index=True
    )
    agg_out.to_csv(os.path.join(out_dir, "aggregated_metrics.csv"), index=False)

    means = {k: float(mean_row[k]) for k in agg_keys}
    stds  = {k: float(std_row[k])  for k in agg_keys}

    # ── Per-class metrics (seed-mean) ──────────────────────────────────────────
    report_mean = (
        pd.concat([r["report_df"] for r in all_results])
        .groupby(level=0).mean()
        .loc[[c for c in all_results[0]["report_df"].index]]
    )
    report_mean["support"] = report_mean["support"].round(0).astype(int)
    report_mean.to_csv(os.path.join(out_dir, "per_class_metrics.csv"))

    # ── Confusion matrix (seed-mean, row-normalised) ───────────────────────────
    cms     = np.stack([r["confusion_matrix"] for r in all_results], axis=0)
    cm_mean = cms.mean(axis=0)
    cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)
    pd.DataFrame(
        cm_norm.round(4), index=class_names, columns=class_names
    ).to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    # ── Training curves ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in all_results:
        log = r["log_df"]
        ax.plot(log["epoch"], log["val_balanced_acc"], alpha=0.8,
                label=f"seed {r['seed']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val balanced accuracy")
    ax.set_title(f"Chest X-ray NAM [{cond}] — Training curves")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

    # ── run_config.json ────────────────────────────────────────────────────────
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT), text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_hash = None

    # R_perp summary across seeds (for concurvity_only; also recorded for plain_nam)
    r_perp_vals = [r["r_perp_val_at_best"] for r in all_results]
    r_perp_mean = float(np.mean(r_perp_vals))
    r_perp_std  = float(np.std(r_perp_vals, ddof=1)) if len(r_perp_vals) > 1 else 0.0

    meta = {
        "condition":         cond,
        "config_name":       f"config_{winner['config_id']}",
        "config_id":         winner["config_id"],
        "hidden_dims":       list(hidden_dims),
        "dropout":           dropout,
        "weight_decay":      weight_decay,
        "lr":                LR,
        "batch_size":        BATCH_SIZE,
        "max_epochs":        max_ep,
        "patience":          pat,
        "concurvity_lambda": lam_c,
        "sparsity_lambda":   lam_s,
        "warmup_epochs":     warmup_ep,
        "seeds":             seeds,
        "val_random_state":  VAL_RANDOM_STATE,
        "val_split_note":    "GroupShuffleSplit(random_state=42), identical across seeds and to STEP 2 plain_nam",
        "n_features":        N_FEATURES,
        "num_classes":       NUM_CLASSES,
        "feature_file":      FEATURES_PATH,
        "split_file":        SPLIT_PATH,
        "label_mapping":     subtype_to_int,
        "r_perp_val_mean":   r_perp_mean,
        "r_perp_val_std":    r_perp_std,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "git_commit":        git_hash,
        "winner_json":       winner_json_path,
    }
    if cond == "concurvity_only":
        meta["concurvity_winner_json"] = args.concurvity_winner_json or CONCURVITY_WINNER_JSON
        meta["operative_lambda_c_source"] = "results/chestxray/concurvity_sweep/winner.json"
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ── Summary printout ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"Chest X-ray NAM [{cond}] — Final results ({len(seeds)} seeds)")
    print(f"  Balanced accuracy  : {means['balanced_accuracy']:.4f} ± {stds['balanced_accuracy']:.4f}")
    print(f"  Macro F1           : {means['macro_f1']:.4f} ± {stds['macro_f1']:.4f}")
    print(f"  Macro AUC (OvR)    : {means['macro_auc_ovr']:.4f} ± {stds['macro_auc_ovr']:.4f}")
    print(f"  Weighted AUC (OvR) : {means['weighted_auc_ovr']:.4f} ± {stds['weighted_auc_ovr']:.4f}")
    print(f"  Per-class test accuracy (seed-mean):")
    for row_name in class_names:
        if row_name in report_mean.index:
            print(f"    {row_name:10s}: recall={report_mean.loc[row_name, 'recall']:.4f}  "
                  f"AUC={report_mean.loc[row_name, 'auc']:.4f}")
    print(f"  Val R_perp at best : {r_perp_mean:.4f} ± {r_perp_std:.4f} (seed-mean)")
    print(f"  Total elapsed      : {elapsed/60:.1f} min  "
          f"({elapsed/len(seeds)/60:.1f} min/seed)")
    print(f"  Reference (plain_nam): bal_acc ≈ 0.7376 ± 0.0091")
    delta = means['balanced_accuracy'] - 0.7376
    if abs(delta) > 0.05:
        print(f"  ⚠ Δ={delta:+.4f} vs plain_nam reference — investigate if unexpected.")
    if stds['balanced_accuracy'] > 0.020:
        print(f"  ⚠ std={stds['balanced_accuracy']:.4f} > 0.020 — higher than expected "
              f"(HAM10000 concurvity_only std ~0.010).")
    best_epochs = [r["best_epoch"] for r in all_results]
    suspicious  = [r["seed"] for r in all_results if r["best_epoch"] <= 3]
    print(f"  Best epochs        : {best_epochs}  "
          + (f"⚠ seed(s) {suspicious} suspicious (epoch≤3)" if suspicious else "✓ none ≤ 3"))
    print(f"  Outputs → {out_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
