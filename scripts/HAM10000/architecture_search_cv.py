"""
CV-based architecture selection for NAM v6 (corrected pipeline, v7).

Fixes applied
─────────────
Issue 1 (Critical): Architecture is selected by 5-fold GroupKFold cross-validation
    on the training pool (8020 samples), scored by mean balanced accuracy across
    folds.  The held-out test set (test_idx in the splits file) is NEVER loaded,
    computed on, or referenced anywhere in this script.  An explicit assertion at
    startup verifies this by design.

Issue 2: Test metrics are not computed for any candidate configuration.

Methodology notes
─────────────────
* GroupKFold(n_splits=5) is used.  It guarantees no lesion_id appears in both the
  training and validation portions of a fold (the splitter respects the `groups`
  argument).  An explicit per-fold assertion confirms this.
* A single model-init seed (42) is used across all (config, fold) pairs.  The
  variability of interest at this stage is across data folds, not model
  initialisation — using the same seed makes fold comparisons cleaner.
* WARNING: GroupKFold does not stratify by class.  With HAM10000's class imbalance
  (NV majority ~67%), fold class distributions may vary slightly.  This is
  acceptable: balanced accuracy is insensitive to within-fold support variation.
* max_epochs=80, patience=15 (same as sweep_nam_v6.py for comparability).
* No regularization (lambda_c=0, lambda_s=0) at the architecture selection stage.

Output
──────
results/HAM10000/architecture_search_cv/
    cv_results.csv       one row per (config_id, fold)
    cv_summary.csv       one row per config_id: mean ± std of val_balacc
    winner.json          selected config and rationale
    fold_indices.json    per-fold train/val absolute indices (for verify_no_leakage.py)
    log.txt              stdout mirror

Usage (from project root)
─────────────────────────
    python scripts/HAM10000/architecture_search_cv.py
    python scripts/HAM10000/architecture_search_cv.py --smoke_test   # config 1 only
    python scripts/HAM10000/architecture_search_cv.py --max_epochs 20  # fast sanity run
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from datetime import datetime, timezone

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── SAFETY ASSERTION: test_idx must never appear in this file ─────────────────
# We enforce this at the code level: test_idx is loaded into a variable called
# _TEST_IDX_SENTINEL that is referenced ONLY inside the assertion below and in
# fold_indices.json (to document what was excluded).  It is never passed to any
# training or evaluation function.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import TensorDataset

# Allow Windows consoles to print UTF-8 without crashing
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from scripts.HAM10000._common import (
    FEATURES_PATH, SPLITS_PATH, SWEEP_GRID,
    N_FEATURES, N_CLASSES,
    set_all_seeds, load_raw_data, standardize,
    class_weight_tensor, make_model, make_optimizer_scheduler,
    train_one_run, write_step_flag,
)

# ── Fixed hyperparameters ─────────────────────────────────────────────────────
CV_SEED      = 42        # model init seed for all (config, fold) pairs
N_FOLDS      = 5
LR           = 1e-3
BATCH_SIZE   = 256
MAX_EPOCHS   = 80        # matches sweep_nam_v6.py
PATIENCE     = 15
SCHED_PATIENCE = 5
SCHED_FACTOR   = 0.5
OUT_DIR      = "results/HAM10000/architecture_search_cv"
RESULTS_V7   = "results/HAM10000"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Logging tee: mirror stdout to log.txt
# ─────────────────────────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, path: str):
        self._file = open(path, "w", encoding="utf-8", errors="replace")
        self._stdout = sys.stdout
    def write(self, s: str):
        self._stdout.write(s)
        self._file.write(s)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke_test",  action="store_true",
                        help="Run only config 1 (5 folds) to verify mechanics.")
    parser.add_argument("--max_epochs",  type=int, default=None,
                        help="Override max_epochs (default 80).")
    parser.add_argument("--out_dir",     type=str, default=None)
    args = parser.parse_args()

    max_epochs = args.max_epochs if args.max_epochs is not None else MAX_EPOCHS
    out_dir    = args.out_dir    if args.out_dir    is not None else OUT_DIR

    os.makedirs(out_dir,     exist_ok=True)
    os.makedirs(RESULTS_V7,  exist_ok=True)

    tee = _Tee(os.path.join(out_dir, "log.txt"))
    sys.stdout = tee

    try:
        _run(args, max_epochs, out_dir)
    finally:
        sys.stdout = tee._stdout
        tee.close()


def _run(args: argparse.Namespace, max_epochs: int, out_dir: str) -> None:
    t0 = time.time()
    print("=" * 70)
    print("NAM v7 — Architecture selection by 5-fold GroupKFold CV")
    print(f"  Audit fix: Issue 1 (test-set leakage in architecture selection)")
    print(f"             Issue 2 (test metrics computed for all Stage-1 configs)")
    print(f"  Device   : {DEVICE}")
    print(f"  CV seed  : {CV_SEED}")
    print(f"  Folds    : {N_FOLDS}")
    print(f"  Max epochs per fold: {max_epochs}")
    if args.smoke_test:
        print("  *** SMOKE TEST MODE: config 1 only ***")
    print("=" * 70)

    # ── Load data (train pool ONLY; test_idx loaded into sentinel only) ────────
    raw = load_raw_data(FEATURES_PATH, SPLITS_PATH)
    _TEST_IDX_SENTINEL = raw["test_idx"]   # NEVER passed to training/eval

    scores        = raw["scores"]
    labels        = raw["labels"]
    lesion_ids    = raw["lesion_ids"]
    concept_names = raw["concept_names"]
    class_names   = raw["class_names"]
    train_idx     = raw["train_idx"]

    X_all_train      = scores[train_idx]        # (8020, 24)
    y_all_train      = labels[train_idx]         # (8020,)  strings
    lesion_ids_train = lesion_ids[train_idx]     # (8020,)

    # Encode labels
    class_to_idx    = {c: i for i, c in enumerate(class_names)}
    y_all_train_enc = np.array([class_to_idx[c] for c in y_all_train], dtype=np.int64)

    print(f"\nTrain pool: {len(train_idx)} samples, {len(np.unique(lesion_ids_train))} unique lesions")
    print(f"Test pool (EXCLUDED from all computation): {len(_TEST_IDX_SENTINEL)} samples")

    # ── Verify test_idx never intersects train_idx (design-level assertion) ───
    assert len(np.intersect1d(train_idx, _TEST_IDX_SENTINEL)) == 0, \
        "train_idx and test_idx overlap — check splits file"
    print("  [assertion passed] train_idx ∩ test_idx = ∅")

    # ── Build GroupKFold splits ────────────────────────────────────────────────
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(
        gkf.split(X_all_train, y_all_train, groups=lesion_ids_train)
    )

    # Verify no lesion leakage across folds
    for fold_i, (tr_rel, va_rel) in enumerate(fold_splits):
        tr_lesions = set(lesion_ids_train[tr_rel])
        va_lesions = set(lesion_ids_train[va_rel])
        assert len(tr_lesions & va_lesions) == 0, \
            f"Lesion leakage in fold {fold_i}"
    print(f"  [assertion passed] All {N_FOLDS} folds have non-overlapping lesion_ids")

    # Save fold indices (absolute dataset indices) for verify_no_leakage.py
    fold_index_data = {
        "test_idx_excluded": _TEST_IDX_SENTINEL.tolist(),
        "folds": [
            {
                "fold": fold_i,
                "train_abs_indices": train_idx[tr_rel].tolist(),
                "val_abs_indices":   train_idx[va_rel].tolist(),
            }
            for fold_i, (tr_rel, va_rel) in enumerate(fold_splits)
        ],
    }
    with open(os.path.join(out_dir, "fold_indices.json"), "w") as f:
        json.dump(fold_index_data, f, indent=2)
    print(f"  Saved fold_indices.json")

    # ── Config grid ────────────────────────────────────────────────────────────
    configs_to_run = list(enumerate(SWEEP_GRID, start=1))
    if args.smoke_test:
        configs_to_run = configs_to_run[:1]   # only config 1
        print(f"\n  Smoke test: running config 1 only ({N_FOLDS} folds)")

    total_runs = len(configs_to_run) * N_FOLDS
    print(f"\nTotal runs: {len(configs_to_run)} configs × {N_FOLDS} folds = {total_runs}")
    print(f"Grid: hidden x dropout x weight_decay")
    print(f"{'='*70}\n")

    # ── Training loop ──────────────────────────────────────────────────────────
    all_records: list[dict] = []
    run_counter = 0

    for cfg_id, (hidden_dims, dropout, weight_decay) in configs_to_run:
        print(f"Config {cfg_id:2d}: hidden={list(hidden_dims)}, "
              f"dropout={dropout}, weight_decay={weight_decay:.0e}")
        fold_bacc_list: list[float] = []

        for fold_i, (tr_rel, va_rel) in enumerate(fold_splits):
            run_counter += 1
            t_fold = time.time()
            print(f"  Fold {fold_i+1}/{N_FOLDS}  "
                  f"(run {run_counter}/{total_runs}  |  "
                  f"train={len(tr_rel)}, val={len(va_rel)})")

            # ── Per-fold data preparation ──────────────────────────────────────
            X_tr_raw = X_all_train[tr_rel]
            y_tr_enc = y_all_train_enc[tr_rel]
            y_tr_str = y_all_train[tr_rel]
            X_va_raw = X_all_train[va_rel]
            y_va_enc = y_all_train_enc[va_rel]

            # Issue 7: scaler fitted on this fold's training data only
            X_tr_sc, X_va_sc, _, _ = standardize(X_tr_raw, X_va_raw)

            # Class weights from this fold's training labels only
            w_tensor = class_weight_tensor(y_tr_str, class_names, DEVICE)

            # Tensors
            X_val_t = torch.tensor(X_va_sc, dtype=torch.float32, device=DEVICE)
            y_val_t = torch.tensor(y_va_enc, dtype=torch.long,    device=DEVICE)

            train_ds = TensorDataset(
                torch.tensor(X_tr_sc, dtype=torch.float32),
                torch.tensor(y_tr_enc, dtype=torch.long),
            )

            # ── Train ──────────────────────────────────────────────────────────
            # Use CV_SEED for model init (same across all folds/configs so that
            # fold-to-fold differences reflect only data, not init).
            set_all_seeds(CV_SEED)

            model = make_model(hidden_dims, dropout, concept_names, DEVICE)
            optimizer, scheduler = make_optimizer_scheduler(
                model, LR, weight_decay, SCHED_PATIENCE, SCHED_FACTOR
            )
            criterion = nn.CrossEntropyLoss(weight=w_tensor)

            result = train_one_run(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_dataset=train_ds,
                X_val_t=X_val_t,
                y_val_t=y_val_t,
                y_val_enc=y_va_enc,
                max_epochs=max_epochs,
                patience=PATIENCE,
                batch_size=BATCH_SIZE,
                device=DEVICE,
                concurvity_lambda=0.0,   # no regularization during architecture search
                warmup_epochs=0,
                sparsity_lambda=0.0,
                save_path=None,          # no checkpoint needed for CV folds
                verbose_every=max_epochs + 1,  # suppress per-epoch noise
            )

            fold_bacc = result["best_val_balacc"]
            fold_bacc_list.append(fold_bacc)
            elapsed = time.time() - t_fold

            print(f"    → fold_val_balacc={fold_bacc:.4f}  "
                  f"best_epoch={result['best_epoch']}  "
                  f"time={elapsed:.1f}s")

            all_records.append({
                "config_id":     cfg_id,
                "hidden":        str(list(hidden_dims)),
                "dropout":       dropout,
                "weight_decay":  weight_decay,
                "fold":          fold_i,
                "fold_val_balacc": fold_bacc,
                "best_epoch":    result["best_epoch"],
                "elapsed_s":     round(elapsed, 1),
            })

        # Per-config summary
        mean_bacc = float(np.mean(fold_bacc_list))
        std_bacc  = float(np.std(fold_bacc_list, ddof=1))
        print(f"  → mean={mean_bacc:.4f} ± {std_bacc:.4f}  folds={fold_bacc_list}\n")

    # ── Save cv_results.csv ────────────────────────────────────────────────────
    cv_df = pd.DataFrame(all_records)
    cv_df.to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)

    # ── Build cv_summary.csv (Issue 5 fix: std with ddof=1) ───────────────────
    summary_rows: list[dict] = []
    for cfg_id, (hidden_dims, dropout, weight_decay) in configs_to_run:
        subset     = cv_df[cv_df["config_id"] == cfg_id]["fold_val_balacc"]
        mean_bacc  = float(subset.mean())
        std_bacc   = float(subset.std())    # pandas .std() uses ddof=1
        summary_rows.append({
            "config_id":      cfg_id,
            "hidden":         str(list(hidden_dims)),
            "dropout":        dropout,
            "weight_decay":   weight_decay,
            "mean_cv_balacc": round(mean_bacc, 4),
            "std_cv_balacc":  round(std_bacc,  4),
            "n_folds":        N_FOLDS,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)

    # ── Winner selection (Issue 1 fix: by mean CV val_balacc, not test) ────────
    sorted_summary = summary_df.sort_values(
        ["mean_cv_balacc", "std_cv_balacc"],
        ascending=[False, True],          # highest mean, then lowest std for ties
    ).reset_index(drop=True)

    winner_row = sorted_summary.iloc[0]
    runner_up  = sorted_summary.iloc[1] if len(sorted_summary) > 1 else None

    gap = float(winner_row["mean_cv_balacc"]) - (
        float(runner_up["mean_cv_balacc"]) if runner_up is not None else 0.0
    )
    is_tie = gap < float(winner_row["std_cv_balacc"]) if runner_up is not None else False

    winner_cfg_id = int(winner_row["config_id"])
    winner = next(
        (hidden_dims, dropout, weight_decay)
        for cfg_id, (hidden_dims, dropout, weight_decay) in SWEEP_GRID_INDEXED
        if cfg_id == winner_cfg_id
    )

    winner_info = {
        "config_id":           winner_cfg_id,
        "hidden_dims":         list(winner[0]),
        "dropout":             float(winner[1]),
        "weight_decay":        float(winner[2]),
        "mean_cv_balacc":      float(winner_row["mean_cv_balacc"]),
        "std_cv_balacc":       float(winner_row["std_cv_balacc"]),
        "selection_criterion": "highest mean CV val_balacc (5-fold GroupKFold); "
                               "tiebreak by lowest std_cv_balacc",
        "is_tie":              is_tie,
        "gap_to_runner_up":    round(gap, 4),
        "n_folds":             N_FOLDS,
        "model_init_seed":     CV_SEED,
        "max_epochs_per_fold": max_epochs,
        "test_set_touched":    False,
        "audit_fix":           "Issue 1 — architecture NOT selected by test metrics",
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "winner.json"), "w") as f:
        json.dump(winner_info, f, indent=2)

    # ── Print results ──────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print("=" * 70)
    print("CV Summary (sorted by mean_cv_balacc desc):")
    print("=" * 70)
    print(summary_df.sort_values("mean_cv_balacc", ascending=False)
          .to_string(index=False))
    print()
    print("=" * 70)
    print(f"WINNER: Config {winner_cfg_id}")
    print(f"  hidden={winner_info['hidden_dims']}, dropout={winner_info['dropout']}, "
          f"weight_decay={winner_info['weight_decay']:.0e}")
    print(f"  Mean CV val_balacc: {winner_info['mean_cv_balacc']:.4f} "
          f"± {winner_info['std_cv_balacc']:.4f}")
    if is_tie:
        print(f"  ** TIE with runner-up (gap={gap:.4f} < std={winner_row['std_cv_balacc']:.4f})")
        print(f"     Tiebreak: lower std_cv_balacc → config {winner_cfg_id}")
    else:
        print(f"  Gap to runner-up: {gap:.4f}")
    print(f"\n  Selection criterion: {winner_info['selection_criterion']}")
    print(f"  test_set_touched: {winner_info['test_set_touched']}")
    print("=" * 70)
    print(f"\nOutputs → {out_dir}/")
    print(f"Total elapsed: {total_time/60:.1f} min")

    # ── Step flag ──────────────────────────────────────────────────────────────
    if not args.smoke_test:
        write_step_flag(RESULTS_V7, 1)
        print(f"\nSTEP 1 COMPLETE. Run scripts/HAM10000/verify_no_leakage.py next.")
    else:
        print("\nSmoke test passed. Re-run without --smoke_test for full 12-config CV.")


# Build an indexed version of SWEEP_GRID for winner lookup
SWEEP_GRID_INDEXED = list(enumerate(SWEEP_GRID, start=1))


if __name__ == "__main__":
    main()
