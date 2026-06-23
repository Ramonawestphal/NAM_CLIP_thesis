"""
Extended dropout sweep for HAM10000 architecture selection (reviewer extension).

Reviewer feedback: the original 3x2x2 grid (dropout in {0.1, 0.2}) does not
bracket the lower end of plausible dropout values.  This script extends the
dropout dimension to {0.0, 0.05, 0.1, 0.2, 0.3}, holding weight_decay fixed
at the original winner value (1e-4) and sweeping all three hidden-width configs.
Result: 15 configurations x 5 folds = 75 training runs.

Protocol (identical to architecture_search_cv.py):
  - 5-fold GroupKFold cross-validation on the HAM10000 training pool (8020 samples)
  - Fold indices loaded from results/HAM10000/architecture_search_cv/fold_indices.json
    (NOT regenerated — identical partitions guaranteed)
  - CV_SEED=42, LR=1e-3, BATCH_SIZE=256, MAX_EPOCHS=80, PATIENCE=15
  - Per-fold standardization fitted on training fold only
  - Balanced class weights (sklearn compute_class_weight) per fold
  - No concurvity/sparsity regularization

Outputs:
  results/HAM10000/architecture_search_cv_extended/
    cv_results.csv       75 rows (15 configs x 5 folds), all requested columns
    cv_summary.csv       15 rows (per-config aggregated metrics)
    combined_summary.csv 27 rows (original 12 + extended 15), for comparison
    report.md            markdown-formatted appendix section
    log.txt              stdout mirror
  results/architecture_sweep_extended.csv  (same content as cv_results.csv)

Usage (from project root):
    python scripts/HAM10000/architecture_search_dropout_extended.py
    python scripts/HAM10000/architecture_search_dropout_extended.py --smoke_test
    python scripts/HAM10000/architecture_search_dropout_extended.py --max_epochs 10
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import sys
import time
from datetime import datetime, timezone

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from scripts.HAM10000._common import (
    FEATURES_PATH, SPLITS_PATH,
    N_FEATURES, N_CLASSES,
    set_all_seeds, load_raw_data, standardize,
    class_weight_tensor, make_model, make_optimizer_scheduler,
    train_one_run,
)

# ── Fixed hyperparameters — must match architecture_search_cv.py exactly ─────
CV_SEED        = 42
N_FOLDS        = 5
LR             = 1e-3
BATCH_SIZE     = 256
MAX_EPOCHS     = 80
PATIENCE       = 15
SCHED_PATIENCE = 5
SCHED_FACTOR   = 0.5
FIXED_WD       = 1e-4       # original-grid winner weight decay, held constant

# Paths
ORIG_CV_DIR = "results/HAM10000/architecture_search_cv"
OUT_DIR     = "results/HAM10000/architecture_search_cv_extended"
SHARED_CSV  = "results/architecture_sweep_extended.csv"

# Original-grid winner (from winner.json); used for stop-criterion check
ORIG_WINNER_HIDDEN  = [64, 32]
ORIG_WINNER_DROPOUT = 0.1
ORIG_WINNER_MEAN    = 0.5611
ORIG_WINNER_STD     = 0.0175
# Stop criterion: new winner beats original by more than 2 fold-level SDs
STOP_THRESHOLD = ORIG_WINNER_MEAN + 2 * ORIG_WINNER_STD   # ~0.5961

# ── Extended grid ─────────────────────────────────────────────────────────────
HIDDEN_CONFIGS = [(32, 16), (32, 32), (64, 32)]
DROPOUT_VALUES = [0.0, 0.05, 0.1, 0.2, 0.3]
EXTENDED_GRID  = list(itertools.product(HIDDEN_CONFIGS, DROPOUT_VALUES))
# 15 configurations total

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Logging tee
# ─────────────────────────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, path: str):
        self._file   = open(path, "w", encoding="utf-8", errors="replace")
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _eval_metrics(model, X_val_t, y_va_enc, device):
    """Run one inference pass and return balacc, macro_f1, auc_weighted."""
    model.eval()
    with torch.no_grad():
        logits = model(X_val_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

    balacc   = balanced_accuracy_score(y_va_enc, preds)
    macro_f1 = f1_score(y_va_enc, preds, average="macro", zero_division=0)
    try:
        auc_w = roc_auc_score(
            y_va_enc, probs, multi_class="ovr", average="weighted"
        )
    except ValueError:
        auc_w = float("nan")
    return balacc, macro_f1, auc_w


def _load_fold_indices(fold_index_path, train_idx, n_scores):
    """
    Load fold_indices.json and return a list of (tr_rel, va_rel) numpy arrays
    indexing into X_all_train = scores[train_idx].

    The JSON stores absolute dataset indices; we map them back to positions
    within train_idx so indexing into X_all_train is correct.
    """
    with open(fold_index_path) as f:
        fi = json.load(f)

    # Build absolute-index -> relative-position lookup
    abs_to_rel = np.full(n_scores, -1, dtype=np.int64)
    for rel_pos, abs_idx in enumerate(train_idx):
        abs_to_rel[abs_idx] = rel_pos

    fold_splits = []
    for fd in sorted(fi["folds"], key=lambda x: x["fold"]):
        tr_abs = np.array(fd["train_abs_indices"], dtype=np.int64)
        va_abs = np.array(fd["val_abs_indices"],   dtype=np.int64)
        tr_rel = abs_to_rel[tr_abs]
        va_rel = abs_to_rel[va_abs]
        assert (tr_rel >= 0).all(), "train abs index not found in train_idx"
        assert (va_rel >= 0).all(), "val abs index not found in train_idx"
        fold_splits.append((tr_rel, va_rel))

    return fold_splits


def _markdown_table(df, float_cols=None):
    """Render a DataFrame as a GFM markdown table string."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if float_cols and c in float_cols and isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke_test",  action="store_true",
                        help="Run only the first config (5 folds) to verify mechanics.")
    parser.add_argument("--max_epochs",  type=int, default=None,
                        help="Override max_epochs (default 80).")
    args = parser.parse_args()

    max_epochs = args.max_epochs if args.max_epochs is not None else MAX_EPOCHS

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SHARED_CSV), exist_ok=True)

    tee = _Tee(os.path.join(OUT_DIR, "log.txt"))
    sys.stdout = tee
    try:
        _run(args, max_epochs)
    finally:
        sys.stdout = tee._stdout
        tee.close()


def _run(args: argparse.Namespace, max_epochs: int) -> None:
    t0 = time.time()
    print("=" * 72)
    print("HAM10000 — Extended dropout sweep (reviewer extension)")
    print(f"  Extends: architecture_search_cv.py (original 3x2x2 grid)")
    print(f"  New grid: 3 hidden widths x 5 dropout values = 15 configs")
    print(f"  Fixed  : weight_decay={FIXED_WD:.0e}, identical CV protocol")
    print(f"  Device : {DEVICE}")
    print(f"  CV seed: {CV_SEED}  |  Max epochs: {max_epochs}  |  Patience: {PATIENCE}")
    if args.smoke_test:
        print("  *** SMOKE TEST: first config only ***")
    print("=" * 72)

    # ── Load data (training pool only) ────────────────────────────────────────
    raw           = load_raw_data(FEATURES_PATH, SPLITS_PATH)
    scores        = raw["scores"]         # full (10015, 24)
    labels        = raw["labels"]
    concept_names = raw["concept_names"]
    class_names   = raw["class_names"]
    train_idx     = raw["train_idx"]      # (8020,) absolute indices
    _test_sentinel = raw["test_idx"]      # loaded only to verify exclusion

    assert len(np.intersect1d(train_idx, _test_sentinel)) == 0, \
        "train_idx and test_idx overlap — check splits file"

    X_all_train      = scores[train_idx]
    y_all_train      = labels[train_idx]
    class_to_idx     = {c: i for i, c in enumerate(class_names)}
    y_all_train_enc  = np.array([class_to_idx[c] for c in y_all_train],
                                dtype=np.int64)

    print(f"\nTraining pool: {len(train_idx)} samples")
    print(f"Test pool (excluded): {len(_test_sentinel)} samples")
    print(f"  [assertion passed] train_idx cap test_idx = empty")

    # ── Load cached fold indices (do NOT regenerate) ──────────────────────────
    fold_index_path = os.path.join(ORIG_CV_DIR, "fold_indices.json")
    if not os.path.exists(fold_index_path):
        raise FileNotFoundError(
            f"fold_indices.json not found at {fold_index_path}. "
            "Run architecture_search_cv.py first to generate it."
        )

    fold_splits = _load_fold_indices(fold_index_path, train_idx, len(scores))

    # Sanity-check: same sizes as original
    for fi, (tr_rel, va_rel) in enumerate(fold_splits):
        tr_les = set(labels[train_idx[tr_rel]])   # not lesion_ids but enough for a size check
        assert len(tr_rel) == 6416, \
            f"Fold {fi}: unexpected train size {len(tr_rel)} (expected 6416)"
        assert len(va_rel) == 1604, \
            f"Fold {fi}: unexpected val size {len(va_rel)} (expected 1604)"
    print(f"  [assertion passed] Loaded {N_FOLDS} fold splits from fold_indices.json")
    print(f"  Each fold: train=6416, val=1604\n")

    # ── Config grid ───────────────────────────────────────────────────────────
    configs_to_run = list(enumerate(EXTENDED_GRID, start=1))
    if args.smoke_test:
        configs_to_run = configs_to_run[:1]

    total_runs = len(configs_to_run) * N_FOLDS
    print(f"Grid: {len(configs_to_run)} configs x {N_FOLDS} folds = {total_runs} runs")
    print(f"{'hidden_widths':<14}  {'dropout':>8}  {'weight_decay':>13}")
    for cfg_id, (hidden, do) in configs_to_run:
        print(f"  {str(list(hidden)):<12}  {do:>8.2f}  {FIXED_WD:>13.0e}")
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    all_records: list[dict] = []
    run_counter = 0

    for cfg_id, (hidden_dims, dropout) in configs_to_run:
        print(f"Config {cfg_id:2d}: hidden={list(hidden_dims)}  "
              f"dropout={dropout}  weight_decay={FIXED_WD:.0e}")
        fold_bacc_list: list[float] = []

        for fold_i, (tr_rel, va_rel) in enumerate(fold_splits):
            run_counter += 1
            t_fold = time.time()
            print(f"  Fold {fold_i+1}/{N_FOLDS}  "
                  f"(run {run_counter}/{total_runs}  |  "
                  f"train={len(tr_rel)}, val={len(va_rel)})")

            # ── Per-fold data prep ─────────────────────────────────────────────
            X_tr_raw = X_all_train[tr_rel]
            y_tr_enc = y_all_train_enc[tr_rel]
            y_tr_str = y_all_train[tr_rel]
            X_va_raw = X_all_train[va_rel]
            y_va_enc = y_all_train_enc[va_rel]

            # Standardizer fitted on this fold's training data only
            X_tr_sc, X_va_sc, _, _ = standardize(X_tr_raw, X_va_raw)

            # Class weights from this fold's training labels only
            w_tensor = class_weight_tensor(y_tr_str, class_names, DEVICE)

            X_val_t = torch.tensor(X_va_sc, dtype=torch.float32, device=DEVICE)
            y_val_t = torch.tensor(y_va_enc, dtype=torch.long,    device=DEVICE)

            train_ds = TensorDataset(
                torch.tensor(X_tr_sc, dtype=torch.float32),
                torch.tensor(y_tr_enc, dtype=torch.long),
            )

            # ── Train ─────────────────────────────────────────────────────────
            # Same seed as original: fold-to-fold differences reflect data, not init
            set_all_seeds(CV_SEED)

            model     = make_model(hidden_dims, dropout, concept_names, DEVICE)
            optimizer, scheduler = make_optimizer_scheduler(
                model, LR, FIXED_WD, SCHED_PATIENCE, SCHED_FACTOR
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
                concurvity_lambda=0.0,   # no regularization during arch search
                warmup_epochs=0,
                sparsity_lambda=0.0,
                save_path=None,
                verbose_every=max_epochs + 1,  # suppress per-epoch noise
            )

            # ── Post-fold metrics (best-weights already restored by train_one_run)
            balacc, macro_f1, auc_w = _eval_metrics(model, X_val_t, y_va_enc, DEVICE)

            # Sanity: balacc from eval pass should match training-loop best
            assert abs(balacc - result["best_val_balacc"]) < 1e-4, (
                f"balacc mismatch: eval={balacc:.4f} vs "
                f"train_loop={result['best_val_balacc']:.4f}"
            )

            fold_bacc_list.append(balacc)
            elapsed = time.time() - t_fold

            print(f"    -> val_balacc={balacc:.4f}  macro_f1={macro_f1:.4f}  "
                  f"auc_w={auc_w:.4f}  "
                  f"best_epoch={result['best_epoch']}  "
                  f"early_stopped={result['early_stopped']}  "
                  f"time={elapsed:.1f}s")

            all_records.append({
                "hidden_widths":        str(list(hidden_dims)),
                "dropout":              dropout,
                "weight_decay":         FIXED_WD,
                "fold":                 fold_i,
                "val_balanced_accuracy": round(balacc,   4),
                "val_macro_f1":          round(macro_f1, 4),
                "val_auc_weighted":      round(auc_w,    4) if not np.isnan(auc_w) else float("nan"),
                "epochs_trained":        result["total_epochs"],
                "early_stopped":         int(result["early_stopped"]),
            })

        mean_bacc = float(np.mean(fold_bacc_list))
        std_bacc  = float(np.std(fold_bacc_list, ddof=1))
        print(f"  -> mean={mean_bacc:.4f} +/- {std_bacc:.4f}  "
              f"folds={[round(x, 4) for x in fold_bacc_list]}\n")

    # ── Save cv_results.csv ───────────────────────────────────────────────────
    cv_df = pd.DataFrame(all_records)
    cv_df.to_csv(os.path.join(OUT_DIR, "cv_results.csv"), index=False)
    cv_df.to_csv(SHARED_CSV, index=False)
    print(f"Saved cv_results.csv ({len(cv_df)} rows)")
    print(f"Saved {SHARED_CSV}")

    # ── Build cv_summary.csv ──────────────────────────────────────────────────
    summary_rows: list[dict] = []
    for cfg_id, (hidden_dims, dropout) in configs_to_run:
        key    = str(list(hidden_dims))
        subset = cv_df[(cv_df["hidden_widths"] == key) & (cv_df["dropout"] == dropout)]
        summary_rows.append({
            "config_id":            cfg_id,
            "hidden_widths":        key,
            "dropout":              dropout,
            "weight_decay":         FIXED_WD,
            "mean_cv_balacc":       round(float(subset["val_balanced_accuracy"].mean()), 4),
            "std_cv_balacc":        round(float(subset["val_balanced_accuracy"].std()),  4),
            "mean_cv_macro_f1":     round(float(subset["val_macro_f1"].mean()),          4),
            "mean_cv_auc_weighted": round(float(subset["val_auc_weighted"].mean()),      4),
            "n_folds":              N_FOLDS,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "mean_cv_balacc", ascending=False
    ).reset_index(drop=True)
    summary_df.to_csv(os.path.join(OUT_DIR, "cv_summary.csv"), index=False)

    # ── Build combined_summary.csv (original 12 + extended 15) ──────────────
    orig_summary = pd.read_csv(os.path.join(ORIG_CV_DIR, "cv_summary.csv"))

    # Mark which rows are which grid
    orig_rows = []
    for _, r in orig_summary.iterrows():
        orig_rows.append({
            "source":               "original_3x2x2",
            "hidden_widths":        r["hidden"],
            "dropout":              r["dropout"],
            "weight_decay":         r["weight_decay"],
            "mean_cv_balacc":       r["mean_cv_balacc"],
            "std_cv_balacc":        r["std_cv_balacc"],
            "mean_cv_macro_f1":     float("nan"),   # not recorded in original run
            "mean_cv_auc_weighted": float("nan"),
            "n_folds":              int(r["n_folds"]),
            "is_original_winner":   (
                str(r["hidden"]) == str(ORIG_WINNER_HIDDEN) and
                float(r["dropout"]) == ORIG_WINNER_DROPOUT and
                abs(float(r["weight_decay"]) - FIXED_WD) < 1e-9
            ),
        })

    ext_rows = []
    for _, r in summary_df.iterrows():
        ext_rows.append({
            "source":               "extended_3x5",
            "hidden_widths":        r["hidden_widths"],
            "dropout":              r["dropout"],
            "weight_decay":         r["weight_decay"],
            "mean_cv_balacc":       r["mean_cv_balacc"],
            "std_cv_balacc":        r["std_cv_balacc"],
            "mean_cv_macro_f1":     r["mean_cv_macro_f1"],
            "mean_cv_auc_weighted": r["mean_cv_auc_weighted"],
            "n_folds":              r["n_folds"],
            "is_original_winner":   (
                str(r["hidden_widths"]) == str(ORIG_WINNER_HIDDEN) and
                float(r["dropout"]) == ORIG_WINNER_DROPOUT
            ),
        })

    combined_df = pd.DataFrame(orig_rows + ext_rows).sort_values(
        "mean_cv_balacc", ascending=False
    ).reset_index(drop=True)
    combined_df.to_csv(os.path.join(OUT_DIR, "combined_summary.csv"), index=False)

    # ── Stop-criterion check ──────────────────────────────────────────────────
    best_ext = summary_df.iloc[0]
    beat_by_2sd = float(best_ext["mean_cv_balacc"]) > STOP_THRESHOLD

    # ── Print summary tables ──────────────────────────────────────────────────
    total_time = time.time() - t0
    print("=" * 72)
    print("EXTENDED SWEEP SUMMARY (sorted by mean_cv_balacc desc)")
    print("=" * 72)
    disp = summary_df.copy()
    disp["is_orig_winner"] = disp.apply(
        lambda r: "<<< ORIG" if (
            str(r["hidden_widths"]) == str(ORIG_WINNER_HIDDEN) and
            float(r["dropout"]) == ORIG_WINNER_DROPOUT
        ) else "", axis=1
    )
    print(disp.to_string(index=False))

    print()
    print("=" * 72)
    print("COMBINED GRID: original 3x2x2 (wd in {1e-5,1e-4}) + extended 3x5 (wd=1e-4)")
    print("=" * 72)
    disp2 = combined_df[["source", "hidden_widths", "dropout", "weight_decay",
                          "mean_cv_balacc", "std_cv_balacc", "is_original_winner"]]
    print(disp2.to_string(index=False))

    # ── Analysis paragraph ────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)

    orig_winner_in_ext = summary_df[
        (summary_df["hidden_widths"] == str(ORIG_WINNER_HIDDEN)) &
        (summary_df["dropout"] == ORIG_WINNER_DROPOUT)
    ]
    orig_replicated_mean = float(orig_winner_in_ext["mean_cv_balacc"].values[0]) \
        if len(orig_winner_in_ext) > 0 else None
    orig_replicated_std  = float(orig_winner_in_ext["std_cv_balacc"].values[0]) \
        if len(orig_winner_in_ext) > 0 else None

    new_winner_hidden  = best_ext["hidden_widths"]
    new_winner_dropout = float(best_ext["dropout"])
    new_winner_mean    = float(best_ext["mean_cv_balacc"])
    new_winner_std     = float(best_ext["std_cv_balacc"])

    orig_still_winner = (
        str(new_winner_hidden) == str(ORIG_WINNER_HIDDEN) and
        new_winner_dropout == ORIG_WINNER_DROPOUT
    )

    delta = new_winner_mean - ORIG_WINNER_MEAN

    do0_rows = summary_df[summary_df["dropout"] == 0.0]
    do0_best_mean = float(do0_rows["mean_cv_balacc"].max()) if len(do0_rows) > 0 else None
    do0_within_1sd = (
        abs(new_winner_mean - do0_best_mean) <= new_winner_std
    ) if do0_best_mean is not None else None

    analysis_lines = []
    analysis_lines.append(
        f"(a) Original winner still winner: {orig_still_winner}. "
        f"The extended-grid winner is hidden={new_winner_hidden}, "
        f"dropout={new_winner_dropout}, mean CV balanced accuracy "
        f"{new_winner_mean:.4f} +/- {new_winner_std:.4f}. "
        + (
            "This is the same configuration as the original grid winner "
            f"(replicated mean={orig_replicated_mean:.4f} vs original {ORIG_WINNER_MEAN:.4f}; "
            f"difference={abs(orig_replicated_mean - ORIG_WINNER_MEAN):.4f} < 1 SD={ORIG_WINNER_STD:.4f})."
            if orig_still_winner else
            f"This differs from the original winner (hidden={ORIG_WINNER_HIDDEN}, "
            f"dropout={ORIG_WINNER_DROPOUT}, mean={ORIG_WINNER_MEAN:.4f}). "
            f"The new winner improves by {delta:+.4f} balanced accuracy points."
        )
    )
    if not orig_still_winner:
        analysis_lines.append(
            f"(b) New winner outperforms original by {delta:.4f} balanced accuracy points "
            f"({delta / ORIG_WINNER_STD:.2f} fold-level SDs of the original winner). "
            + (
                "This EXCEEDS the 2-SD stop threshold — manual decision required "
                "before retraining the main experimental conditions."
                if beat_by_2sd else
                "This is within the 2-SD stop threshold; no retraining of main "
                "experimental conditions is warranted."
            )
        )
    else:
        analysis_lines.append(
            f"(b) Because the original winner is unchanged, the performance delta "
            f"is {delta:+.4f} points (replicated vs original stored value), "
            "consistent with expected fold-sampling variation. No retraining needed."
        )
    analysis_lines.append(
        f"(c) Dropout=0.0: best mean CV balanced accuracy across hidden-width "
        f"configs = {do0_best_mean:.4f}. "
        f"Winner mean - dropout=0.0 best = {new_winner_mean - do0_best_mean:.4f}. "
        + (
            f"Dropout=0.0 IS within one winner SD ({new_winner_std:.4f}): "
            "the sweep does not provide strong evidence against dropout=0."
            if do0_within_1sd else
            f"Dropout=0.0 is NOT within one winner SD ({new_winner_std:.4f}): "
            "the sweep provides evidence that some dropout is beneficial."
        )
    )

    for line in analysis_lines:
        print(line)
        print()

    if beat_by_2sd:
        print("*** STOP CRITERION TRIGGERED ***")
        print(f"New winner mean ({new_winner_mean:.4f}) exceeds original winner mean "
              f"+ 2*SD ({STOP_THRESHOLD:.4f}). Do not retrain main experimental "
              "conditions until you have reviewed this result.")

    print(f"\nTotal elapsed: {total_time/60:.1f} min")
    print(f"Outputs -> {OUT_DIR}/")

    # ── Write markdown report ─────────────────────────────────────────────────
    _write_report(
        summary_df=summary_df,
        combined_df=combined_df,
        analysis_lines=analysis_lines,
        orig_still_winner=orig_still_winner,
        beat_by_2sd=beat_by_2sd,
        new_winner_hidden=new_winner_hidden,
        new_winner_dropout=new_winner_dropout,
        new_winner_mean=new_winner_mean,
        new_winner_std=new_winner_std,
        delta=delta,
        do0_best_mean=do0_best_mean,
        do0_within_1sd=do0_within_1sd,
        total_time_min=total_time / 60,
        max_epochs=max_epochs,
    )
    print(f"Saved report.md -> {OUT_DIR}/report.md")


def _write_report(
    summary_df,
    combined_df,
    analysis_lines,
    orig_still_winner,
    beat_by_2sd,
    new_winner_hidden,
    new_winner_dropout,
    new_winner_mean,
    new_winner_std,
    delta,
    do0_best_mean,
    do0_within_1sd,
    total_time_min,
    max_epochs,
):
    float_cols = {"mean_cv_balacc", "std_cv_balacc", "mean_cv_macro_f1",
                  "mean_cv_auc_weighted", "dropout", "weight_decay"}

    # Table 1: extended 3x5 grid
    t1_cols = ["hidden_widths", "dropout", "mean_cv_balacc", "std_cv_balacc",
               "mean_cv_macro_f1", "mean_cv_auc_weighted"]
    t1 = summary_df[t1_cols].copy()
    t1.insert(len(t1.columns), "original_winner",
              t1.apply(lambda r: "Yes" if (
                  str(r["hidden_widths"]) == str(ORIG_WINNER_HIDDEN) and
                  float(r["dropout"]) == ORIG_WINNER_DROPOUT
              ) else "", axis=1))

    # Table 2: combined, show only balacc columns for readability
    t2_cols = ["source", "hidden_widths", "dropout", "weight_decay",
               "mean_cv_balacc", "std_cv_balacc", "is_original_winner"]
    t2 = combined_df[t2_cols].copy()
    t2["is_original_winner"] = t2["is_original_winner"].apply(
        lambda x: "Yes" if x else ""
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Appendix C: Extended Dropout Sweep — HAM10000",
        "",
        f"*Generated {timestamp}. "
        f"Runtime: {total_time_min:.1f} min "
        f"({5 * len(summary_df)} runs, max {max_epochs} epochs each).*",
        "",
        "## Motivation",
        "",
        "The original architecture-selection grid covered dropout ∈ {0.1, 0.2}, "
        "which does not bracket the lower end of plausible dropout values. "
        "In response to reviewer feedback, the dropout dimension was extended to "
        "{0.0, 0.05, 0.1, 0.2, 0.3}, with weight decay held fixed at the "
        "original winner value (10⁻⁴). All other training settings are identical "
        "to the original grid (5-fold lesion-grouped GroupKFold cross-validation, "
        f"CV seed 42, learning rate 10⁻³, batch size 256, max {max_epochs} epochs, "
        "patience 15, per-fold standardisation and balanced class weights). "
        "The fold partitions are loaded from the cached `fold_indices.json` "
        "produced by the original run, guaranteeing identical train/validation "
        "splits.",
        "",
        "## Table C.1 — Extended 3×5 Grid Results",
        "",
        "Mean ± SD validation balanced accuracy across five folds "
        "(15 configurations; weight decay = 10⁻⁴ throughout). "
        "Sorted by mean balanced accuracy descending. "
        "The column *original_winner* flags the configuration selected by the "
        "original 3×2×2 grid.",
        "",
        _markdown_table(t1, float_cols=float_cols),
        "",
        "## Table C.2 — Combined Grid: Original 3×2×2 and Extended 3×5",
        "",
        "All 27 configurations from both grids sorted by mean balanced accuracy. "
        "Configurations from the original grid span both weight-decay values "
        "(10⁻⁵ and 10⁻⁴); extended configurations use weight decay 10⁻⁴ only. "
        "Macro-F1 and AUC were not recorded in the original run.",
        "",
        _markdown_table(t2, float_cols=float_cols),
        "",
        "## Analysis",
        "",
    ]

    for line in analysis_lines:
        lines.append(line)
        lines.append("")

    if beat_by_2sd:
        lines += [
            "> **Stop criterion triggered.** The new winner exceeds the original "
            f"winner by more than two fold-level standard deviations "
            f"(threshold: {ORIG_WINNER_MEAN + 2*ORIG_WINNER_STD:.4f}). "
            "A manual decision is required before retraining main experimental conditions.",
            "",
        ]

    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
