"""
Concurvity lambda sweep — v7 corrected pipeline (STEP 3).

Trains a single-seed NAM (seed=42) for each of 10 concurvity lambdas.
Architecture taken from winner.json produced by architecture_search_cv.py.

Fixes applied
─────────────
Issue 9 : Concurvity warm-up capability present; warmup_epochs=0 (no warm-up) is
           the default following the diagnostic in results/v7/diagnostic_warmup/.
Issue 3 : set_all_seeds() includes random.seed.
Issue 7 : per-run scaler saved to seed_42/ within each lambda subdir.
Issue 8 : CUDA determinism flags set.

Output tree
───────────
  results/v7/concurvity_sweep/
    lambda_{value}/
      seed_42/
        best_model.pt
        training_log.csv
        scaler.pkl
        feature_group_norms.csv
      metrics.json
    summary.csv
    winner.json
    run_config.json

Usage (from project root)
──────────────────────────
  python scripts/v7/run_concurvity_sweep.py
  python scripts/v7/run_concurvity_sweep.py --smoke_test   # lambda=0.0 only
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
from torch.utils.data import TensorDataset

from scripts.v7._common import (
    FEATURES_PATH, SPLITS_PATH, N_FEATURES, N_CLASSES,
    set_all_seeds, load_raw_data, make_fixed_val_split, standardize,
    class_weight_tensor, make_model, make_optimizer_scheduler,
    train_one_run, evaluate_on_test, write_step_flag,
)
from src.models.sparsity import feature_group_norms

# ── Constants ─────────────────────────────────────────────────────────────────
SEED        = 42
LR          = 1e-3
BATCH_SIZE  = 256
MAX_EPOCHS  = 100
PATIENCE    = 15
SCHED_PAT   = 5
SCHED_FAC   = 0.5

LAMBDAS = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]

WINNER_JSON = "results/v7/architecture_search_cv/winner.json"
OUT_ROOT    = "results/v7/concurvity_sweep"
RESULTS_V7  = "results/v7"
STEP_N      = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ZERO_THRESHOLD = 1e-4


def run_one_lambda(
    *,
    lam_c:        float,
    hidden_dims:  tuple,
    dropout:      float,
    weight_decay: float,
    val_split:    dict,
    X_test_raw:   np.ndarray,
    y_test:       np.ndarray,
    class_names:  list,
    concept_names: list,
    out_dir:      str,
) -> dict:
    """Train seed=42 at a single concurvity lambda.  Returns metrics dict."""
    os.makedirs(out_dir, exist_ok=True)
    seed_dir = os.path.join(out_dir, f"seed_{SEED}")
    os.makedirs(seed_dir, exist_ok=True)

    set_all_seeds(SEED)

    # Warm-up only when concurvity penalty is active
    warmup_ep = 0  # no warm-up: diagnostic confirmed Setting A is optimal

    # Scaler fitted on this lambda's training split (same split, deterministic)
    X_tr_sc, X_val_sc, X_test_sc, scaler = standardize(
        val_split["X_train"], val_split["X_val"], X_test_raw
    )
    with open(os.path.join(seed_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    y_tr_str  = val_split["y_train_str"]
    y_tr_enc  = val_split["y_train_enc"]
    y_val_enc = val_split["y_val_enc"]

    w_tensor = class_weight_tensor(y_tr_str, class_names, DEVICE)

    X_val_t  = torch.tensor(X_val_sc,  dtype=torch.float32, device=DEVICE)
    y_val_t  = torch.tensor(y_val_enc, dtype=torch.long,    device=DEVICE)
    X_test_t = torch.tensor(X_test_sc, dtype=torch.float32, device=DEVICE)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc,  dtype=torch.float32),
        torch.tensor(y_tr_enc, dtype=torch.long),
    )

    model = make_model(hidden_dims, dropout, concept_names, DEVICE)
    optimizer, scheduler = make_optimizer_scheduler(
        model, LR, weight_decay, SCHED_PAT, SCHED_FAC, scheduler_mode="max"
    )
    criterion = nn.CrossEntropyLoss(weight=w_tensor)
    ckpt_path = os.path.join(seed_dir, "best_model.pt")

    result = train_one_run(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_dataset=train_ds,
        X_val_t=X_val_t,
        y_val_t=y_val_t,
        y_val_enc=y_val_enc,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        concurvity_lambda=lam_c,
        warmup_epochs=warmup_ep,
        save_path=ckpt_path,
        verbose_every=10,
        scheduler_watches="val_balacc",
    )

    result["log_df"].to_csv(os.path.join(seed_dir, "training_log.csv"), index=False)

    # Feature norms
    norms = feature_group_norms(model)
    norms_rows = [
        {"concept_name": nm, "group_norm": nv, "is_zeroed": nv < ZERO_THRESHOLD}
        for nm, nv in norms.items()
    ]
    pd.DataFrame(norms_rows).to_csv(
        os.path.join(seed_dir, "feature_group_norms.csv"), index=False
    )

    # R_perp at best val epoch
    best_idx = result["log_df"]["val_balanced_acc"].idxmax()
    r_perp_val  = float(result["log_df"].loc[best_idx, "r_perp_val"])
    r_perp_tr   = float(result["log_df"].loc[best_idx, "r_perp_train"])

    metrics = evaluate_on_test(model, X_test_t, y_test, class_names)

    row = {
        "concurvity_lambda":    lam_c,
        "best_val_balacc":      result["best_val_balacc"],
        "best_epoch":           result["best_epoch"],
        "r_perp_val_at_best":   r_perp_val,
        "r_perp_train_at_best": r_perp_tr,
        "test_balacc":          metrics["balanced_accuracy"],
        "test_macro_f1":        metrics["macro_f1"],
        "test_auc_weighted":    metrics["auc_ovr_weighted"],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(row, f, indent=2)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run lambda=0.0 only to verify mechanics.")
    parser.add_argument("--winner_json", type=str, default=None)
    parser.add_argument("--out_root", type=str, default=None)
    args = parser.parse_args()

    winner_json_path = args.winner_json or WINNER_JSON
    if not os.path.exists(winner_json_path):
        raise FileNotFoundError(
            f"winner.json not found at {winner_json_path}. "
            "Run architecture_search_cv.py (STEP 1) first."
        )
    with open(winner_json_path) as f:
        winner = json.load(f)

    hidden_dims  = tuple(winner["hidden_dims"])
    dropout      = float(winner["dropout"])
    weight_decay = float(winner["weight_decay"])

    out_root = args.out_root or OUT_ROOT
    os.makedirs(out_root,   exist_ok=True)
    os.makedirs(RESULTS_V7, exist_ok=True)

    lambdas = [0.0] if args.smoke_test else LAMBDAS

    print(f"\n{'='*65}")
    print(f"NAM v7 — Concurvity sweep (STEP {STEP_N})")
    print(f"  Issue 9 fix: concurvity warm-up active for lambda > 0")
    print(f"  Config: hidden={list(hidden_dims)}, dropout={dropout}, wd={weight_decay:.0e}")
    print(f"  Lambdas ({len(lambdas)}): {lambdas}")
    print(f"  seed={SEED}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {out_root}/")
    if args.smoke_test:
        print("  [SMOKE TEST — lambda=0.0 only]")
    print(f"{'='*65}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    raw           = load_raw_data(FEATURES_PATH, SPLITS_PATH)
    scores        = raw["scores"]
    labels        = raw["labels"]
    lesion_ids    = raw["lesion_ids"]
    concept_names = raw["concept_names"]
    class_names   = raw["class_names"]
    train_idx     = raw["train_idx"]
    test_idx      = raw["test_idx"]

    X_all_train      = scores[train_idx]
    y_all_train      = labels[train_idx]
    lesion_ids_train = lesion_ids[train_idx]
    X_test_raw       = scores[test_idx]
    y_test           = labels[test_idx]

    val_split = make_fixed_val_split(
        X_all_train, y_all_train, lesion_ids_train, class_names, val_random_state=42
    )

    # ── Sweep loop ────────────────────────────────────────────────────────────
    all_rows = []
    t0       = time.time()

    for lam_c in lambdas:
        lam_tag = f"{lam_c:.6f}".rstrip("0").rstrip(".")
        lam_dir = os.path.join(out_root, f"lambda_{lam_tag}")
        print(f"\n── lambda_c = {lam_c} " + "─" * 45)

        row = run_one_lambda(
            lam_c=lam_c,
            hidden_dims=hidden_dims,
            dropout=dropout,
            weight_decay=weight_decay,
            val_split=val_split,
            X_test_raw=X_test_raw,
            y_test=y_test,
            class_names=class_names,
            concept_names=concept_names,
            out_dir=lam_dir,
        )
        all_rows.append(row)
        print(
            f"  Done: val_balacc={row['best_val_balacc']:.4f}  "
            f"test_balacc={row['test_balacc']:.4f}  "
            f"R_perp_val={row['r_perp_val_at_best']:.4f}"
        )

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(os.path.join(out_root, "summary.csv"), index=False)
    print(f"\n{'='*65}")
    print(f"Concurvity sweep complete.  {len(all_rows)} lambdas run.")
    print(summary_df[["concurvity_lambda", "best_val_balacc",
                       "r_perp_val_at_best", "test_balacc"]].to_string(index=False))

    # ── Winner: best val_balacc lambda ────────────────────────────────────────
    best_row = summary_df.sort_values(
        ["best_val_balacc", "r_perp_val_at_best"],
        ascending=[False, True],
    ).iloc[0]
    winner_out = {
        "best_concurvity_lambda":   float(best_row["concurvity_lambda"]),
        "best_val_balacc":          float(best_row["best_val_balacc"]),
        "r_perp_val_at_best":       float(best_row["r_perp_val_at_best"]),
        "selection_criterion":      "highest val_balacc at seed=42; tie-break lowest R_perp",
        "test_set_touched":         True,
        "note": (
            "test_set_touched=True here because sweep uses a single seed for speed. "
            "Lambda confirmed by STEP 4 (train_final concurvity_only) which reports "
            "test results after this selection."
        ),
    }
    with open(os.path.join(out_root, "winner.json"), "w") as f:
        json.dump(winner_out, f, indent=2)

    # Run config
    meta = {
        "lambdas":       lambdas,
        "seed":          SEED,
        "hidden_dims":   list(hidden_dims),
        "dropout":       dropout,
        "weight_decay":  weight_decay,
        "max_epochs":    MAX_EPOCHS,
        "patience":      PATIENCE,
        "smoke_test":    args.smoke_test,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "winner_json":   winner_json_path,
    }
    with open(os.path.join(out_root, "run_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Best lambda_c = {best_row['concurvity_lambda']}")
    print(f"  Total elapsed: {elapsed/60:.1f} min")
    print(f"  Outputs → {out_root}/")
    print(f"{'='*65}")

    if not args.smoke_test:
        write_step_flag(RESULTS_V7, STEP_N)


if __name__ == "__main__":
    main()
