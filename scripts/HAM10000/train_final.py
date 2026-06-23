"""
Final 5-seed NAM training — v7 corrected pipeline.

A single configurable script used for all four experimental conditions:
  plain_nam         : --concurvity_lambda 0.0 --sparsity_lambda 0.0
  concurvity_only   : --concurvity_lambda 1.0 --sparsity_lambda 0.0
  sparsity_only     : --concurvity_lambda 0.0 --sparsity_lambda <lambda_s>
  sparsity+conc     : --concurvity_lambda 1.0 --sparsity_lambda <lambda_s>

Fixes applied
─────────────
Issue 1/2 : Architecture config read from winner.json (selected by CV, not test).
Issue 3   : random.seed() called (via set_all_seeds).
Issue 7   : per-seed scaler saved to seed_{N}/scaler.pkl.
Issue 8   : CUDA determinism flags set (via set_all_seeds).
Issue 9   : Concurvity warm-up capability added; default set to warmup_epochs=0
             (diagnostic experiment results/HAM10000/diagnostic_warmup/comparison.md
              showed Setting A — no warm-up — is optimal for this dataset).
Issue 10  : weight_decay kept at config value throughout (no zeroing).

Usage (from project root)
──────────────────────────
  python scripts/HAM10000/train_final.py --condition plain_nam
  python scripts/HAM10000/train_final.py --condition concurvity_only --concurvity_lambda 1.0
  python scripts/HAM10000/train_final.py --condition sparsity_only   --sparsity_lambda 23.7
  python scripts/HAM10000/train_final.py --condition sparsity_conc   --concurvity_lambda 1.0 \\
                                               --sparsity_lambda 12.0
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from scripts.v7._common import (
    FEATURES_PATH, SPLITS_PATH, N_FEATURES, N_CLASSES,
    set_all_seeds, load_raw_data, make_fixed_val_split, standardize,
    class_weight_tensor, make_model, make_optimizer_scheduler,
    train_one_run, evaluate_on_test, aggregate_seed_results, write_step_flag,
)
from src.models.sparsity import feature_group_norms

# ── Fixed training settings ───────────────────────────────────────────────────
SEEDS          = [42, 43, 44, 45, 46]
LR             = 1e-3
BATCH_SIZE     = 256
MAX_EPOCHS     = 100
PATIENCE       = 15
SCHED_PATIENCE = 5
SCHED_FACTOR   = 0.5
ZERO_THRESHOLD = 1e-4

CONVERGENCE_THRESHOLD = 0.50
CONVERGENCE_EPOCH     = 30

WINNER_JSON             = "results/HAM10000/architecture_search_cv/winner.json"
CONCURVITY_WINNER_JSON  = "results/HAM10000/concurvity_sweep/winner.json"
RESULTS_V7              = "results/HAM10000"

# Concurvity warm-up: first 5% of MAX_EPOCHS (Issue 9 fix)
WARMUP_EPOCHS = 0   # diagnostic (results/HAM10000/diagnostic_warmup/) confirmed Setting A is optimal

# Step numbers per condition (for flag files)
CONDITION_STEP = {
    "plain_nam":       2,
    "concurvity_only": 4,
    "sparsity_only":   7,
    "sparsity_conc":   7,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True,
        choices=["plain_nam", "concurvity_only", "sparsity_only", "sparsity_conc"],
        help="Experimental condition to train.")
    parser.add_argument("--concurvity_lambda", type=float, default=None)
    parser.add_argument("--sparsity_lambda",   type=float, default=None)
    parser.add_argument("--proximal_sparsity", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--out_dir",   type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience",   type=int, default=None)
    parser.add_argument("--seed",       type=int, default=None,
                        help="Run a single seed only (for testing).")
    parser.add_argument("--skip_convergence_check", action="store_true")
    parser.add_argument("--winner_json", type=str, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None,
                        help="Override concurvity warm-up epochs. "
                             "Default: max(1, int(0.05*max_epochs)) when lambda_c>0, else 0.")
    parser.add_argument("--no_step_flag", action="store_true",
                        help="Skip writing STEP_N_COMPLETE.flag (use for diagnostic runs).")
    args = parser.parse_args()

    # ── Resolve hyperparameters ────────────────────────────────────────────────
    cond = args.condition

    # Concurvity lambda: explicit CLI > concurvity_sweep/winner.json > fallback 1.0
    if args.concurvity_lambda is not None:
        lam_c = args.concurvity_lambda
    elif cond in ("concurvity_only", "sparsity_conc"):
        if os.path.exists(CONCURVITY_WINNER_JSON):
            with open(CONCURVITY_WINNER_JSON) as _f:
                _cw = json.load(_f)
            lam_c = float(_cw["best_concurvity_lambda"])
            print(f"  [lambda_c] read {lam_c} from {CONCURVITY_WINNER_JSON}")
        else:
            lam_c = 1.0
            print(f"  [lambda_c] concurvity_sweep/winner.json not found; "
                  f"defaulting to {lam_c}. Run STEP 3 first or pass --concurvity_lambda.")
    else:
        lam_c = 0.0
    lam_s  = args.sparsity_lambda if args.sparsity_lambda is not None else 0.0
    max_ep = args.max_epochs if args.max_epochs is not None else MAX_EPOCHS
    pat    = args.patience   if args.patience   is not None else PATIENCE
    seeds  = [args.seed]     if args.seed is not None else SEEDS
    proximal = args.proximal_sparsity

    if args.warmup_epochs is not None:
        warmup_ep = args.warmup_epochs  # explicit CLI override
    else:
        warmup_ep = 0                   # default: no warm-up (diagnostic confirmed Setting A)

    # ── Read winner config ─────────────────────────────────────────────────────
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

    # ── Output directory ───────────────────────────────────────────────────────
    default_out = f"results/HAM10000/{cond}"
    out_dir     = args.out_dir if args.out_dir is not None else default_out
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(RESULTS_V7,  exist_ok=True)

    # ── Print banner ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"NAM v7 — Final training [{cond}]")
    print(f"  Audit fixes: Issues 1,2,3,7,8" + (",9" if lam_c > 0 else ""))
    print(f"  Config (from {winner_json_path}):")
    print(f"    hidden={list(hidden_dims)}, dropout={dropout}, wd={weight_decay:.0e}")
    print(f"  lambda_c={lam_c}, lambda_s={lam_s}, proximal={proximal}")
    if lam_c > 0:
        print(f"  Concurvity warm-up: {warmup_ep} epochs (Issue 9 fix)")
    print(f"  seeds={seeds}, max_epochs={max_ep}, patience={pat}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {out_dir}/")
    print(f"{'='*65}\n")

    # ── Data loading ───────────────────────────────────────────────────────────
    raw  = load_raw_data(FEATURES_PATH, SPLITS_PATH)
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

    # ── Training loop ──────────────────────────────────────────────────────────
    all_results = []
    t0 = time.time()

    for seed in seeds:
        print(f"── Seed {seed} " + "─" * 50)
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        set_all_seeds(seed)   # Issues 3, 8

        # ── Per-seed standardization (Issue 7 fix) ─────────────────────────────
        X_tr_sc, X_val_sc, X_test_sc, scaler = standardize(
            val_split["X_train"], val_split["X_val"], X_test_raw
        )
        # Save per-seed scaler (Issue 7)
        with open(os.path.join(seed_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        y_tr_str = val_split["y_train_str"]
        y_tr_enc = val_split["y_train_enc"]
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
            model, LR, weight_decay, SCHED_PATIENCE, SCHED_FACTOR
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
            max_epochs=max_ep,
            patience=pat,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            concurvity_lambda=lam_c,
            warmup_epochs=warmup_ep,    # Issue 9
            sparsity_lambda=lam_s,
            proximal_sparsity=proximal,
            save_path=ckpt_path,
            verbose_every=10,
        )

        result["log_df"].to_csv(os.path.join(seed_dir, "training_log.csv"), index=False)

        # Convergence check
        reached = any(
            row["val_balanced_acc"] >= CONVERGENCE_THRESHOLD
            for _, row in result["log_df"].iterrows()
            if row["epoch"] <= CONVERGENCE_EPOCH
        )
        if not args.skip_convergence_check:
            all_results_so_far_reached = reached
        else:
            all_results_so_far_reached = True

        # Feature group norms
        norms = feature_group_norms(model)
        norms_rows = [
            {"concept_name": nm, "group_norm": nv,
             "is_zeroed": nv < ZERO_THRESHOLD}
            for nm, nv in norms.items()
        ]
        pd.DataFrame(norms_rows).to_csv(
            os.path.join(seed_dir, "feature_group_norms.csv"), index=False
        )
        n_selected = sum(1 for r in norms_rows if not r["is_zeroed"])

        # Test evaluation (once per seed, at the end)
        metrics = evaluate_on_test(model, X_test_t, y_test, class_names)

        best_idx = result["log_df"]["val_balanced_acc"].idxmax()
        r_perp_val_best = float(result["log_df"].loc[best_idx, "r_perp_val"])
        r_perp_tr_best  = float(result["log_df"].loc[best_idx, "r_perp_train"])
        r_sparse_best   = float(result["log_df"].loc[best_idx, "r_sparse_train"])

        early_stopped = result["early_stopped"]
        total_epochs  = result["total_epochs"]
        stop_reason   = "early_stop" if early_stopped else "max_epochs"
        print(
            f"  Test: bal_acc={metrics['balanced_accuracy']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}  "
            f"AUC={metrics['auc_ovr_weighted']:.4f}  "
            f"n_active={n_selected}/{N_FEATURES}"
        )
        print(
            f"  best_epoch={result['best_epoch']}  val_balacc={result['best_val_balacc']:.4f}  "
            f"R_perp_val@best={r_perp_val_best:.4f}  "
            f"total_epochs={total_epochs}  stop={stop_reason}"
        )

        all_results.append({
            "seed":                   seed,
            "best_val_balacc":        result["best_val_balacc"],
            "best_epoch":             result["best_epoch"],
            "n_selected":             n_selected,
            "r_perp_val_at_best":     r_perp_val_best,
            "r_perp_train_at_best":   r_perp_tr_best,
            "r_sparse_train_at_best": r_sparse_best,
            "reached_threshold":      reached,
            "early_stopped":          early_stopped,
            "total_epochs":           total_epochs,
            "log_df":                 result["log_df"],
            **{k: v for k, v in metrics.items()
               if k not in ("report_df", "confusion_matrix", "proba", "y_pred_str")},
            "report_df":        metrics["report_df"],
            "confusion_matrix": metrics["confusion_matrix"],
        })

    # ── Convergence guard ──────────────────────────────────────────────────────
    if not args.skip_convergence_check and not any(
        r["reached_threshold"] for r in all_results
    ):
        raise RuntimeError(
            f"No seed reached val_balacc >= {CONVERGENCE_THRESHOLD} within "
            f"epoch {CONVERGENCE_EPOCH}. Check data/features."
        )

    # ── Aggregate (Issue 5: pandas .std() = ddof=1) ────────────────────────────
    agg_keys = [
        "balanced_accuracy", "macro_f1", "weighted_f1", "top1_accuracy",
        "auc_ovr_weighted", "r_perp_val_at_best", "r_perp_train_at_best",
        "r_sparse_train_at_best",
    ]
    agg_rows = [{"seed": r["seed"], **{k: r[k] for k in agg_keys},
                 "n_selected": r["n_selected"]}
                for r in all_results]
    agg_df   = pd.DataFrame(agg_rows)
    mean_row = {**agg_df[agg_keys].mean().to_dict(), "seed": "mean"}
    std_row  = {**agg_df[agg_keys].std().to_dict(),  "seed": "std"}   # ddof=1
    agg_out  = pd.concat([agg_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    agg_out.to_csv(os.path.join(out_dir, "aggregated_metrics.csv"), index=False)

    means = {k: float(mean_row[k]) for k in agg_keys}
    stds  = {k: float(std_row[k])  for k in agg_keys}

    # Per-class (seed-mean)
    report_mean = (
        pd.concat([r["report_df"] for r in all_results])
        .groupby(level=0).mean()
        .loc[[c for c in all_results[0]["report_df"].index]]
    )
    report_mean["support"] = report_mean["support"].round(0).astype(int)
    report_mean.to_csv(os.path.join(out_dir, "per_class_metrics.csv"))

    # Confusion matrix (seed-mean, row-normalised)
    cms    = np.stack([r["confusion_matrix"] for r in all_results], axis=0)
    cm_mean = cms.mean(axis=0)
    cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)
    pd.DataFrame(cm_norm.round(4), index=class_names, columns=class_names).to_csv(
        os.path.join(out_dir, "confusion_matrix.csv")
    )

    # Training curves plot
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in all_results:
        log = r["log_df"]
        ax.plot(log["epoch"], log["val_balanced_acc"], alpha=0.8,
                label=f"seed {r['seed']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val balanced accuracy")
    ax.set_title(f"NAM v7 [{cond}] — Training curves")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

    # Metadata
    meta = {
        "condition":          cond,
        "hidden_dims":        list(hidden_dims),
        "dropout":            dropout,
        "weight_decay":       weight_decay,
        "lr":                 LR,
        "batch_size":         BATCH_SIZE,
        "max_epochs":         max_ep,
        "patience":           pat,
        "concurvity_lambda":  lam_c,
        "sparsity_lambda":    lam_s,
        "proximal_sparsity":  proximal,
        "warmup_epochs":      warmup_ep,
        "seeds":              seeds,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "winner_json":        winner_json_path,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Summary printout ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"NAM v7 [{cond}] — Final results ({len(seeds)} seeds)")
    print(f"  Balanced accuracy : {means['balanced_accuracy']:.4f} ± {stds['balanced_accuracy']:.4f}")
    print(f"  Macro F1          : {means['macro_f1']:.4f} ± {stds['macro_f1']:.4f}")
    print(f"  AUC (OvR wtd)     : {means['auc_ovr_weighted']:.4f} ± {stds['auc_ovr_weighted']:.4f}")
    print(f"  R_perp val@best   : {means['r_perp_val_at_best']:.4f} ± {stds['r_perp_val_at_best']:.4f}")
    if lam_s > 0:
        print(f"  R_sparse@best     : {means['r_sparse_train_at_best']:.6f}")
    print(f"  Total elapsed     : {elapsed/60:.1f} min")
    print(f"  Outputs → {out_dir}/")
    print(f"{'='*65}")

    # Write step-complete flag (skipped for diagnostic/experimental runs)
    step = CONDITION_STEP.get(cond)
    if step and len(seeds) == len(SEEDS) and not args.no_step_flag:
        write_step_flag(RESULTS_V7, step)


if __name__ == "__main__":
    main()
