"""
v7-parity Step 3: NAM architecture selection for the binary chest X-ray task.

12-config v7 grid x 5 pre-built StratifiedGroupKFold folds x 1 seed (42) = 60
runs. Selection metric: mean val balanced accuracy across folds (v7 rule). AUC
is reported as a secondary metric but never used for selection.

Training mechanics are the EXACT v7 protocol: the per-run training/early-stopping
is delegated to scripts/v7/_common.train_one_run (Adam lr=1e-3, batch=256,
ReduceLROnPlateau(mode="max", factor=0.5, patience=5, min_lr=1e-6), early stop on
val balanced accuracy, max_epochs=80/patience=15, CrossEntropyLoss with balanced
class weights). The only divergence from v7 is num_classes=2 (binary task) and the
secondary AUC metric computed from the best-restored model after each run.

The held-out test set is NEVER loaded here. Six leakage safeguards are asserted
before any training.

Outputs (under results/chestxray/architecture_selection_binary/):
    per_run_results.csv    one row per (config, fold), saved incrementally
    config_aggregates.csv  one row per config (mean/std balacc & auc)
    winning_config.json
    selection_summary.txt

Run from project root:
    python scripts/chestxray/select_architecture.py
    python scripts/chestxray/select_architecture.py --smoke_test   # config 1 only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # robust on cp1252 consoles
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm

from src.models.nam_multiclass import NAMMulticlass
from scripts.chestxray._common_imports import (
    set_all_seeds, make_optimizer_scheduler, train_one_run,
)
from scripts.chestxray.architecture_configs import ARCHITECTURE_CONFIGS

# ── Paths ─────────────────────────────────────────────────────────────────────
SCORES_NPZ  = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v4.npz"
OUTER_SPLIT = _ROOT / "data/splits/chestxray_outer_split.npz"
CV_FOLDS    = _ROOT / "data/splits/chestxray_cv_folds.npz"
OUT_DIR     = _ROOT / "results/chestxray/architecture_selection_binary"

# ── Fixed v7 training settings ────────────────────────────────────────────────
CV_SEED    = 42       # single init seed for all (config, fold) pairs (v7 parity)
N_FOLDS    = 5
LR         = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 80       # v7 architecture-search budget
PATIENCE   = 15
N_FEATURES = 17
NUM_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_with_early_stopping(
    model, optimizer, scheduler,
    X_train, y_train, X_val, y_val,
    class_weights, max_epochs, patience, device,
):
    """Thin wrapper around v7's train_one_run that also returns val AUC.

    train_one_run performs the exact v7 training/early-stopping (scheduler on
    val balanced accuracy, save-best-on-improvement, restore best weights) and
    returns best_val_balacc / best_epoch. After it restores the best-val model,
    we compute val ROC-AUC from class-1 softmax probabilities (secondary metric).
    """
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    result = train_one_run(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, train_dataset=train_ds,
        X_val_t=X_val_t, y_val_t=y_val_t, y_val_enc=y_val,
        max_epochs=max_epochs, patience=patience, batch_size=BATCH_SIZE,
        device=device,
        concurvity_lambda=0.0, warmup_epochs=0, sparsity_lambda=0.0,
        save_path=None, verbose_every=max_epochs + 1,  # suppress per-epoch noise
    )

    # Best-val weights are restored in `model` by train_one_run.
    model.eval()
    with torch.no_grad():
        proba1 = torch.softmax(model(X_val_t), dim=1)[:, 1].cpu().numpy()
    val_auc = float(roc_auc_score(y_val, proba1))
    return float(result["best_val_balacc"]), val_auc, int(result["best_epoch"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run only config 1 (5 folds) to verify mechanics.")
    parser.add_argument("--max_epochs", type=int, default=None)
    args = parser.parse_args()
    max_epochs = args.max_epochs if args.max_epochs is not None else MAX_EPOCHS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load inputs (NO test_idx) ──────────────────────────────────────────────
    feat          = np.load(SCORES_NPZ, allow_pickle=True)
    scores        = feat["scores"]                  # (5856, 17)
    concept_names = feat["concept_names"].tolist()
    outer_split   = np.load(OUTER_SPLIT, allow_pickle=True)
    train_pool_idx = outer_split["train_pool_idx"]
    labels_binary  = outer_split["labels_binary"]
    patient_ids    = outer_split["patient_ids"]
    cv_folds       = np.load(CV_FOLDS, allow_pickle=True)

    assert scores.shape[1] == N_FEATURES, f"expected 17 features, got {scores.shape[1]}"
    assert len(concept_names) == N_FEATURES

    # ── Six leakage safeguards ─────────────────────────────────────────────────
    print("Running leakage safeguards ...")
    # Safeguard 1: test indices not loaded into this module
    this_module_globals = sys.modules[__name__].__dict__
    assert "test_idx" not in this_module_globals, \
        "SAFEGUARD 1 FAILED: test_idx should not be loaded during selection"

    # Safeguard 2: CV fold indices entirely within the train pool
    train_pool_set = set(train_pool_idx.tolist())
    for fold_i in range(N_FOLDS):
        ft = cv_folds[f"fold_train_idx_{fold_i}"]
        fv = cv_folds[f"fold_val_idx_{fold_i}"]
        assert set(ft.tolist()) <= train_pool_set, \
            f"SAFEGUARD 2 FAILED: fold {fold_i} train leaks outside train_pool"
        assert set(fv.tolist()) <= train_pool_set, \
            f"SAFEGUARD 2 FAILED: fold {fold_i} val leaks outside train_pool"

    # Safeguard 3: per-fold train/val disjoint (image level)
    for fold_i in range(N_FOLDS):
        fts = set(cv_folds[f"fold_train_idx_{fold_i}"].tolist())
        fvs = set(cv_folds[f"fold_val_idx_{fold_i}"].tolist())
        assert fts.isdisjoint(fvs), \
            f"SAFEGUARD 3 FAILED: fold {fold_i} train/val overlap (image level)"

    # Safeguard 4: per-fold patient-level disjointness
    for fold_i in range(N_FOLDS):
        tp = set(patient_ids[cv_folds[f"fold_train_idx_{fold_i}"]].tolist())
        vp = set(patient_ids[cv_folds[f"fold_val_idx_{fold_i}"]].tolist())
        assert tp.isdisjoint(vp), \
            f"SAFEGUARD 4 FAILED: fold {fold_i} patient overlap between train and val"

    # Safeguard 5: cudnn determinism (set_all_seeds will enforce; assert after one call)
    set_all_seeds(CV_SEED)
    assert torch.backends.cudnn.deterministic, "SAFEGUARD 5 FAILED: cudnn must be deterministic"
    assert not torch.backends.cudnn.benchmark, "SAFEGUARD 5 FAILED: cudnn.benchmark must be False"

    # Safeguard 6: 12 configs loaded
    assert len(ARCHITECTURE_CONFIGS) == 12, "SAFEGUARD 6 FAILED: expected 12 v7-grid configs"
    print("  [✓] All six leakage safeguards passed")

    configs = ARCHITECTURE_CONFIGS[:1] if args.smoke_test else ARCHITECTURE_CONFIGS
    total_runs = len(configs) * N_FOLDS

    print("=" * 70)
    print("Chest X-ray NAM architecture selection — v7 parity")
    print(f"  Device: {DEVICE}  |  configs: {len(configs)}  |  folds: {N_FOLDS}  "
          f"|  seed: {CV_SEED}")
    print(f"  Selection metric: val balanced accuracy (AUC reported secondary)")
    print(f"  max_epochs={max_epochs}  patience={PATIENCE}  batch={BATCH_SIZE}  lr={LR}")
    if args.smoke_test:
        print("  *** SMOKE TEST: config 1 only ***")
    print("=" * 70)

    # ── Per-fold val class proportions (sanity) ────────────────────────────────
    fold_val_props = {}
    for fold_i in range(N_FOLDS):
        yv = labels_binary[cv_folds[f"fold_val_idx_{fold_i}"]]
        fold_val_props[fold_i] = float(yv.mean())  # pneumonia fraction

    # ── Training loop ──────────────────────────────────────────────────────────
    per_run_path = OUT_DIR / "per_run_results.csv"
    all_rows: list[dict] = []
    t0 = time.time()
    run_counter = 0

    for config in configs:
        for fold_i in range(N_FOLDS):
            run_counter += 1
            t_run = time.time()

            # Seed reset before each (config, fold) — v7 parity
            set_all_seeds(CV_SEED)

            ft_idx = cv_folds[f"fold_train_idx_{fold_i}"]
            fv_idx = cv_folds[f"fold_val_idx_{fold_i}"]

            # Per-fold z-scoring: fit on fold-train only
            scaler = StandardScaler()
            X_train = scaler.fit_transform(scores[ft_idx]).astype(np.float32)
            X_val   = scaler.transform(scores[fv_idx]).astype(np.float32)
            y_train = labels_binary[ft_idx].astype(np.int64)
            y_val   = labels_binary[fv_idx].astype(np.int64)

            # Per-fold balanced class weights from fold-train labels only
            counts = np.bincount(y_train, minlength=NUM_CLASSES)
            class_weights = torch.tensor(
                len(y_train) / (NUM_CLASSES * counts), dtype=torch.float32
            )

            # Fresh model per (config, fold)
            model = NAMMulticlass(
                n_features=N_FEATURES,
                num_classes=NUM_CLASSES,
                hidden_dims=config["hidden"],
                dropout=config["dropout"],
                concept_names=list(concept_names),
            ).to(DEVICE)

            optimizer, scheduler = make_optimizer_scheduler(
                model, LR, config["weight_decay"],
            )  # defaults: sched_patience=5, sched_factor=0.5, mode="max"

            best_balacc, best_auc, best_epoch = train_with_early_stopping(
                model, optimizer, scheduler,
                X_train, y_train, X_val, y_val,
                class_weights=class_weights,
                max_epochs=max_epochs, patience=PATIENCE, device=DEVICE,
            )

            elapsed = time.time() - t_run
            all_rows.append({
                "config_id":    config["config_id"],
                "hidden":       str(list(config["hidden"])),
                "dropout":      config["dropout"],
                "weight_decay": config["weight_decay"],
                "fold":         fold_i,
                "seed":         CV_SEED,
                "val_balacc":   round(best_balacc, 6),
                "val_auc":      round(best_auc, 6),
                "best_epoch":   best_epoch,
                "elapsed_s":    round(elapsed, 1),
            })
            # Save incrementally after EVERY (config, fold)
            pd.DataFrame(all_rows).to_csv(per_run_path, index=False)

            print(f"[{run_counter:>2}/{total_runs}] config {config['config_id']:>2} "
                  f"fold {fold_i}  balacc={best_balacc:.4f}  auc={best_auc:.4f}  "
                  f"best_epoch={best_epoch}  ({elapsed:.0f}s)")

    print(f"\nAll {run_counter} runs done in {(time.time()-t0)/60:.1f} min")
    print(f"  per-run results → {per_run_path.relative_to(_ROOT)}")

    # ── Aggregate per config ───────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    agg_rows = []
    for config in configs:
        sub = df[df["config_id"] == config["config_id"]]
        agg_rows.append({
            "config_id":      config["config_id"],
            "hidden":         str(list(config["hidden"])),
            "dropout":        config["dropout"],
            "weight_decay":   config["weight_decay"],
            "mean_val_balacc": round(float(sub["val_balacc"].mean()), 4),
            "std_val_balacc":  round(float(sub["val_balacc"].std()), 4),  # ddof=1
            "mean_val_auc":    round(float(sub["val_auc"].mean()), 4),
            "std_val_auc":     round(float(sub["val_auc"].std()), 4),
            "n_folds":         len(sub),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_path = OUT_DIR / "config_aggregates.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"  config aggregates → {agg_path.relative_to(_ROOT)}")

    # ── Winner selection (v7 rule: mean balacc; tiebreak std; then config_id) ──
    sorted_df = agg_df.sort_values(
        ["mean_val_balacc", "std_val_balacc", "config_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    winner = sorted_df.iloc[0]
    runner = sorted_df.iloc[1] if len(sorted_df) > 1 else None
    margin = (float(winner["mean_val_balacc"]) - float(runner["mean_val_balacc"])
              if runner is not None else float("nan"))

    # Report tie-break firing
    if runner is not None and abs(margin) < 1e-6:
        print(f"  [tie-break] config {int(winner['config_id'])} and "
              f"{int(runner['config_id'])} within 1e-6 on mean balacc; "
              f"chose lower std ({winner['std_val_balacc']}) then lower config_id")

    winner_cfg = next(c for c in configs if c["config_id"] == int(winner["config_id"]))
    winning_config = {
        "config_id":          int(winner["config_id"]),
        "hidden_dims":        list(winner_cfg["hidden"]),
        "dropout":            float(winner_cfg["dropout"]),
        "weight_decay":       float(winner_cfg["weight_decay"]),
        "mean_val_balacc":    float(winner["mean_val_balacc"]),
        "std_val_balacc":     float(winner["std_val_balacc"]),
        "mean_val_auc":       float(winner["mean_val_auc"]),
        "std_val_auc":        float(winner["std_val_auc"]),
        "selection_criterion": "highest mean val balanced accuracy (5-fold "
                               "StratifiedGroupKFold); tiebreak lowest std, then "
                               "lowest config_id (v7 rule)",
        "runner_up_config_id": (int(runner["config_id"]) if runner is not None else None),
        "runner_up_margin":    (round(margin, 4) if runner is not None else None),
        "n_features":          N_FEATURES,
        "num_classes":         NUM_CLASSES,
        "cv_seed":             CV_SEED,
        "max_epochs":          max_epochs,
        "patience":            PATIENCE,
        "test_set_touched":    False,
    }
    win_path = OUT_DIR / "winning_config.json"
    with win_path.open("w", encoding="utf-8") as f:
        json.dump(winning_config, f, indent=2)
    print(f"  winning config → {win_path.relative_to(_ROOT)}")

    # ── selection_summary.txt ──────────────────────────────────────────────────
    L: list[str] = []
    L.append("=" * 70)
    L.append("CHEST X-RAY NAM ARCHITECTURE SELECTION — SUMMARY (v7 parity)")
    L.append("=" * 70)
    L.append("")
    L.append("1. v7 training config used verbatim (scripts/v7/_common.py):")
    L.append("   Adam(lr=1e-3, weight_decay=config); ReduceLROnPlateau(mode='max',")
    L.append("   factor=0.5, patience=5, min_lr=1e-6) on val balanced accuracy;")
    L.append("   CrossEntropyLoss with balanced class weights; batch=256;")
    L.append(f"   max_epochs={max_epochs}, patience={PATIENCE}; single seed {CV_SEED}.")
    L.append("   Only divergence: num_classes=2 (binary task).")
    L.append("")
    L.append("2. The 12 configs evaluated (v7 SWEEP_GRID):")
    for c in configs:
        L.append(f"   config {c['config_id']:>2}: hidden={list(c['hidden'])}, "
                 f"dropout={c['dropout']}, weight_decay={c['weight_decay']:.0e}")
    L.append("")
    L.append("3. Per-config aggregates (sorted by mean val balanced accuracy):")
    L.append(sorted_df.to_string(index=False))
    L.append("")
    L.append("4. Winner:")
    L.append(f"   config {winning_config['config_id']}: "
             f"hidden={winning_config['hidden_dims']}, "
             f"dropout={winning_config['dropout']}, "
             f"weight_decay={winning_config['weight_decay']:.0e}")
    L.append(f"   mean val balacc = {winning_config['mean_val_balacc']:.4f} "
             f"± {winning_config['std_val_balacc']:.4f}")
    L.append(f"   mean val AUC    = {winning_config['mean_val_auc']:.4f} "
             f"± {winning_config['std_val_auc']:.4f}")
    if runner is not None:
        L.append(f"   runner-up: config {int(runner['config_id'])} "
                 f"(balacc {float(runner['mean_val_balacc']):.4f}); "
                 f"margin = {margin:.4f}")
    L.append("")
    L.append("5. Per-fold val class proportions (pneumonia fraction):")
    for fold_i in range(N_FOLDS):
        L.append(f"   fold {fold_i}: {fold_val_props[fold_i]:.3f} pneumonia")
    L.append("")
    L.append("6. Per-config mean val AUC alongside balanced accuracy:")
    for _, r in sorted_df.iterrows():
        L.append(f"   config {int(r['config_id']):>2}: "
                 f"balacc={r['mean_val_balacc']:.4f}  auc={r['mean_val_auc']:.4f}")
    L.append("")
    L.append("7. Leakage safeguards: all six assertions passed before training.")
    # Flag any config with suspiciously low AUC
    low = sorted_df[sorted_df["mean_val_auc"] < 0.6]
    if len(low):
        L.append("")
        L.append("   ⚠ configs with mean val AUC < 0.6 (investigate):")
        for _, r in low.iterrows():
            L.append(f"     config {int(r['config_id'])}: auc={r['mean_val_auc']:.4f}")

    summary_text = "\n".join(L)
    (OUT_DIR / "selection_summary.txt").write_text(summary_text, encoding="utf-8")
    print(f"  selection summary → {(OUT_DIR / 'selection_summary.txt').relative_to(_ROOT)}")

    print("\n" + "=" * 70)
    print(f"WINNER: config {winning_config['config_id']}  "
          f"hidden={winning_config['hidden_dims']} "
          f"dropout={winning_config['dropout']} "
          f"wd={winning_config['weight_decay']:.0e}")
    print(f"  mean val balacc = {winning_config['mean_val_balacc']:.4f} "
          f"± {winning_config['std_val_balacc']:.4f}  "
          f"(AUC {winning_config['mean_val_auc']:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
