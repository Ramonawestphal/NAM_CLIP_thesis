"""
v7-parity three-way NAM architecture selection (PRIMARY chest X-ray task).

Three-way classification (normal / bacteria / virus). Parallel in every way to
the binary selection (scripts/chestxray/select_architecture_binary.py) except the
label dimension: num_classes=3 and the secondary metric is macro-OvR AUC.

12-config v7 grid x 5 pre-built StratifiedGroupKFold folds x 1 seed (42) = 60
runs. Selection metric: mean val balanced accuracy across folds (v7 rule). The
per-run training/early-stopping is delegated to scripts/v7/_common.train_one_run
(Adam lr=1e-3, batch=256, ReduceLROnPlateau(mode="max", factor=0.5, patience=5,
min_lr=1e-6), early stop on val balanced accuracy, max_epochs=80/patience=15,
CrossEntropyLoss with balanced class weights). Only divergence from v7:
num_classes=3.

The held-out test set is NEVER loaded here. Seven leakage safeguards are asserted
before any training.

Outputs (under results/chestxray/architecture_selection/):
    label_mapping.json
    per_run_results.csv    one row per (config, fold), saved incrementally
    config_aggregates.csv
    winning_config.json
    selection_summary.txt

Run from project root:
    python scripts/chestxray/select_architecture.py
    python scripts/chestxray/select_architecture.py --smoke_test
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

from src.models.nam_multiclass import NAMMulticlass
from scripts.chestxray._common_imports import (
    set_all_seeds, make_optimizer_scheduler, train_one_run,
)
from scripts.chestxray.architecture_configs import ARCHITECTURE_CONFIGS

# ── Paths ─────────────────────────────────────────────────────────────────────
SCORES_NPZ  = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v4.npz"
OUTER_SPLIT = _ROOT / "data/splits/chestxray_outer_split.npz"
CV_FOLDS    = _ROOT / "data/splits/chestxray_cv_folds.npz"
OUT_DIR     = _ROOT / "results/chestxray/architecture_selection"
BINARY_WINNER = _ROOT / "results/chestxray/architecture_selection_binary/winning_config.json"

# ── Fixed v7 training settings ────────────────────────────────────────────────
CV_SEED     = 42
N_FOLDS     = 5
LR          = 1e-3
BATCH_SIZE  = 256
MAX_EPOCHS  = 80
PATIENCE    = 15
N_FEATURES  = 17
NUM_CLASSES = 3

SUBTYPE_TO_INT = {"normal": 0, "bacteria": 1, "virus": 2}
INT_TO_SUBTYPE = {v: k for k, v in SUBTYPE_TO_INT.items()}
CLASS_ORDER    = [0, 1, 2]   # for roc_auc labels / per-class iteration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_with_early_stopping(
    model, optimizer, scheduler,
    X_train, y_train, X_val, y_val,
    class_weights, max_epochs, patience, device,
):
    """v7 train_one_run wrapper; returns three-way val metrics.

    train_one_run runs the exact v7 training/early-stopping (scheduler on val
    balanced accuracy, save-best, restore-best). After the best-val model is
    restored we compute the secondary metrics on the val set:
      - macro-OvR AUC (needs full 3-class probability matrix)
      - per-class accuracy (recall) for normal/bacteria/virus

    Returns (best_val_balacc, macro_auc, per_class_acc dict, best_epoch).
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
        save_path=None, verbose_every=max_epochs + 1,
    )

    model.eval()
    with torch.no_grad():
        logits = model(X_val_t)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()   # (N, 3)
        preds  = logits.argmax(dim=1).cpu().numpy()
    macro_auc = float(roc_auc_score(
        y_val, proba, multi_class="ovr", average="macro", labels=CLASS_ORDER
    ))
    per_class_acc = {}
    for c in CLASS_ORDER:
        mask = (y_val == c)
        per_class_acc[INT_TO_SUBTYPE[c]] = (
            float((preds[mask] == c).mean()) if mask.sum() else float("nan")
        )
    return float(result["best_val_balacc"]), macro_auc, per_class_acc, int(result["best_epoch"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run only config 1 (5 folds) to verify mechanics.")
    parser.add_argument("--max_epochs", type=int, default=None)
    args = parser.parse_args()
    max_epochs = args.max_epochs if args.max_epochs is not None else MAX_EPOCHS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Persist label mapping (single source of truth for downstream code)
    with (OUT_DIR / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(SUBTYPE_TO_INT, f, indent=2)

    # ── Load inputs (NO test_idx) ──────────────────────────────────────────────
    feat          = np.load(SCORES_NPZ, allow_pickle=True)
    scores        = feat["scores"]
    concept_names = feat["concept_names"].tolist()
    outer_split   = np.load(OUTER_SPLIT, allow_pickle=True)
    train_pool_idx = outer_split["train_pool_idx"]
    labels_subtype = outer_split["labels_subtype"]
    patient_ids    = outer_split["patient_ids"]
    cv_folds       = np.load(CV_FOLDS, allow_pickle=True)

    labels_threeway = np.array(
        [SUBTYPE_TO_INT[s] for s in labels_subtype], dtype=np.int64
    )

    assert scores.shape[1] == N_FEATURES, f"expected 17 features, got {scores.shape[1]}"
    assert len(concept_names) == N_FEATURES

    # ── Seven leakage safeguards ───────────────────────────────────────────────
    print("Running leakage safeguards ...")
    this_module_globals = sys.modules[__name__].__dict__
    assert "test_idx" not in this_module_globals, \
        "SAFEGUARD 1 FAILED: test_idx should not be loaded during selection"

    train_pool_set = set(train_pool_idx.tolist())
    for fold_i in range(N_FOLDS):
        ft = cv_folds[f"fold_train_idx_{fold_i}"]
        fv = cv_folds[f"fold_val_idx_{fold_i}"]
        assert set(ft.tolist()) <= train_pool_set, \
            f"SAFEGUARD 2 FAILED: fold {fold_i} train leaks outside train_pool"
        assert set(fv.tolist()) <= train_pool_set, \
            f"SAFEGUARD 2 FAILED: fold {fold_i} val leaks outside train_pool"

    for fold_i in range(N_FOLDS):
        fts = set(cv_folds[f"fold_train_idx_{fold_i}"].tolist())
        fvs = set(cv_folds[f"fold_val_idx_{fold_i}"].tolist())
        assert fts.isdisjoint(fvs), \
            f"SAFEGUARD 3 FAILED: fold {fold_i} train/val overlap (image level)"

    for fold_i in range(N_FOLDS):
        tp = set(patient_ids[cv_folds[f"fold_train_idx_{fold_i}"]].tolist())
        vp = set(patient_ids[cv_folds[f"fold_val_idx_{fold_i}"]].tolist())
        assert tp.isdisjoint(vp), \
            f"SAFEGUARD 4 FAILED: fold {fold_i} patient overlap between train and val"

    set_all_seeds(CV_SEED)
    assert torch.backends.cudnn.deterministic, "SAFEGUARD 5 FAILED: cudnn must be deterministic"
    assert not torch.backends.cudnn.benchmark, "SAFEGUARD 5 FAILED: cudnn.benchmark must be False"

    assert len(ARCHITECTURE_CONFIGS) == 12, "SAFEGUARD 6 FAILED: expected 12 v7-grid configs"

    # Safeguard 7 (new for three-way): label array has exactly {0,1,2}, each non-trivial
    assert set(np.unique(labels_threeway)) == {0, 1, 2}, \
        "SAFEGUARD 7 FAILED: labels_threeway must contain exactly {0, 1, 2}"
    assert (labels_threeway == 0).sum() > 100, "SAFEGUARD 7 FAILED: normal class too small"
    assert (labels_threeway == 1).sum() > 100, "SAFEGUARD 7 FAILED: bacteria class too small"
    assert (labels_threeway == 2).sum() > 100, "SAFEGUARD 7 FAILED: virus class too small"
    print("  [✓] All seven leakage safeguards passed")

    configs = ARCHITECTURE_CONFIGS[:1] if args.smoke_test else ARCHITECTURE_CONFIGS
    total_runs = len(configs) * N_FOLDS

    print("=" * 70)
    print("Chest X-ray NAM architecture selection — THREE-WAY (v7 parity)")
    print(f"  Device: {DEVICE}  |  configs: {len(configs)}  |  folds: {N_FOLDS}  "
          f"|  seed: {CV_SEED}")
    print(f"  Selection metric: val balanced accuracy (macro-OvR AUC secondary)")
    print(f"  max_epochs={max_epochs}  patience={PATIENCE}  batch={BATCH_SIZE}  lr={LR}")
    if args.smoke_test:
        print("  *** SMOKE TEST: config 1 only ***")
    print("=" * 70)

    # Per-fold val three-way class proportions (sanity)
    fold_val_props = {}
    for fold_i in range(N_FOLDS):
        sub = labels_subtype[cv_folds[f"fold_val_idx_{fold_i}"]]
        fold_val_props[fold_i] = {c: float((sub == c).mean())
                                   for c in ["normal", "bacteria", "virus"]}

    # ── Training loop ──────────────────────────────────────────────────────────
    per_run_path = OUT_DIR / "per_run_results.csv"
    all_rows: list[dict] = []
    t0 = time.time()
    run_counter = 0

    for config in configs:
        for fold_i in range(N_FOLDS):
            run_counter += 1
            t_run = time.time()
            set_all_seeds(CV_SEED)   # v7 parity: same init seed each (config,fold)

            ft_idx = cv_folds[f"fold_train_idx_{fold_i}"]
            fv_idx = cv_folds[f"fold_val_idx_{fold_i}"]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(scores[ft_idx]).astype(np.float32)
            X_val   = scaler.transform(scores[fv_idx]).astype(np.float32)
            y_train = labels_threeway[ft_idx]
            y_val   = labels_threeway[fv_idx]

            counts = np.bincount(y_train, minlength=NUM_CLASSES)
            class_weights = torch.tensor(
                len(y_train) / (NUM_CLASSES * counts), dtype=torch.float32
            )

            model = NAMMulticlass(
                n_features=N_FEATURES,
                num_classes=NUM_CLASSES,
                hidden_dims=config["hidden"],
                dropout=config["dropout"],
                concept_names=list(concept_names),
            ).to(DEVICE)
            optimizer, scheduler = make_optimizer_scheduler(
                model, LR, config["weight_decay"],
            )

            best_balacc, macro_auc, per_class_acc, best_epoch = train_with_early_stopping(
                model, optimizer, scheduler,
                X_train, y_train, X_val, y_val,
                class_weights=class_weights,
                max_epochs=max_epochs, patience=PATIENCE, device=DEVICE,
            )

            elapsed = time.time() - t_run
            all_rows.append({
                "config_id":      config["config_id"],
                "hidden":         str(list(config["hidden"])),
                "dropout":        config["dropout"],
                "weight_decay":   config["weight_decay"],
                "fold":           fold_i,
                "seed":           CV_SEED,
                "val_balacc":     round(best_balacc, 6),
                "val_macro_auc_ovr": round(macro_auc, 6),
                "val_acc_normal":   round(per_class_acc["normal"], 6),
                "val_acc_bacteria": round(per_class_acc["bacteria"], 6),
                "val_acc_virus":    round(per_class_acc["virus"], 6),
                "best_epoch":     best_epoch,
                "elapsed_s":      round(elapsed, 1),
            })
            pd.DataFrame(all_rows).to_csv(per_run_path, index=False)

            print(f"[{run_counter:>2}/{total_runs}] config {config['config_id']:>2} "
                  f"fold {fold_i}  balacc={best_balacc:.4f}  macroAUC={macro_auc:.4f}  "
                  f"acc(N/B/V)={per_class_acc['normal']:.2f}/"
                  f"{per_class_acc['bacteria']:.2f}/{per_class_acc['virus']:.2f}  "
                  f"({elapsed:.0f}s)")

    print(f"\nAll {run_counter} runs done in {(time.time()-t0)/60:.1f} min")

    # ── Aggregate per config ───────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    metric_cols = ["val_balacc", "val_macro_auc_ovr",
                   "val_acc_normal", "val_acc_bacteria", "val_acc_virus"]
    agg_rows = []
    for config in configs:
        sub = df[df["config_id"] == config["config_id"]]
        row = {
            "config_id":    config["config_id"],
            "hidden":       str(list(config["hidden"])),
            "dropout":      config["dropout"],
            "weight_decay": config["weight_decay"],
            "n_folds":      len(sub),
        }
        for m in metric_cols:
            row[f"mean_{m}"] = round(float(sub[m].mean()), 4)
            row[f"std_{m}"]  = round(float(sub[m].std()), 4)
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / "config_aggregates.csv", index=False)

    # ── Winner selection (v7 rule: mean balacc; tiebreak std; then config_id) ──
    sorted_df = agg_df.sort_values(
        ["mean_val_balacc", "std_val_balacc", "config_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    winner = sorted_df.iloc[0]
    runner = sorted_df.iloc[1] if len(sorted_df) > 1 else None
    margin = (float(winner["mean_val_balacc"]) - float(runner["mean_val_balacc"])
              if runner is not None else float("nan"))
    if runner is not None and abs(margin) < 1e-6:
        print(f"  [tie-break] config {int(winner['config_id'])} vs "
              f"{int(runner['config_id'])} within 1e-6 on mean balacc; "
              f"chose lower std then lower config_id")

    winner_cfg = next(c for c in configs if c["config_id"] == int(winner["config_id"]))
    winning_config = {
        "config_id":          int(winner["config_id"]),
        "hidden_dims":        list(winner_cfg["hidden"]),
        "dropout":            float(winner_cfg["dropout"]),
        "weight_decay":       float(winner_cfg["weight_decay"]),
        "mean_val_balacc":    float(winner["mean_val_balacc"]),
        "std_val_balacc":     float(winner["std_val_balacc"]),
        "mean_val_macro_auc_ovr": float(winner["mean_val_macro_auc_ovr"]),
        "std_val_macro_auc_ovr":  float(winner["std_val_macro_auc_ovr"]),
        "mean_val_acc_normal":   float(winner["mean_val_acc_normal"]),
        "mean_val_acc_bacteria": float(winner["mean_val_acc_bacteria"]),
        "mean_val_acc_virus":    float(winner["mean_val_acc_virus"]),
        "selection_criterion": "highest mean val balanced accuracy (5-fold "
                               "StratifiedGroupKFold); tiebreak lowest std, then "
                               "lowest config_id (v7 rule)",
        "runner_up_config_id": (int(runner["config_id"]) if runner is not None else None),
        "runner_up_margin":    (round(margin, 4) if runner is not None else None),
        "task":               "three_way",
        "label_mapping":      SUBTYPE_TO_INT,
        "n_features":         N_FEATURES,
        "num_classes":        NUM_CLASSES,
        "cv_seed":            CV_SEED,
        "max_epochs":         max_epochs,
        "patience":           PATIENCE,
        "test_set_touched":   False,
    }
    with (OUT_DIR / "winning_config.json").open("w", encoding="utf-8") as f:
        json.dump(winning_config, f, indent=2)

    # ── Comparison to binary task ──────────────────────────────────────────────
    binary_balacc, binary_std = 0.9325, 0.0061  # from the task context (fallback)
    if BINARY_WINNER.exists():
        try:
            bw = json.loads(BINARY_WINNER.read_text(encoding="utf-8"))
            binary_balacc = float(bw.get("mean_val_balacc", binary_balacc))
            binary_std    = float(bw.get("std_val_balacc", binary_std))
        except Exception:
            pass

    # ── selection_summary.txt ──────────────────────────────────────────────────
    L: list[str] = []
    L.append("=" * 70)
    L.append("CHEST X-RAY NAM ARCHITECTURE SELECTION — THREE-WAY (v7 parity)")
    L.append("=" * 70)
    L.append("")
    # Pre-flight stratification (Step 0b) — include if present
    preflight = OUT_DIR / "preflight_stratification.txt"
    if preflight.exists():
        L.append("0. Pre-flight three-way stratification check (Step 0b):")
        for ln in preflight.read_text(encoding="utf-8").splitlines():
            L.append("   " + ln)
        L.append("")
    L.append("1. v7 training config used verbatim; only divergence: num_classes=3.")
    L.append("   Adam(lr=1e-3, wd=config); ReduceLROnPlateau(mode='max', factor=0.5,")
    L.append("   patience=5, min_lr=1e-6) on val balacc; CrossEntropyLoss + balanced")
    L.append(f"   class weights; batch=256; max_epochs={max_epochs}, patience={PATIENCE}; seed {CV_SEED}.")
    L.append("")
    L.append("2. 12 configs (v7 SWEEP_GRID):")
    for c in configs:
        L.append(f"   config {c['config_id']:>2}: hidden={list(c['hidden'])}, "
                 f"dropout={c['dropout']}, weight_decay={c['weight_decay']:.0e}")
    L.append("")
    L.append("3. Per-config aggregates (sorted by mean val balanced accuracy):")
    show_cols = ["config_id", "hidden", "dropout", "weight_decay",
                 "mean_val_balacc", "std_val_balacc", "mean_val_macro_auc_ovr"]
    L.append(sorted_df[show_cols].to_string(index=False))
    L.append("")
    L.append("4. Winner:")
    L.append(f"   config {winning_config['config_id']}: hidden={winning_config['hidden_dims']}, "
             f"dropout={winning_config['dropout']}, weight_decay={winning_config['weight_decay']:.0e}")
    L.append(f"   mean val balacc        = {winning_config['mean_val_balacc']:.4f} "
             f"± {winning_config['std_val_balacc']:.4f}")
    L.append(f"   mean val macro-OvR AUC = {winning_config['mean_val_macro_auc_ovr']:.4f} "
             f"± {winning_config['std_val_macro_auc_ovr']:.4f}")
    if runner is not None:
        L.append(f"   runner-up: config {int(runner['config_id'])} "
                 f"(balacc {float(runner['mean_val_balacc']):.4f}); margin = {margin:.4f}")
    L.append("")
    L.append("5. Per-class val accuracy of the WINNING config (mean across folds):")
    L.append(f"   Normal:   {winning_config['mean_val_acc_normal']:.4f}")
    L.append(f"   Bacteria: {winning_config['mean_val_acc_bacteria']:.4f}")
    L.append(f"   Virus:    {winning_config['mean_val_acc_virus']:.4f}")
    L.append("   (Is the model balanced across classes, or dominating on bacteria")
    L.append("    at the expense of virus? See per-class figures above.)")
    L.append("")
    L.append("6. Per-fold val three-way class proportions:")
    for fold_i in range(N_FOLDS):
        p = fold_val_props[fold_i]
        L.append(f"   fold {fold_i}: normal={p['normal']:.3f} "
                 f"bacteria={p['bacteria']:.3f} virus={p['virus']:.3f}")
    L.append("")
    L.append("7. Comparison to binary task (now a sensitivity analysis):")
    L.append(f"   Binary winning val balacc:    {binary_balacc:.4f} ± {binary_std:.4f}  (saturated)")
    L.append(f"   Three-way winning val balacc: {winning_config['mean_val_balacc']:.4f} "
             f"± {winning_config['std_val_balacc']:.4f}")
    delta = binary_balacc - winning_config['mean_val_balacc']
    L.append(f"   Three-way is {delta*100:.1f} pp lower → non-saturated; architectural")
    L.append(f"   choices matter more for the downstream sparsity/concurvity work.")
    L.append("")
    L.append("8. Leakage safeguards: all seven assertions passed before training.")

    (OUT_DIR / "selection_summary.txt").write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"WINNER: config {winning_config['config_id']}  "
          f"hidden={winning_config['hidden_dims']} "
          f"dropout={winning_config['dropout']} wd={winning_config['weight_decay']:.0e}")
    print(f"  mean val balacc = {winning_config['mean_val_balacc']:.4f} "
          f"± {winning_config['std_val_balacc']:.4f}  "
          f"(macro-OvR AUC {winning_config['mean_val_macro_auc_ovr']:.4f})")
    print(f"  per-class val acc: N={winning_config['mean_val_acc_normal']:.3f} "
          f"B={winning_config['mean_val_acc_bacteria']:.3f} "
          f"V={winning_config['mean_val_acc_virus']:.3f}")
    # Saturation guard per the task's calibration note
    wb = winning_config['mean_val_balacc']
    if wb > 0.92:
        print(f"  ⚠ val balacc {wb:.4f} > 0.92 — three-way may ALSO be saturated; escalate.")
    elif wb < 0.60:
        print(f"  ⚠ val balacc {wb:.4f} < 0.60 — likely a class-weight/label bug; investigate.")
    print("=" * 70)


if __name__ == "__main__":
    main()
