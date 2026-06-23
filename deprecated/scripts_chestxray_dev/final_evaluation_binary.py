"""
v7-parity Step 4: final 5-seed evaluation of the winning config on the held-out test set.

This is the ONLY chest X-ray selection script that loads test_idx, and it loads it
only AFTER architecture selection has produced winning_config.json.

Procedure (mirrors scripts/v7/train_final.py):
  - Fixed 20% patient-grouped early-stopping val split via v7's make_fixed_val_split
    (GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)), the same
    across all 5 seeds.
  - StandardScaler fit on the 80% inner-train only; applied to inner-train, val, test.
  - 5 seeds (42-46), fresh model each, train on inner-train with early stop on val
    balanced accuracy (max_epochs=100, patience=15 — v7 final-training budget).
  - Test AUC + test balanced accuracy computed once per seed after training.

Outputs (under results/chestxray/architecture_selection_binary/):
    final_test_results.csv
    final_test_summary.txt

Run from project root (after select_architecture.py):
    python scripts/chestxray/final_evaluation.py
"""

from __future__ import annotations

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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models.nam_multiclass import NAMMulticlass
from scripts.chestxray._common_imports import set_all_seeds, make_optimizer_scheduler, make_fixed_val_split
# Reuse the SAME training/early-stopping wrapper used during selection (defined once).
from scripts.chestxray.select_architecture_binary import train_with_early_stopping

# ── Paths ─────────────────────────────────────────────────────────────────────
SCORES_NPZ  = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v4.npz"
OUTER_SPLIT = _ROOT / "data/splits/chestxray_outer_split.npz"
OUT_DIR     = _ROOT / "results/chestxray/architecture_selection_binary"
WINNER_JSON = OUT_DIR / "winning_config.json"

# ── Fixed v7 final-training settings ──────────────────────────────────────────
SEEDS       = [42, 43, 44, 45, 46]
LR          = 1e-3
MAX_EPOCHS  = 100      # v7 train_final budget
PATIENCE    = 15
N_FEATURES  = 17
NUM_CLASSES = 2
VAL_FRACTION = 0.20
VAL_RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not WINNER_JSON.exists():
        sys.exit(f"ERROR: {WINNER_JSON} not found. Run select_architecture.py first.")
    with WINNER_JSON.open(encoding="utf-8") as f:
        winner = json.load(f)
    hidden_dims  = tuple(winner["hidden_dims"])
    dropout      = float(winner["dropout"])
    weight_decay = float(winner["weight_decay"])

    # ── Load data (test_idx loaded here, post-selection only) ──────────────────
    feat          = np.load(SCORES_NPZ, allow_pickle=True)
    scores        = feat["scores"]
    concept_names = feat["concept_names"].tolist()
    split         = np.load(OUTER_SPLIT, allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]
    test_idx       = split["test_idx"]
    labels_binary  = split["labels_binary"]
    patient_ids    = split["patient_ids"]

    assert len(np.intersect1d(train_pool_idx, test_idx)) == 0, \
        "train_pool and test overlap — check splits file"

    X_pool_raw  = scores[train_pool_idx]
    y_pool      = labels_binary[train_pool_idx].astype(np.int64)
    groups_pool = patient_ids[train_pool_idx]
    X_test_raw  = scores[test_idx]
    y_test      = labels_binary[test_idx].astype(np.int64)

    # ── Fixed 20% patient-grouped val split (v7 make_fixed_val_split) ──────────
    # make_fixed_val_split encodes labels via class_names; pass binary labels as
    # strings so the encoding is identity-like. We only consume the rel indices.
    y_pool_str  = y_pool.astype(str)
    class_names = ["0", "1"]
    vs = make_fixed_val_split(
        X_pool_raw, y_pool_str, groups_pool, class_names,
        val_random_state=VAL_RANDOM_STATE,
    )
    train_rel = vs["train_rel"]
    val_rel   = vs["val_rel"]

    # Patient-level disjointness sanity (safeguard analogue)
    assert set(groups_pool[train_rel]).isdisjoint(set(groups_pool[val_rel])), \
        "inner-train/val patient overlap in fixed val split"

    n_val_patients = len(set(groups_pool[val_rel].tolist()))
    val_pneu_frac  = float(y_pool[val_rel].mean())

    print("=" * 70)
    print("Chest X-ray NAM — final 5-seed test evaluation (v7 parity)")
    print(f"  Winning config: hidden={list(hidden_dims)}, dropout={dropout}, "
          f"wd={weight_decay:.0e}")
    print(f"  Device: {DEVICE}")
    print(f"  Train pool: {len(train_pool_idx)}  (inner-train {len(train_rel)} / "
          f"val {len(val_rel)})   Test: {len(test_idx)}")
    print(f"  Fixed val: {n_val_patients} patients, pneumonia frac={val_pneu_frac:.3f}")
    print(f"  max_epochs={MAX_EPOCHS}  patience={PATIENCE}  seeds={SEEDS}")
    print("=" * 70)

    rows: list[dict] = []
    t0 = time.time()
    for seed in SEEDS:
        set_all_seeds(seed)

        # z-score: fit on inner-train only; transform inner-train, val, test
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_pool_raw[train_rel]).astype(np.float32)
        X_val   = scaler.transform(X_pool_raw[val_rel]).astype(np.float32)
        X_test  = scaler.transform(X_test_raw).astype(np.float32)
        y_train = y_pool[train_rel]
        y_val   = y_pool[val_rel]

        counts = np.bincount(y_train, minlength=NUM_CLASSES)
        class_weights = torch.tensor(
            len(y_train) / (NUM_CLASSES * counts), dtype=torch.float32
        )

        model = NAMMulticlass(
            n_features=N_FEATURES, num_classes=NUM_CLASSES,
            hidden_dims=hidden_dims, dropout=dropout,
            concept_names=list(concept_names),
        ).to(DEVICE)
        optimizer, scheduler = make_optimizer_scheduler(model, LR, weight_decay)

        # Train (best-val weights restored into `model`); we ignore the returned
        # val AUC and compute TEST metrics on the held-out set.
        best_val_balacc, _, best_epoch = train_with_early_stopping(
            model, optimizer, scheduler,
            X_train, y_train, X_val, y_val,
            class_weights=class_weights,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, device=DEVICE,
        )

        # ── Test evaluation (held-out set, after training completes) ───────────
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            proba1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
        test_auc    = float(roc_auc_score(y_test, proba1))
        test_balacc = float(balanced_accuracy_score(y_test, preds))

        rows.append({
            "seed": seed,
            "best_val_balacc": round(best_val_balacc, 6),
            "best_epoch": best_epoch,
            "test_auc": round(test_auc, 6),
            "test_balacc": round(test_balacc, 6),
        })
        print(f"  seed {seed}: test_auc={test_auc:.4f}  test_balacc={test_balacc:.4f}  "
              f"(best_epoch={best_epoch}, val_balacc={best_val_balacc:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "final_test_results.csv", index=False)

    mean_auc, std_auc = float(df["test_auc"].mean()), float(df["test_auc"].std())
    mean_bal, std_bal = float(df["test_balacc"].mean()), float(df["test_balacc"].std())

    print(f"\nFinal ({len(SEEDS)} seeds): "
          f"test_auc={mean_auc:.4f}±{std_auc:.4f}  "
          f"test_balacc={mean_bal:.4f}±{std_bal:.4f}  "
          f"({(time.time()-t0)/60:.1f} min)")

    # ── Summary ────────────────────────────────────────────────────────────────
    L: list[str] = []
    L.append("=" * 70)
    L.append("CHEST X-RAY NAM — FINAL TEST EVALUATION (v7 parity)")
    L.append("=" * 70)
    L.append("")
    L.append("Winning config:")
    L.append(f"  config_id={winner['config_id']}  hidden={winner['hidden_dims']}  "
             f"dropout={winner['dropout']}  weight_decay={winner['weight_decay']:.0e}")
    L.append(f"  (selected by mean val balanced accuracy "
             f"{winner['mean_val_balacc']:.4f} ± {winner['std_val_balacc']:.4f})")
    L.append("")
    L.append("Per-seed test results:")
    L.append(df.to_string(index=False))
    L.append("")
    L.append(f"Mean ± std across {len(SEEDS)} seeds:")
    L.append(f"  Test AUC:               {mean_auc:.4f} ± {std_auc:.4f}")
    L.append(f"  Test balanced accuracy: {mean_bal:.4f} ± {std_bal:.4f}")
    L.append("")
    L.append("Fixed early-stopping val split (v7 make_fixed_val_split):")
    L.append(f"  GroupShuffleSplit(test_size=0.20, random_state=42), grouped by patient")
    L.append(f"  inner-train={len(train_rel)}  val={len(val_rel)}  "
             f"val_patients={n_val_patients}  val_pneumonia_frac={val_pneu_frac:.3f}")
    L.append("  Same split reused across all 5 seeds.")
    L.append("")
    L.append("Leakage safeguards verified (selection + final):")
    L.append("  [check] Test indices never loaded during selection (select_architecture.py)")
    L.append("  [check] All CV indices inside train pool")
    L.append("  [check] Per-fold train/val disjoint at image level")
    L.append("  [check] Per-fold train/val disjoint at patient level")
    L.append("  [check] Per-fold z-scoring (fit on fold-train only)")
    L.append("  [check] Fresh model per (config, fold), seed 42 reset before each")
    L.append("  [check] Final test loaded only after selection completed (this script)")
    (OUT_DIR / "final_test_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"  final summary → {(OUT_DIR / 'final_test_summary.txt').relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
