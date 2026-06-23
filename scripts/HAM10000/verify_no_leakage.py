"""
Leakage verification for the v7 corrected pipeline (Part 4).

Runs after STEP 1 (architecture_search_cv.py) and verifies:
  1. No test_idx index appears in any CV fold's train or val indices.
  2. winner.json was selected by mean CV val_balacc (not by any test metric).
  3. All 5 GroupKFold folds have non-overlapping lesion_ids.
  4. fold_indices.json is self-consistent (union of all fold train+val = train_idx).

Usage (from project root):
    python scripts/HAM10000/verify_no_leakage.py
"""

from __future__ import annotations

import json
import os
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

CV_DIR     = "results/HAM10000/architecture_search_cv"
SPLITS_PATH = "data/splits/train_test_lesion_split.npz"
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"


def main() -> None:
    print("=" * 60)
    print("Leakage verification — v7 pipeline (Part 4)")
    print("=" * 60)

    all_ok = True

    # ── Load ground-truth splits ───────────────────────────────────────────────
    split     = np.load(SPLITS_PATH)
    train_idx = set(split["train_idx"].tolist())
    test_idx  = set(split["test_idx"].tolist())

    # Load lesion_ids for fold non-overlap check
    feat       = np.load(FEATURES_PATH, allow_pickle=True)
    lesion_ids = feat["lesion_ids"]

    # ── Load fold_indices.json ─────────────────────────────────────────────────
    fold_path = os.path.join(CV_DIR, "fold_indices.json")
    if not os.path.exists(fold_path):
        print(f"  [FAIL] fold_indices.json not found at {fold_path}")
        print("         Run architecture_search_cv.py first.")
        sys.exit(1)

    with open(fold_path) as f:
        fold_data = json.load(f)

    sentinel_test = set(fold_data["test_idx_excluded"])
    folds         = fold_data["folds"]

    # ── Check 1: sentinel matches splits file ──────────────────────────────────
    if sentinel_test == test_idx:
        print("  [OK] fold_indices.json test_idx_excluded matches splits file")
    else:
        print("  [FAIL] test_idx_excluded in fold_indices.json does NOT match splits file")
        all_ok = False

    # ── Check 2: no test index in any fold train or val ────────────────────────
    for fold in folds:
        fi = fold["fold"]
        tr = set(fold["train_abs_indices"])
        va = set(fold["val_abs_indices"])
        tr_test_overlap = tr & test_idx
        va_test_overlap = va & test_idx
        if tr_test_overlap:
            print(f"  [FAIL] Fold {fi}: {len(tr_test_overlap)} test indices found in TRAIN")
            all_ok = False
        elif va_test_overlap:
            print(f"  [FAIL] Fold {fi}: {len(va_test_overlap)} test indices found in VAL")
            all_ok = False
        else:
            print(f"  [OK] Fold {fi}: no test indices in train or val")

    # ── Check 3: all folds have non-overlapping lesion_ids ────────────────────
    for fold in folds:
        fi         = fold["fold"]
        tr_idx     = np.array(fold["train_abs_indices"])
        va_idx     = np.array(fold["val_abs_indices"])
        tr_lesions = set(lesion_ids[tr_idx].tolist())
        va_lesions = set(lesion_ids[va_idx].tolist())
        overlap    = tr_lesions & va_lesions
        if overlap:
            print(f"  [FAIL] Fold {fi}: {len(overlap)} lesions appear in both train and val")
            all_ok = False
        else:
            print(f"  [OK] Fold {fi}: lesion_ids non-overlapping between train and val")

    # ── Check 4: union of all fold samples = train_idx ────────────────────────
    all_val_indices: set[int] = set()
    for fold in folds:
        all_val_indices |= set(fold["val_abs_indices"])
    if all_val_indices == train_idx:
        print(f"  [OK] Union of all fold val sets = train_idx (complete coverage)")
    else:
        missing = train_idx - all_val_indices
        extra   = all_val_indices - train_idx
        if missing:
            print(f"  [FAIL] {len(missing)} train_idx samples never appear as fold val")
        if extra:
            print(f"  [FAIL] {len(extra)} fold val samples are not in train_idx")
        all_ok = False

    # ── Check 5: winner.json selection criterion ───────────────────────────────
    winner_path = os.path.join(CV_DIR, "winner.json")
    if not os.path.exists(winner_path):
        print(f"  [FAIL] winner.json not found at {winner_path}")
        all_ok = False
    else:
        with open(winner_path) as f:
            winner = json.load(f)

        forbidden_keys = {"test_balacc", "test_auc", "test_f1", "mean_test"}
        winner_keys    = set(str(k).lower() for k in winner.keys())
        bad_keys       = forbidden_keys & winner_keys
        if bad_keys:
            print(f"  [FAIL] winner.json contains test-related keys: {bad_keys}")
            all_ok = False
        else:
            print(f"  [OK] winner.json contains no test-related keys")

        # Confirm the stated selection criterion
        criterion = winner.get("selection_criterion", "")
        if "cv" in criterion.lower() and "test" not in criterion.lower():
            print(f"  [OK] winner.json selection_criterion: '{criterion}'")
        elif "test" in criterion.lower():
            print(f"  [FAIL] winner.json selection_criterion mentions 'test': '{criterion}'")
            all_ok = False
        else:
            print(f"  [WARN] winner.json selection_criterion unclear: '{criterion}'")

        assert winner.get("test_set_touched") is False, \
            "winner.json test_set_touched must be False"
        print(f"  [OK] winner.json test_set_touched = False")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED — no test-set leakage detected.")
    else:
        print("ONE OR MORE CHECKS FAILED — review output above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
