"""
v3 Step 1: select a stratified 300-image smoke-test subsample from the train pool.

Stratified by subtype label (normal / bacteria / virus) at the train-pool
proportions (~27% / 48% / 25%). Seed 42 (project convention).

Outputs:
    results/chestxray/v3_smoketest/subsample_indices.npz
        keys: subsample_idx, subsample_labels_binary, subsample_labels_subtype

Run from project root:
    python scripts/chestxray/v3_select_smoketest_subsample.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # robust on cp1252 consoles
except Exception:
    pass

import numpy as np
from sklearn.model_selection import train_test_split

OUTER_SPLIT = _ROOT / "data/splits/chestxray_outer_split.npz"
OUT_DIR     = _ROOT / "results/chestxray/v3_smoketest"
SUBSAMPLE_NPZ = OUT_DIR / "subsample_indices.npz"

N_SUBSAMPLE = 300
RANDOM_SEED = 42


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = np.load(OUTER_SPLIT, allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]
    labels_binary  = split["labels_binary"]
    labels_subtype = split["labels_subtype"]

    pool_subtype = labels_subtype[train_pool_idx]
    pool_binary  = labels_binary[train_pool_idx]

    print(f"Train pool size: {len(train_pool_idx)}")
    for cls in ("normal", "bacteria", "virus"):
        n = int((pool_subtype == cls).sum())
        print(f"  {cls}: {n} ({n / len(train_pool_idx) * 100:.1f}%)")

    # Stratified subsample of exactly N_SUBSAMPLE images by subtype.
    # train_test_split with stratify on the subtype labels; take the
    # "train_size = N_SUBSAMPLE" partition as the subsample.
    sub_local, _ = train_test_split(
        np.arange(len(train_pool_idx)),
        train_size=N_SUBSAMPLE,
        stratify=pool_subtype,
        random_state=RANDOM_SEED,
    )
    sub_local = np.sort(sub_local)

    subsample_idx            = train_pool_idx[sub_local]   # global image indices
    subsample_labels_binary  = pool_binary[sub_local]
    subsample_labels_subtype = pool_subtype[sub_local]

    print(f"\nSubsample size: {len(subsample_idx)}")
    for cls in ("normal", "bacteria", "virus"):
        n = int((subsample_labels_subtype == cls).sum())
        print(f"  {cls}: {n} ({n / len(subsample_idx) * 100:.1f}%)")

    np.savez(
        SUBSAMPLE_NPZ,
        subsample_idx            = subsample_idx,
        subsample_labels_binary  = subsample_labels_binary,
        subsample_labels_subtype = subsample_labels_subtype,
    )
    print(f"\nSaved → {SUBSAMPLE_NPZ.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
