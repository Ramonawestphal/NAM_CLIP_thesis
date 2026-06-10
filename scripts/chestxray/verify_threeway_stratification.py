"""
Three-way Step 0b: verify the pre-built CV folds are acceptably stratified for
the three-way (normal/bacteria/virus) label.

The folds in chestxray_cv_folds.npz were built with StratifiedGroupKFold on the
BINARY label. Before reusing them for the three-way task, we check that each
fold's validation partition has three-way class proportions within 5 percentage
points of the global train-pool proportions.

Decision rule:
  max |fold_val_prop - global_prop| <= 5pp  ->  [VERIFIED], proceed
  otherwise                                 ->  [REJECTED], stop and ask Ramona
  (do NOT silently rebuild folds — they have verified patient grouping)

Output:
    results/chestxray/architecture_selection/preflight_stratification.txt

Run from project root:
    python scripts/chestxray/verify_threeway_stratification.py
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

OUTER_SPLIT = _ROOT / "data/splits/chestxray_outer_split.npz"
CV_FOLDS    = _ROOT / "data/splits/chestxray_cv_folds.npz"
OUT_DIR     = _ROOT / "results/chestxray/architecture_selection"

CLASSES   = ["normal", "bacteria", "virus"]
N_FOLDS   = 5
TOL_PP    = 5.0  # percentage-point tolerance


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = np.load(OUTER_SPLIT, allow_pickle=True)
    cv    = np.load(CV_FOLDS, allow_pickle=True)
    labels_subtype = split["labels_subtype"]
    train_pool_idx = split["train_pool_idx"]

    # Global three-way proportions across the train pool
    pool_sub = labels_subtype[train_pool_idx]
    global_prop = {c: float((pool_sub == c).mean()) for c in CLASSES}

    L: list[str] = []
    L.append("=" * 68)
    L.append("THREE-WAY STRATIFICATION PREFLIGHT")
    L.append("(CV folds were stratified by the BINARY label; checking three-way)")
    L.append("=" * 68)
    L.append(f"Train pool: {len(train_pool_idx)} images")
    L.append("Global three-way proportions:")
    for c in CLASSES:
        L.append(f"  {c:<9s}: {global_prop[c]*100:5.1f}%  "
                 f"({int((pool_sub == c).sum())} images)")
    L.append("")
    L.append(f"Per-fold VAL partition proportions (deviation from global, pp):")
    header = "  fold  " + "".join(f"{c:>22s}" for c in CLASSES)
    L.append(header)

    max_dev = 0.0
    max_dev_where = None
    for fold_i in range(N_FOLDS):
        fv = cv[f"fold_val_idx_{fold_i}"]
        sub = labels_subtype[fv]
        cells = []
        for c in CLASSES:
            prop = float((sub == c).mean())
            dev_pp = abs(prop - global_prop[c]) * 100
            if dev_pp > max_dev:
                max_dev = dev_pp
                max_dev_where = (fold_i, c)
            cells.append(f"{prop*100:6.1f}% (Δ{dev_pp:4.1f})")
        L.append(f"  {fold_i:>4}  " + "".join(f"{cell:>22s}" for cell in cells))

    L.append("")
    L.append(f"Max absolute deviation: {max_dev:.2f} pp "
             f"(fold {max_dev_where[0]}, {max_dev_where[1]})")
    L.append(f"Tolerance: {TOL_PP:.1f} pp")

    accepted = max_dev <= TOL_PP
    L.append("")
    if accepted:
        L.append(f"[VERIFIED] Three-way stratification within tolerance.")
    else:
        L.append(f"[REJECTED] Three-way stratification exceeds {TOL_PP:.0f}pp tolerance.")

    report = "\n".join(L)
    (OUT_DIR / "preflight_stratification.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved → {(OUT_DIR / 'preflight_stratification.txt').relative_to(_ROOT)}")

    if not accepted:
        print("\n[STOP] Do not rebuild folds silently. Ask Ramona before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
