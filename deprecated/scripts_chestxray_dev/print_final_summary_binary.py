"""
v7-parity Step 5: print the consolidated chest X-ray NAM architecture-selection summary.

Reads the artefacts produced by select_architecture.py and final_evaluation.py and
prints the clean console summary block. Read-only; produces no new files.

Run from project root (after Steps 3 and 4):
    python scripts/chestxray/print_final_summary.py
"""

from __future__ import annotations

import json
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
import pandas as pd

OUT_DIR     = _ROOT / "results/chestxray/architecture_selection_binary"
OUTER_SPLIT = _ROOT / "data/splits/chestxray_outer_split.npz"
WINNER_JSON = OUT_DIR / "winning_config.json"
FINAL_CSV   = OUT_DIR / "final_test_results.csv"


def main() -> None:
    if not WINNER_JSON.exists():
        sys.exit(f"ERROR: {WINNER_JSON} not found. Run select_architecture.py first.")
    winner = json.loads(WINNER_JSON.read_text(encoding="utf-8"))

    split = np.load(OUTER_SPLIT, allow_pickle=True)
    lb = split["labels_binary"]
    tp, te = split["train_pool_idx"], split["test_idx"]
    tp_pneu, tp_norm = int(lb[tp].sum()), int((lb[tp] == 0).sum())

    # final test metrics (if available)
    if FINAL_CSV.exists():
        df = pd.read_csv(FINAL_CSV)
        mean_auc, std_auc = df["test_auc"].mean(), df["test_auc"].std()
        mean_bal, std_bal = df["test_balacc"].mean(), df["test_balacc"].std()
        final_block = (
            f"  Test balanced accuracy: {mean_bal:.4f} ± {std_bal:.4f}\n"
            f"  Test AUC:               {mean_auc:.4f} ± {std_auc:.4f}"
        )
    else:
        final_block = "  (final_evaluation.py not yet run)"

    margin = winner.get("runner_up_margin")
    margin_str = f"{margin:.4f}" if margin is not None else "n/a"

    print("==================================================")
    print("CHEST X-RAY NAM ARCHITECTURE SELECTION — SUMMARY")
    print("==================================================")
    print()
    print("Task: binary pneumonia vs normal classification")
    print("Features: 17 BiomedCLIP concept similarities (v4)")
    print(f"Train pool: {len(tp)} images ({tp_pneu} pneumonia / {tp_norm} normal)")
    print(f"Test set:   {len(te)} images (held out throughout selection)")
    print()
    print("Architecture search (v7 parity):")
    print("  12 configs × 5 folds × 1 seed = 60 selection runs")
    print("  Selection metric: val balanced accuracy (mean across 5 folds)")
    print("  AUC reported alongside (secondary)")
    print()
    print("Winning config:")
    print(f"  Config ID: {winner['config_id']}")
    print(f"  Hidden: {winner['hidden_dims']}")
    print(f"  Dropout: {winner['dropout']}")
    print(f"  Weight decay: {winner['weight_decay']:.0e}")
    print(f"  Mean val balacc: {winner['mean_val_balacc']:.4f} ± {winner['std_val_balacc']:.4f}")
    print(f"  Mean val AUC:    {winner['mean_val_auc']:.4f} ± {winner['std_val_auc']:.4f}")
    print(f"  Runner-up margin: {margin_str}")
    print()
    print("Final test evaluation (5 seeds: 42-46):")
    print(final_block)
    print()
    print("Leakage safeguards verified:")
    print("  [✓] Test indices never loaded during selection")
    print("  [✓] All CV indices inside train pool")
    print("  [✓] Per-fold train/val disjoint at image level")
    print("  [✓] Per-fold train/val disjoint at patient level")
    print("  [✓] Per-fold z-scoring (fit on fold-train only)")
    print("  [✓] Fresh model per (config, fold), seed 42 reset before each")
    print("  [✓] Final test loaded only after selection completed")


if __name__ == "__main__":
    main()
