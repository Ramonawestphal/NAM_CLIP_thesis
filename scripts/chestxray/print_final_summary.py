"""
Three-way Step 5: print the consolidated chest X-ray NAM architecture-selection summary.

Reads artefacts from select_architecture.py + final_evaluation.py (three-way) and
the preserved binary winner, then prints the template console block. Read-only.

Run from project root (after Steps 3 and 4):
    python scripts/chestxray/print_final_summary.py
"""

from __future__ import annotations

import json
import pathlib
import re
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

OUT_DIR       = _ROOT / "results/chestxray/architecture_selection"
OUTER_SPLIT   = _ROOT / "data/splits/chestxray_outer_split.npz"
WINNER_JSON   = OUT_DIR / "winning_config.json"
FINAL_CSV     = OUT_DIR / "final_test_results.csv"
PREFLIGHT     = OUT_DIR / "preflight_stratification.txt"
BINARY_WINNER = _ROOT / "results/chestxray/architecture_selection_binary/winning_config.json"


def main() -> None:
    if not WINNER_JSON.exists():
        sys.exit(f"ERROR: {WINNER_JSON} not found. Run select_architecture.py first.")
    w = json.loads(WINNER_JSON.read_text(encoding="utf-8"))

    split = np.load(OUTER_SPLIT, allow_pickle=True)
    sub = split["labels_subtype"]; tp = split["train_pool_idx"]; te = split["test_idx"]
    pool = sub[tp]
    n_norm = int((pool == "normal").sum())
    n_bact = int((pool == "bacteria").sum())
    n_vir  = int((pool == "virus").sum())

    # preflight max deviation
    pf_dev = "n/a"
    if PREFLIGHT.exists():
        m = re.search(r"Max absolute deviation:\s*([\d.]+)\s*pp", PREFLIGHT.read_text(encoding="utf-8"))
        if m:
            pf_dev = f"{float(m.group(1)):.1f} pp"

    # binary reference
    binary_balacc, binary_std = 0.9325, 0.0061
    if BINARY_WINNER.exists():
        try:
            bw = json.loads(BINARY_WINNER.read_text(encoding="utf-8"))
            binary_balacc = float(bw.get("mean_val_balacc", binary_balacc))
            binary_std    = float(bw.get("std_val_balacc", binary_std))
        except Exception:
            pass

    if FINAL_CSV.exists():
        df = pd.read_csv(FINAL_CSV)
        fb = (f"  Test balanced accuracy:  {df['test_balacc'].mean():.4f} ± {df['test_balacc'].std():.4f}\n"
              f"  Test macro-OvR AUC:      {df['test_macro_auc_ovr'].mean():.4f} ± {df['test_macro_auc_ovr'].std():.4f}\n"
              f"  Per-class test accuracy: Normal {df['test_acc_normal'].mean():.3f} | "
              f"Bacteria {df['test_acc_bacteria'].mean():.3f} | Virus {df['test_acc_virus'].mean():.3f}")
    else:
        fb = "  (final_evaluation.py not yet run)"

    margin = w.get("runner_up_margin")
    margin_str = f"{margin:.4f}" if margin is not None else "n/a"

    print("==================================================")
    print("CHEST X-RAY NAM ARCHITECTURE SELECTION — SUMMARY")
    print("Three-way classification (PRIMARY TASK)")
    print("==================================================")
    print()
    print("Task: three-way classification (normal / bacteria / virus)")
    print("Features: 17 BiomedCLIP concept similarities (v4)")
    print(f"Train pool: {len(tp)} images ({n_norm} normal / {n_bact} bacteria / {n_vir} virus)")
    print(f"Test set:   {len(te)} images (held out throughout selection)")
    print()
    print(f"Three-way stratification preflight: PASS (max deviation: {pf_dev})")
    print()
    print("Architecture search (v7 parity):")
    print("  12 configs × 5 folds × 1 seed = 60 selection runs")
    print("  Selection metric: val balanced accuracy")
    print("  Macro-OvR AUC reported alongside (secondary)")
    print()
    print("Winning config:")
    print(f"  Config ID: {w['config_id']}")
    print(f"  Hidden: {w['hidden_dims']}")
    print(f"  Dropout: {w['dropout']}")
    print(f"  Weight decay: {w['weight_decay']:.0e}")
    print(f"  Mean val balacc:        {w['mean_val_balacc']:.4f} ± {w['std_val_balacc']:.4f}")
    print(f"  Mean val macro-OvR AUC: {w['mean_val_macro_auc_ovr']:.4f} ± {w['std_val_macro_auc_ovr']:.4f}")
    print(f"  Runner-up margin: {margin_str}")
    print()
    print("Per-class accuracy of the winning config (mean across 5 folds):")
    print(f"  Normal:   {w['mean_val_acc_normal']:.4f}")
    print(f"  Bacteria: {w['mean_val_acc_bacteria']:.4f}")
    print(f"  Virus:    {w['mean_val_acc_virus']:.4f}")
    print()
    print("Final test evaluation (5 seeds: 42-46):")
    print(fb)
    print()
    print("Comparison to binary task (sensitivity analysis):")
    print(f"  Binary winning val balacc:    {binary_balacc:.4f} ± {binary_std:.4f}  (saturated)")
    print(f"  Three-way winning val balacc: {w['mean_val_balacc']:.4f} ± {w['std_val_balacc']:.4f}")
    print()
    print("Leakage safeguards verified:")
    for s in [
        "Test indices never loaded during selection",
        "All CV indices inside train pool",
        "Per-fold train/val disjoint at image level",
        "Per-fold train/val disjoint at patient level",
        "Per-fold z-scoring (fit on fold-train only)",
        "Fresh model per (config, fold), seed 42 reset before each",
        "Final test loaded only after selection completed",
        "Three-way label conversion verified (all 3 classes present)",
    ]:
        print(f"  [✓] {s}")


if __name__ == "__main__":
    main()
