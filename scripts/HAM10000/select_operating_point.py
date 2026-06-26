"""
Select sparsity operating point — v7 corrected pipeline (STEP 6).

Issue 11 fix: codifies the operating point selection rule that was previously
applied informally.  Selection is fully determined by the validation path; the
test set is NOT inspected at any point during this script.

Selection rule
──────────────
  1. Load the warm-start path CSV from run_sparsity_sweep.py (STEP 5).
  2. Compute: dense_val_balacc = val_balacc at lambda=0 (from dense phase).
  3. Mark a lambda step as *acceptable* if:
       val_balacc >= dense_val_balacc - 0.02
  4. Among acceptable steps, select the one with the fewest active features
     (minimum n_active).
  5. Tie-break: largest lambda (maximizes regularization).
  6. Write the selected lambda to results/HAM10000/operating_point.json.

Usage (from project root)
──────────────────────────
  python scripts/HAM10000/select_operating_point.py
  python scripts/HAM10000/select_operating_point.py --path_csv results/HAM10000/sparsity_sweep/path_seed42.csv
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

from scripts.HAM10000._common import write_step_flag

# ── Constants ─────────────────────────────────────────────────────────────────
SEED               = 42
TOLERANCE          = 0.02     # Issue 11: acceptable = dense_val_balacc - 0.02
DEFAULT_PATH_CSV   = "results/HAM10000/sparsity_sweep/path_seed42.csv"
DEFAULT_DENSE_JSON = "results/HAM10000/sparsity_sweep/seed_42/dense_seed42_conc1p0_ep100.json"
OUT_JSON           = "results/HAM10000/operating_point.json"
RESULTS_V7         = "results/HAM10000"
STEP_N             = 6


def find_dense_json(sweep_root: str, concurvity_lambda: float = 1.0) -> str | None:
    """Locate the dense checkpoint JSON in the sweep directory."""
    conc_tag  = str(concurvity_lambda).replace(".", "p")
    seed_dir  = os.path.join(sweep_root, f"seed_{SEED}")
    if not os.path.isdir(seed_dir):
        return None
    for fname in os.listdir(seed_dir):
        if fname.startswith(f"dense_seed{SEED}_conc{conc_tag}") and fname.endswith(".json"):
            return os.path.join(seed_dir, fname)
    return None


def select_operating_point(
    path_df:          pd.DataFrame,
    dense_val_balacc: float,
    tolerance:        float = TOLERANCE,
    verbose:          bool  = True,
) -> dict:
    """Apply the Issue 11 operating-point selection rule.

    Returns a dict with selected_lambda, n_active, val_balacc, rule description.
    """
    acceptable = path_df[path_df["val_balacc"] >= dense_val_balacc - tolerance].copy()

    if verbose:
        print(f"  Dense val_balacc   : {dense_val_balacc:.4f}")
        print(f"  Tolerance          : -{tolerance:.4f}")
        print(f"  Threshold          : {dense_val_balacc - tolerance:.4f}")
        print(f"  Total path steps   : {len(path_df)}")
        print(f"  Acceptable steps   : {len(acceptable)}")

    if acceptable.empty:
        # Fallback: pick the step with the highest val_balacc
        best_idx = path_df["val_balacc"].idxmax()
        selected = path_df.loc[best_idx]
        rule     = (
            f"FALLBACK — no step met val_balacc >= {dense_val_balacc - tolerance:.4f}; "
            f"selected highest val_balacc step"
        )
        if verbose:
            print(f"  [WARN] No acceptable step found — using fallback (highest val_balacc)")
    else:
        # Step 4: minimum n_active among acceptable
        min_n_active = acceptable["n_active"].min()
        candidates   = acceptable[acceptable["n_active"] == min_n_active]

        # Step 5: tie-break by largest lambda
        best_idx = candidates["lambda"].idxmax()
        selected = candidates.loc[best_idx]
        rule     = (
            f"Issue 11 fix: min n_active (={int(min_n_active)}) among "
            f"val_balacc >= {dense_val_balacc:.4f} - {tolerance:.4f} = "
            f"{dense_val_balacc - tolerance:.4f}; "
            f"tie-break: largest lambda"
        )
        if verbose:
            print(f"  Min n_active among acceptable: {int(min_n_active)}")
            print(f"  Candidates at min n_active   : {len(candidates)}")

    return {
        "selected_lambda":      float(selected["lambda"]),
        "selected_n_active":    int(selected["n_active"]),
        "selected_val_balacc":  float(selected["val_balacc"]),
        "dense_val_balacc":     float(dense_val_balacc),
        "tolerance":            float(tolerance),
        "threshold":            float(dense_val_balacc - tolerance),
        "n_acceptable_steps":   int(len(acceptable)),
        "selection_rule":       rule,
        "test_set_touched":     False,
        "seed":                 SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path_csv",   type=str, default=None,
                        help=f"Path to path_seed{SEED}.csv (default: {DEFAULT_PATH_CSV})")
    parser.add_argument("--dense_json", type=str, default=None,
                        help="Path to dense checkpoint JSON (auto-detected if omitted).")
    parser.add_argument("--sweep_root", type=str, default="results/HAM10000/sparsity_sweep",
                        help="Root of the sparsity sweep directory (for auto-detection).")
    parser.add_argument("--concurvity_lambda", type=float, default=1.0,
                        help="Concurvity lambda used in the sweep (for auto-detection).")
    parser.add_argument("--tolerance",  type=float, default=TOLERANCE,
                        help=f"Val balacc tolerance (default={TOLERANCE}).")
    parser.add_argument("--out_json",   type=str,   default=OUT_JSON)
    args = parser.parse_args()

    path_csv = args.path_csv or DEFAULT_PATH_CSV
    if not os.path.exists(path_csv):
        raise FileNotFoundError(
            f"Path CSV not found: {path_csv}. "
            "Run run_sparsity_sweep.py (STEP 5) first."
        )

    # ── Load path CSV ──────────────────────────────────────────────────────────
    path_df = pd.read_csv(path_csv, comment="#")
    if "lambda" not in path_df.columns:
        raise ValueError(f"Expected 'lambda' column in {path_csv}.")

    print(f"\n{'='*65}")
    print(f"NAM v7 — Operating point selection (STEP {STEP_N})")
    print(f"  Issue 11 fix: selection rule codified (val-only, 0.02 tolerance)")
    print(f"  Path CSV: {path_csv}")
    print(f"  Steps in path: {len(path_df)}")
    print(f"  Lambda range:  {path_df['lambda'].min():.4e} – {path_df['lambda'].max():.4e}")
    print(f"{'='*65}\n")

    # ── Load dense val_balacc ──────────────────────────────────────────────────
    dense_json_path = args.dense_json
    if dense_json_path is None:
        dense_json_path = find_dense_json(args.sweep_root, args.concurvity_lambda)
    if dense_json_path is None or not os.path.exists(dense_json_path):
        raise FileNotFoundError(
            f"Dense checkpoint JSON not found.  "
            "Provide --dense_json or run run_sparsity_sweep.py first."
        )

    with open(dense_json_path) as f:
        dense_meta = json.load(f)
    dense_val_balacc = float(dense_meta["dense_val_balacc"])
    print(f"  Dense JSON: {dense_json_path}")

    # ── Apply selection rule ───────────────────────────────────────────────────
    result = select_operating_point(
        path_df=path_df,
        dense_val_balacc=dense_val_balacc,
        tolerance=args.tolerance,
        verbose=True,
    )

    print(f"\n  [SELECTED]")
    print(f"    lambda        = {result['selected_lambda']:.4e}")
    print(f"    n_active      = {result['selected_n_active']}")
    print(f"    val_balacc    = {result['selected_val_balacc']:.4f}  "
          f"(dense={result['dense_val_balacc']:.4f}, "
          f"delta={result['selected_val_balacc'] - result['dense_val_balacc']:+.4f})")

    # ── Annotate path CSV with acceptance flag ─────────────────────────────────
    path_df["is_acceptable"] = (
        path_df["val_balacc"] >= dense_val_balacc - args.tolerance
    )
    path_df["is_selected"] = path_df["lambda"] == result["selected_lambda"]
    annotated_csv = path_csv.replace(".csv", "_annotated.csv")
    path_df.to_csv(annotated_csv, index=False)
    print(f"\n  Annotated path → {annotated_csv}")

    # ── Write operating_point.json ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Operating point → {args.out_json}")

    print(f"\n{'='*65}")
    print(f"  Selection rule: {result['selection_rule']}")
    print(f"{'='*65}")

    write_step_flag(RESULTS_V7, STEP_N)


if __name__ == "__main__":
    main()
