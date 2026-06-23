"""
Trade-off analysis for the NAM v6 concurvity-regularization sweep.

Reads final-epoch validation metrics from each lambda run produced by
run_concurvity_sweep.sh, then:

  1. Prints a summary table: lambda, val_balanced_acc, r_perp_val, delta_acc.
  2. Identifies the elbow: largest lambda whose val_balanced_acc has not dropped
     more than TOLERANCE below the lambda=0 baseline.
  3. Saves a publication-ready trade-off plot (PNG + PDF) following the
     convention of Siems et al. (2023) "Curve Your Enthusiasm" Figs 2b/3b/4:
     x = val balanced accuracy, y = R_perp_val, one point per lambda.

Reads from: reports/nam/v6_concurvity_sweep/lambda_{value}/seed_{N}/training_log.csv
            (final row = final training epoch, matching Siems et al.'s reporting)

If multiple seeds are present for a lambda (Stage 2 output), metrics are
averaged across seeds before plotting.

Run from project root after run_concurvity_sweep.sh (Stage 1) finishes:
    python scripts/plot_concurvity_tradeoff.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SWEEP_ROOT = pathlib.Path("reports/nam/v6_concurvity_sweep")
OUT_PNG    = SWEEP_ROOT / "concurvity_tradeoff.png"
OUT_PDF    = SWEEP_ROOT / "concurvity_tradeoff.pdf"

# ── Elbow tolerance ───────────────────────────────────────────────────────────
# 1 SD of test balanced_accuracy across 5 seeds from the plain-NAM run.
# Source: aggregated_metrics.csv from reports/nam/v6_final (or v6_sweep_multiseed).
TOLERANCE = 0.013

# ── Column names (must match training_log.csv written by train_nam_v6_final.py)
COL_VAL_ACC  = "val_balanced_acc"
COL_RPERP    = "r_perp_val"
COL_EPOCH    = "epoch"


# ─────────────────────────────────────────────────────────────────────────────
# Discover and load runs
# ─────────────────────────────────────────────────────────────────────────────
def _lambda_from_dir(d: pathlib.Path) -> float:
    """Parse lambda value from directory name like 'lambda_0.01'."""
    return float(d.name.split("_", 1)[1])


def _load_final_epoch(lam_dir: pathlib.Path) -> dict | None:
    """Load final-epoch metrics averaged over all available seeds.

    Returns None and prints a warning if no usable seed is found.
    """
    seed_dirs = sorted(lam_dir.glob("seed_*/"))
    if not seed_dirs:
        print(f"  [warn] No seed_*/ subdirectories in {lam_dir} — skipping.")
        return None

    rows = []
    for sd in seed_dirs:
        log_path = sd / "training_log.csv"
        if not log_path.exists():
            print(f"  [warn] {log_path} not found — skipping seed {sd.name}.")
            continue
        log = pd.read_csv(log_path)
        if log.empty:
            print(f"  [warn] {log_path} is empty — skipping seed {sd.name}.")
            continue
        for col in (COL_VAL_ACC, COL_RPERP):
            if col not in log.columns:
                print(f"  [warn] Column '{col}' missing in {log_path} — "
                      f"skipping seed {sd.name}. "
                      f"Available: {list(log.columns)}")
                break
        else:
            # Take the last row (final training epoch)
            final = log.iloc[-1]
            rows.append({
                "seed":       sd.name,
                "n_epochs":   int(final[COL_EPOCH]),
                "val_acc":    float(final[COL_VAL_ACC]),
                "r_perp_val": float(final[COL_RPERP]),
            })

    if not rows:
        print(f"  [warn] No valid seed data in {lam_dir} — skipping.")
        return None

    df = pd.DataFrame(rows)
    return {
        "n_seeds":    len(df),
        "n_epochs":   df["n_epochs"].iloc[0],
        "val_acc":    float(df["val_acc"].mean()),
        "val_acc_std": float(df["val_acc"].std()) if len(df) > 1 else float("nan"),
        "r_perp_val":  float(df["r_perp_val"].mean()),
        "r_perp_std":  float(df["r_perp_val"].std()) if len(df) > 1 else float("nan"),
    }


if not SWEEP_ROOT.exists():
    print(f"[error] Sweep directory not found: {SWEEP_ROOT}")
    print("        Run  bash scripts/run_concurvity_sweep.sh  first.")
    sys.exit(1)

lam_dirs = sorted(
    [d for d in SWEEP_ROOT.iterdir() if d.is_dir() and d.name.startswith("lambda_")],
    key=_lambda_from_dir,
)

if not lam_dirs:
    print(f"[error] No lambda_* subdirectories found in {SWEEP_ROOT}")
    sys.exit(1)

records = []
for d in lam_dirs:
    lam = _lambda_from_dir(d)
    data = _load_final_epoch(d)
    if data is not None:
        records.append({"lambda": lam, **data})

if not records:
    print("[error] No usable runs found.  Check warnings above.")
    sys.exit(1)

df = pd.DataFrame(records).sort_values("lambda").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline and delta
# ─────────────────────────────────────────────────────────────────────────────
baseline_rows = df[df["lambda"] == 0.0]
if baseline_rows.empty:
    print("[error] No lambda=0.0 run found.  Cannot compute delta or elbow.")
    sys.exit(1)

baseline_acc   = float(baseline_rows["val_acc"].iloc[0])
baseline_rperp = float(baseline_rows["r_perp_val"].iloc[0])

df["delta_acc"]   = df["val_acc"]   - baseline_acc
df["delta_rperp"] = df["r_perp_val"] - baseline_rperp


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  Concurvity sweep — Stage 1 trade-off summary (final epoch metrics)")
print("=" * 72)
print(f"  Baseline (lambda=0): val_acc={baseline_acc:.4f}  "
      f"R_perp_val={baseline_rperp:.4f}")
print(f"  Tolerance           : {TOLERANCE} (1 SD of test bal_acc, plain NAM)")
print(f"  Threshold           : {baseline_acc - TOLERANCE:.4f}  "
      f"(baseline - tolerance)")
print("-" * 72)
header = (f"  {'lambda':>10}  {'val_acc':>10}  {'delta_acc':>10}  "
          f"{'R_perp_val':>12}  {'delta_R':>10}  {'n_seeds':>7}  {'n_epochs':>8}")
print(header)
print("-" * 72)
for _, row in df.iterrows():
    flag = ""
    if row["lambda"] == 0.0:
        flag = " [baseline]"
    elif row["val_acc"] >= baseline_acc - TOLERANCE:
        flag = " [within tol]"
    else:
        flag = " [below tol]"
    print(f"  {row['lambda']:>10.4g}  {row['val_acc']:>10.4f}  "
          f"{row['delta_acc']:>+10.4f}  {row['r_perp_val']:>12.4f}  "
          f"{row['delta_rperp']:>+10.4f}  {int(row['n_seeds']):>7d}  "
          f"{int(row['n_epochs']):>8d}{flag}")
print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Elbow recommendation
# ─────────────────────────────────────────────────────────────────────────────
within_tol = df[(df["lambda"] > 0) & (df["val_acc"] >= baseline_acc - TOLERANCE)]

print("")
if within_tol.empty:
    print("[warn] No lambda > 0 stays within tolerance.")
    print("       Options:")
    print("       (a) Extend the grid below 0.001, e.g. {0.0001, 0.0003, 0.001}")
    print("       (b) The regularizer is too aggressive at all tested values.")
    print("           Consider whether the val accuracy drop is acceptable for")
    print("           the R_perp reduction achieved.")
    elbow_lam    = None
    elbow_rperp  = None
else:
    elbow_row    = within_tol.loc[within_tol["lambda"].idxmax()]
    elbow_lam    = float(elbow_row["lambda"])
    elbow_rperp  = float(elbow_row["r_perp_val"])
    abs_rperp_reduction  = baseline_rperp - elbow_rperp
    pct_rperp_reduction  = abs_rperp_reduction / baseline_rperp * 100

    print("==== Elbow recommendation ====")
    print(f"  Largest lambda within tolerance : {elbow_lam}")
    print(f"  Val balanced accuracy           : {elbow_row['val_acc']:.4f} "
          f"(delta {elbow_row['delta_acc']:+.4f} from baseline)")
    print(f"  R_perp_val at elbow             : {elbow_rperp:.4f}")
    print(f"  R_perp reduction vs baseline    : "
          f"{abs_rperp_reduction:.4f} ({pct_rperp_reduction:.1f}%)")
    print(f"  Suggested Stage 2 command:")
    print(f"")
    print(f"    python scripts/train_nam_v6_final.py \\")
    print(f"        --concurvity_lambda {elbow_lam} \\")
    print(f"        --out_dir reports/nam/v6_final_lambda{elbow_lam}")
    print("==============================")


# ─────────────────────────────────────────────────────────────────────────────
# Trade-off plot (Siems et al. 2023 convention, Fig 2b/3b/4)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.0))

# Color-map points by log-lambda (0 gets a distinct marker)
cmap   = plt.get_cmap("viridis")
lam_vals = df["lambda"].values
# Use log-scale for color, treating lambda=0 specially
log_lam_nonzero = np.log10(lam_vals[lam_vals > 0])
v_min  = log_lam_nonzero.min() - 0.5 if len(log_lam_nonzero) else -4
v_max  = log_lam_nonzero.max() + 0.5 if len(log_lam_nonzero) else  0

# Draw connecting line first (behind points)
ax.plot(df["val_acc"], df["r_perp_val"],
        color="gray", linewidth=0.8, linestyle="-", zorder=1,
        alpha=0.6)

# Plot each point
for _, row in df.iterrows():
    lam  = row["lambda"]
    x    = row["val_acc"]
    y    = row["r_perp_val"]
    if lam == 0.0:
        color  = "#2166ac"   # blue for baseline
        marker = "D"
        ms     = 9
        zorder = 4
    else:
        norm_val = (np.log10(lam) - v_min) / (v_max - v_min + 1e-9)
        color    = cmap(float(np.clip(norm_val, 0, 1)))
        marker   = "o"
        ms       = 8
        zorder   = 3

    ax.scatter(x, y, color=color, marker=marker, s=ms**2,
               zorder=zorder, edgecolors="white", linewidths=0.5)

    # Lambda annotation
    lam_str = f"λ={lam:.4g}"
    x_off   = 0.0008
    y_off   = -0.006 if lam > 0.05 else 0.006
    ax.annotate(
        lam_str,
        xy=(x, y),
        xytext=(x + x_off, y + y_off),
        fontsize=8,
        color="black",
        ha="left",
        va="center",
    )

# Elbow marker
if elbow_lam is not None:
    elbow_x = float(df.loc[df["lambda"] == elbow_lam, "val_acc"].iloc[0])
    elbow_y = float(elbow_rperp)
    ax.scatter(elbow_x, elbow_y,
               color="none", edgecolors="#d6604d", linewidths=2.2,
               s=14**2, zorder=5, label=f"Elbow (λ={elbow_lam:.4g})")
    ax.legend(fontsize=8, framealpha=0.85, loc="upper right")

# Tolerance band (shaded region where acc is still acceptable)
ax.axvline(baseline_acc - TOLERANCE,
           color="#d6604d", linestyle="--", linewidth=0.9, alpha=0.7,
           label=f"Threshold (baseline − {TOLERANCE})")

# "Best" region annotation
ax.text(0.97, 0.04, "← better accuracy\n↓ lower concurvity",
        transform=ax.transAxes, fontsize=7.5, color="gray",
        ha="right", va="bottom", style="italic")

# Axes
ax.set_xlabel("Validation balanced accuracy  (higher = better)", fontsize=10)
ax.set_ylabel("Validation $R_{\\perp}$ (lower = better)", fontsize=10)
ax.set_title(
    "Concurvity regularization trade-off\n"
    "NAM v6, BiomedCLIP features, HAM10000",
    fontsize=10.5, fontweight="bold",
)
ax.text(0.5, -0.13,
        "Multiclass extension of Siems et al. (2023) Figs 2b/3b/4  "
        "(arXiv:2305.11475).  "
        "Final-epoch validation metrics, seed 42.",
        transform=ax.transAxes, fontsize=7, ha="center", color="dimgray",
        wrap=True)

ax.tick_params(labelsize=8)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(rect=[0, 0.06, 1, 1])

SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
fig.savefig(OUT_PDF,           bbox_inches="tight")
plt.close(fig)

print(f"\nPlot saved:")
print(f"  {OUT_PNG}")
print(f"  {OUT_PDF}")
