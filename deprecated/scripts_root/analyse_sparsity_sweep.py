"""
Post-hoc analysis of warm-start Group LASSO sparsity sweep results.

Covers Steps 5–7 of the production plan:
  Step 5: Seed stability — 5-seed mean±CI bands on val_auc and active count.
  Step 6: Cross-condition comparison (concurvity_lambda=0 vs 1) and
          cold-vs-warm comparison at matched lambdas.
  Step 7: Plain-text summary report of key numbers (no interpretation).

Reads:
  results/sparsity_sweep/warm_noconc/path_seed{42..46}.csv
  results/sparsity_sweep/warm_conc/path_seed{42..46}.csv
  results/sparsity_sweep/cold_noconc/lambda_*/seed_*/training_log.csv  (optional)
  results/sparsity_sweep/cold_conc/lambda_*/seed_*/training_log.csv    (optional)

Writes under results/sparsity_sweep/analysis/:
  seed_stability_noconc.png
  seed_stability_conc.png
  cross_condition.png
  cold_vs_warm.png          (only when cold-start results available)
  report.txt

Run from project root:
    python scripts/analyse_sparsity_sweep.py
    python scripts/analyse_sparsity_sweep.py --warm_noconc results/sparsity_sweep/warm_noconc
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [42, 43, 44, 45, 46]

# Threshold below which a sub-network's group norm is considered zeroed.
ZERO_THRESHOLD = 1e-8

# Cold-start CSV structure: each lambda run writes
# reports/nam/v6_concurvity_sweep/lambda_{value}/seed_{N}/training_log.csv
# The sparsity cold-start follows the same convention (from run_sparsity_sweep.py).
# We only need final-epoch val_balacc and val_auc per (lambda, seed).


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_warm_paths(warm_dir: str, seeds: list[int]) -> dict[int, pd.DataFrame]:
    """Load path_seed{N}.csv for each seed in seeds.  Returns {seed: df}."""
    paths: dict[int, pd.DataFrame] = {}
    for s in seeds:
        p = os.path.join(warm_dir, f"path_seed{s}.csv")
        if not os.path.exists(p):
            print(f"  [skip] {p} not found")
            continue
        df = pd.read_csv(p, comment="#").sort_values("lambda").reset_index(drop=True)
        paths[s] = df
    return paths


def _get_concept_names(df: pd.DataFrame) -> list[str]:
    return [c[5:] for c in df.columns if c.startswith("norm_")]


def load_cold_start_summary(cold_dir: str) -> pd.DataFrame | None:
    """
    Collect per-lambda metrics from cold-start runs.

    train_nam_v6_final.py writes two summary files per lambda directory:
        cold_dir/lambda_{lam}/aggregated_metrics.csv  (best-epoch metrics, per seed)
        cold_dir/lambda_{lam}/seed_{N}/training_log.csv  (per-epoch log, fallback)

    Priority: aggregated_metrics.csv (uses best-epoch snapshot and includes val_auc).
    Fallback:  training_log.csv last row (uses final training epoch, no val_auc).

    Column mapping from aggregated_metrics.csv:
        balanced_accuracy  -> val_balacc
        auc_ovr_weighted   -> val_auc
        n_selected         -> n_active  (number of non-zeroed feature groups)

    Returns a DataFrame with columns: lambda, seed, val_balacc, val_auc, n_active.
    Returns None if the directory is absent or no files are found.
    """
    if not os.path.isdir(cold_dir):
        return None
    records = []
    for lam_dir in sorted(pathlib.Path(cold_dir).iterdir()):
        if not lam_dir.is_dir() or not lam_dir.name.startswith("lambda_"):
            continue
        try:
            lam = float(lam_dir.name.replace("lambda_", ""))
        except ValueError:
            continue

        # ── Primary: aggregated_metrics.csv ────────────────────────────────
        agg_csv = lam_dir / "aggregated_metrics.csv"
        if agg_csv.exists():
            agg = pd.read_csv(agg_csv)
            # Rows: one per seed, plus "mean" and "std" rows — keep numeric seeds only.
            for _, row in agg.iterrows():
                try:
                    seed_val = int(row["seed"])
                except (ValueError, KeyError):
                    continue  # skip "mean" / "std" rows
                records.append({
                    "lambda":     lam,
                    "seed":       seed_val,
                    "val_balacc": float(row.get("balanced_accuracy", float("nan"))),
                    "val_auc":    float(row.get("auc_ovr_weighted",  float("nan"))),
                    "n_active":   float(row.get("n_selected",        float("nan"))),
                })
            continue

        # ── Fallback: per-seed training_log.csv ────────────────────────────
        for seed_dir in sorted(lam_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            log = seed_dir / "training_log.csv"
            if not log.exists():
                continue
            tlog = pd.read_csv(log)
            last = tlog.iloc[-1]
            # training_log uses "val_balanced_acc" (not "val_balacc"); no val_auc col.
            balacc = last.get("val_balanced_acc",
                     last.get("val_balacc", float("nan")))
            records.append({
                "lambda":     lam,
                "seed":       int(seed_dir.name.replace("seed_", "")),
                "val_balacc": float(balacc),
                "val_auc":    float("nan"),
                "n_active":   float("nan"),
            })

    if not records:
        return None
    return pd.DataFrame(records).sort_values(["lambda", "seed"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared interpolation for cross-seed aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _shared_log_grid(seed_dfs: dict[int, pd.DataFrame], n_pts: int = 300) -> np.ndarray:
    """Log-spaced lambda grid covering the intersection of all seeds' ranges."""
    lo = max(df["lambda"].min() for df in seed_dfs.values())
    hi = min(df["lambda"].max() for df in seed_dfs.values())
    return np.logspace(np.log10(lo), np.log10(hi), n_pts)


def _interp_col(df: pd.DataFrame, col: str, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, df["lambda"].values, df[col].values)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Seed stability
# ─────────────────────────────────────────────────────────────────────────────

def plot_seed_stability(
    seed_dfs:   dict[int, pd.DataFrame],
    condition:  str,
    out_dir:    str,
) -> dict:
    """
    Plot val_auc mean ± 95% CI band and active-subnet mean ± CI across seeds.

    Returns a summary dict with keys:
      - n_seeds_loaded
      - phase_transition_lambda_{k}: first lambda where mean active count <= k,
        for k in [20, 15, 10, 5, 1]
      - val_auc_at_{lambda}: val_auc mean ± std at nearest-available lambda values
        [0.1, 0.3, 1, 3, 10, 30, 100, 300]
    """
    if not seed_dfs:
        print(f"  [skip] no seeds loaded for {condition}")
        return {}

    grid = _shared_log_grid(seed_dfs)

    auc_mat    = np.stack([_interp_col(df, "val_auc",  grid) for df in seed_dfs.values()])
    balacc_mat = np.stack([_interp_col(df, "val_balacc", grid) for df in seed_dfs.values()])
    active_mat = np.stack([_interp_col(df, "n_active", grid) for df in seed_dfs.values()])

    n = len(seed_dfs)
    ci_fac = 1.96 / np.sqrt(n)  # 95% CI half-width = 1.96 * SE

    auc_mean    = auc_mat.mean(0)
    auc_ci      = auc_mat.std(0, ddof=1) * ci_fac if n > 1 else np.zeros_like(auc_mean)
    active_mean = active_mat.mean(0)
    active_ci   = active_mat.std(0, ddof=1) * ci_fac if n > 1 else np.zeros_like(active_mean)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Individual seed traces (thin, translucent)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))
    for (seed, df), c in zip(seed_dfs.items(), colors):
        ax1.semilogx(df["lambda"], df["val_auc"],
                     color=c, linewidth=0.7, alpha=0.45, label=f"seed {seed}")
        ax2.semilogx(df["lambda"], df["n_active"],
                     color=c, linewidth=0.7, alpha=0.45)

    # Mean ± CI band
    ax1.semilogx(grid, auc_mean, color="black", linewidth=2.0, label="mean")
    ax1.fill_between(grid, auc_mean - auc_ci, auc_mean + auc_ci,
                     color="black", alpha=0.15, label="95% CI")
    ax1.set_ylabel("Val AUC  (OvR weighted)", fontsize=10)
    ax1.set_title(f"Seed stability — {condition}\n"
                  f"({n} seeds: {list(seed_dfs.keys())})", fontsize=10)
    ax1.legend(fontsize=7, ncol=2, loc="lower left")

    ax2.semilogx(grid, active_mean, color="black", linewidth=2.0)
    ax2.fill_between(grid, active_mean - active_ci, active_mean + active_ci,
                     color="black", alpha=0.15)
    ax2.axhline(24, color="gray", linestyle=":", linewidth=1.0, label="all 24")
    ax2.set_xlabel("λ  (log scale)", fontsize=10)
    ax2.set_ylabel("Active subnets", fontsize=10)
    ax2.set_ylim(-0.5, 25.5)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(4))

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"seed_stability_{condition}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")

    # ── Summary statistics ────────────────────────────────────────────────────
    summary: dict = {"n_seeds_loaded": n, "condition": condition}

    # Phase transition: first lambda where mean active ≤ k
    for k in [20, 15, 10, 5, 1]:
        mask = active_mean <= k
        if mask.any():
            summary[f"phase_trans_lam_le{k:02d}"] = float(grid[mask][0])
        else:
            summary[f"phase_trans_lam_le{k:02d}"] = float("inf")

    # Val AUC at key lambdas
    for lam_key in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        idx = int(np.argmin(np.abs(grid - lam_key)))
        key = f"val_auc_at_{lam_key:.1f}".replace(".", "p")
        summary[key] = {
            "mean": float(auc_mean[idx]),
            "std":  float(auc_mat.std(0, ddof=1)[idx]) if n > 1 else 0.0,
        }

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Step 5b: Concept elimination stability
# ─────────────────────────────────────────────────────────────────────────────

def print_elimination_stability(seed_dfs: dict[int, pd.DataFrame],
                                condition: str) -> list[tuple]:
    """
    For each concept, report the lambda at which it was first zeroed across seeds.
    Returns list of (concept, mean_lam, std_lam, n_seeds_zeroed) sorted by mean_lam.
    """
    if not seed_dfs:
        return []

    sample_df   = next(iter(seed_dfs.values()))
    concepts    = _get_concept_names(sample_df)
    elim_lambdas: dict[str, list[float]] = {c: [] for c in concepts}

    for df in seed_dfs.values():
        for c in concepts:
            col = f"norm_{c}"
            if col not in df.columns:
                continue
            zeroed = df[df[col] <= ZERO_THRESHOLD]["lambda"]
            if len(zeroed):
                elim_lambdas[c].append(float(zeroed.iloc[0]))

    rows = []
    for c in concepts:
        lams = elim_lambdas[c]
        rows.append((c,
                     float(np.mean(lams)) if lams else float("inf"),
                     float(np.std(lams, ddof=1)) if len(lams) > 1 else 0.0,
                     len(lams)))
    rows.sort(key=lambda r: r[1])

    print(f"\n  Elimination order — {condition} (mean across seeds):")
    print(f"  {'order':>5}  {'concept':30s}  {'mean_lam':>10}  "
          f"{'std_lam':>9}  {'n_zeroed':>8}")
    print("  " + "-" * 68)
    for i, (c, mean_l, std_l, n_z) in enumerate(rows, 1):
        suffix = f"  ({i})" if n_z == len(seed_dfs) else f"  ({n_z}/{len(seed_dfs)} seeds)"
        if mean_l == float("inf"):
            print(f"  {i:5d}  {c:30s}  {'never':>10}  {'':>9}  {n_z:>8}")
        else:
            print(f"  {i:5d}  {c:30s}  {mean_l:10.3e}  {std_l:9.3e}  {n_z:>8}{suffix}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Cross-condition comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_condition(
    seed_dfs_noconc: dict[int, pd.DataFrame],
    seed_dfs_conc:   dict[int, pd.DataFrame],
    out_dir: str,
) -> None:
    """
    3-panel figure:
      Panel 1: Val AUC mean ± CI  (both conditions overlaid)
      Panel 2: Active subnets     (both conditions)
      Panel 3: Val balanced acc   (both conditions)
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for seed_dfs, label, color in [
        (seed_dfs_noconc, "sparsity only  (λ_c=0)",   "steelblue"),
        (seed_dfs_conc,   "sparsity+conc  (λ_c=1.0)", "darkorange"),
    ]:
        if not seed_dfs:
            continue
        grid = _shared_log_grid(seed_dfs)
        n    = len(seed_dfs)
        ci_fac = 1.96 / np.sqrt(n) if n > 1 else 0.0

        for metric, ax in zip(["val_auc", "n_active", "val_balacc"], axes):
            mat  = np.stack([_interp_col(df, metric, grid)
                             for df in seed_dfs.values()])
            mean = mat.mean(0)
            ci   = mat.std(0, ddof=1) * ci_fac if n > 1 else np.zeros_like(mean)
            ax.semilogx(grid, mean, color=color, linewidth=2.0, label=label)
            ax.fill_between(grid, mean - ci, mean + ci,
                            color=color, alpha=0.12)

    axes[0].set_ylabel("Val AUC", fontsize=10)
    axes[1].set_ylabel("Active subnets", fontsize=10)
    axes[1].set_ylim(-0.5, 25.5)
    axes[1].yaxis.set_major_locator(plt.MultipleLocator(4))
    axes[2].set_ylabel("Val balanced acc", fontsize=10)
    axes[2].set_xlabel("λ  (log scale)", fontsize=10)

    for ax in axes:
        ax.legend(fontsize=8, loc="lower left")

    axes[0].set_title(
        "Cross-condition comparison — sparsity-only vs sparsity+concurvity\n"
        "Mean ± 95% CI across 5 seeds",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, "cross_condition.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6b: Cold vs warm comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_cold_vs_warm(
    seed_dfs_warm: dict[int, pd.DataFrame],
    cold_df: pd.DataFrame,
    condition: str,
    out_dir: str,
) -> None:
    """Warm-start path (mean±CI) overlaid with cold-start points."""
    if not seed_dfs_warm:
        return
    grid = _shared_log_grid(seed_dfs_warm)
    n    = len(seed_dfs_warm)
    ci_fac = 1.96 / np.sqrt(n) if n > 1 else 0.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for metric, ax, ylabel in [
        ("val_auc",    ax1, "Val AUC"),
        ("val_balacc", ax2, "Val balanced acc"),
    ]:
        mat  = np.stack([_interp_col(df, metric, grid)
                         for df in seed_dfs_warm.values()])
        mean = mat.mean(0)
        ci   = mat.std(0, ddof=1) * ci_fac if n > 1 else np.zeros_like(mean)
        ax.semilogx(grid, mean, color="steelblue", linewidth=2.0,
                    label=f"warm-start mean ± CI  (n={n})")
        ax.fill_between(grid, mean - ci, mean + ci, color="steelblue", alpha=0.15)

        # Cold-start: per lambda, single seed (or mean if multi-seed)
        if cold_df is not None:
            cold_pivot = cold_df.groupby("lambda")[metric].mean()
            ax.scatter(cold_pivot.index, cold_pivot.values,
                       color="darkorange", zorder=5, s=40,
                       label="cold-start (seed 42)")

        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="lower left")

    ax1.set_title(f"Cold-start vs warm-start — {condition}\n"
                  "Warm path: mean ± 95% CI  |  Cold: individual lambda runs",
                  fontsize=10)
    ax2.set_xlabel("λ  (log scale)", fontsize=10)
    fig.tight_layout()

    safe = condition.replace(" ", "_").replace("=", "").replace(".", "p")
    out_path = os.path.join(out_dir, f"cold_vs_warm_{safe}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Plain-text report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    summaries: dict,
    elim_noconc: list,
    elim_conc:   list,
    out_dir: str,
) -> None:
    lines = [
        "Sparsity sweep — plain-text report",
        "=" * 65,
        "",
    ]

    for cond, summ in summaries.items():
        if not summ:
            continue
        lines += [
            f"Condition: {cond}",
            f"  Seeds loaded: {summ.get('n_seeds_loaded', '?')}",
            "",
            "  Phase transition lambda (first lambda where mean active <= k):",
        ]
        for k in [20, 15, 10, 5, 1]:
            val = summ.get(f"phase_trans_lam_le{k:02d}", float("inf"))
            val_str = f"{val:.3e}" if val != float("inf") else "not reached"
            lines.append(f"    <= {k:2d} active: {val_str}")
        lines.append("")
        lines.append("  Val AUC at key lambdas (mean ± std across seeds):")
        for lam_key in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
            key = f"val_auc_at_{lam_key:.1f}".replace(".", "p")
            if key in summ:
                v = summ[key]
                lines.append(f"    lambda = {lam_key:6.1f}: "
                              f"AUC = {v['mean']:.4f} ± {v['std']:.4f}")
        lines += ["", "-" * 65, ""]

    if elim_noconc:
        lines.append("Concept elimination order — sparsity only (λ_c=0):")
        lines.append(f"  {'order':>5}  {'concept':30s}  mean_lam    n_zeroed")
        lines.append("  " + "-" * 60)
        for i, (c, ml, sl, nz) in enumerate(elim_noconc, 1):
            ml_s = f"{ml:.3e}" if ml != float("inf") else "never"
            lines.append(f"  {i:5d}  {c:30s}  {ml_s:10s}  {nz}")
        lines.append("")

    if elim_conc:
        lines.append("Concept elimination order — sparsity+concurvity (λ_c=1.0):")
        lines.append(f"  {'order':>5}  {'concept':30s}  mean_lam    n_zeroed")
        lines.append("  " + "-" * 60)
        for i, (c, ml, sl, nz) in enumerate(elim_conc, 1):
            ml_s = f"{ml:.3e}" if ml != float("inf") else "never"
            lines.append(f"  {i:5d}  {c:30s}  {ml_s:10s}  {nz}")
        lines.append("")

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[text] Report saved: {report_path}")

    # Also print to stdout
    print("\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm_noconc", default="results/sparsity_sweep/warm_noconc",
                        help="Dir with path_seed*.csv for concurvity_lambda=0 condition.")
    parser.add_argument("--warm_conc",   default="results/sparsity_sweep/warm_conc",
                        help="Dir with path_seed*.csv for concurvity_lambda=1 condition.")
    parser.add_argument("--cold_noconc", default="results/sparsity_sweep/cold_noconc",
                        help="Cold-start results dir, concurvity_lambda=0 (optional).")
    parser.add_argument("--cold_conc",   default="results/sparsity_sweep/cold_conc",
                        help="Cold-start results dir, concurvity_lambda=1 (optional).")
    parser.add_argument("--out_dir",     default="results/sparsity_sweep/analysis",
                        help="Output directory for plots and report.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Seeds to analyse.  Default: 42 43 44 45 46")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = args.seeds

    print("=" * 65)
    print("Sparsity sweep analysis  (Steps 5–7)")
    print("=" * 65)

    # ── Load warm-start paths ─────────────────────────────────────────────────
    print("\nLoading warm-start paths (concurvity_lambda=0)...")
    dfs_noconc = load_warm_paths(args.warm_noconc, seeds)
    print(f"  Loaded {len(dfs_noconc)} / {len(seeds)} seeds: {sorted(dfs_noconc)}")

    print("\nLoading warm-start paths (concurvity_lambda=1.0)...")
    dfs_conc = load_warm_paths(args.warm_conc, seeds)
    print(f"  Loaded {len(dfs_conc)} / {len(seeds)} seeds: {sorted(dfs_conc)}")

    # ── Load cold-start results (optional) ───────────────────────────────────
    cold_noconc = load_cold_start_summary(args.cold_noconc)
    cold_conc   = load_cold_start_summary(args.cold_conc)
    if cold_noconc is None:
        print("\n[cold] No cold-start results found at "
              f"{args.cold_noconc} — cold-vs-warm plot skipped.")
    if cold_conc is None:
        print(f"[cold] No cold-start results found at "
              f"{args.cold_conc} — cold-vs-warm plot skipped.")

    # ── Step 5: Seed stability ────────────────────────────────────────────────
    print("\n[Step 5] Seed stability...")
    summ_noconc = {}
    summ_conc   = {}

    if dfs_noconc:
        summ_noconc = plot_seed_stability(
            dfs_noconc, "noconc_lc0", args.out_dir
        )
        print_elimination_stability(dfs_noconc, "sparsity only (λ_c=0)")

    if dfs_conc:
        summ_conc = plot_seed_stability(
            dfs_conc, "conc_lc1", args.out_dir
        )
        print_elimination_stability(dfs_conc, "sparsity+concurvity (λ_c=1.0)")

    # ── Step 6: Cross-condition and cold-vs-warm ──────────────────────────────
    print("\n[Step 6] Cross-condition comparison...")
    if dfs_noconc and dfs_conc:
        plot_cross_condition(dfs_noconc, dfs_conc, args.out_dir)
    else:
        print("  [skip] need both conditions to plot cross-condition comparison")

    print("\n[Step 6b] Cold-vs-warm comparison...")
    if dfs_noconc and cold_noconc is not None:
        plot_cold_vs_warm(dfs_noconc, cold_noconc, "sparsity_only_lc0", args.out_dir)
    if dfs_conc and cold_conc is not None:
        plot_cold_vs_warm(dfs_conc, cold_conc, "sparsity_conc_lc1p0", args.out_dir)

    # ── Step 7: Report ────────────────────────────────────────────────────────
    print("\n[Step 7] Writing report...")

    # Elimination order (from the first available seed's data per condition,
    # then re-computed across seeds by print_elimination_stability).
    elim_noconc: list[tuple] = []
    elim_conc:   list[tuple] = []
    if dfs_noconc:
        elim_noconc = print_elimination_stability(dfs_noconc, "sparsity only")
    if dfs_conc:
        elim_conc   = print_elimination_stability(dfs_conc,   "sparsity+conc")

    write_report(
        summaries={"sparsity only (λ_c=0)": summ_noconc,
                   "sparsity+concurvity (λ_c=1.0)": summ_conc},
        elim_noconc=elim_noconc,
        elim_conc=elim_conc,
        out_dir=args.out_dir,
    )

    print(f"\nDone. Outputs under {args.out_dir}/")


if __name__ == "__main__":
    main()
