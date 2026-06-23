#!/usr/bin/env python
"""
Pairwise log-odds shape function plots for primary K=10 NAM models.

For a softmax multiclass NAM:

    log P(A|x) / P(B|x) = (β_A − β_B) + Σ_k [f_k,A(x_k) − f_k,B(x_k)]

This decomposition is exact. Plotting f_k,A(x_k) − f_k,B(x_k) shows the
contribution of concept k to the model's log-odds preference for class A over
class B at every input value of concept k, holding all other concepts fixed.
Positive values push toward A; negative values push toward B.

HAM10000 (sparsity_concurvity) and chest X-ray (sparsity_conc), 5 seeds each.
Reads existing checkpoints — no retraining.

Usage (from project root):
    python scripts/analysis/plot_shape_functions_pairwise.py --sanity_only
    python scripts/analysis/plot_shape_functions_pairwise.py
    python scripts/analysis/plot_shape_functions_pairwise.py --dataset chestxray
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.analysis.plot_shape_functions import (
    DATASET_CFGS,
    GRID_PCTHI,
    GRID_PCTLO,
    HIST_ALPHA,
    LINREF_ALPHA,
    LINREF_LW,
    MEAN_LW,
    N_GRID,
    SEED_ALPHA,
    SEED_LW,
    YREF_ALPHA,
    YREF_LW,
    SEEDS,
    compute_shape_fn,
    load_dataset,
    load_seed,
    monotonicity_label,
    peak_trough_pct,
)

# ── Pairwise configuration ─────────────────────────────────────────────────────
PAIRINGS: Dict[str, List[Tuple[str, str]]] = {
    "ham10000": [
        ("mel", "nv"),
        ("bcc", "nv"),
    ],
    "chestxray": [
        ("bacteria", "normal"),
        ("virus", "normal"),
        ("bacteria", "virus"),
    ],
}

OUT_DIR = _ROOT / "results" / "analysis" / "shape_function_plots_pairwise"
MULTICLASS_OUT_DIR = _ROOT / "results" / "analysis" / "shape_function_plots"
PAIR_COLOR = "#3060A8"

CENTRING_TOL = 5e-3
SPREAD_LOW = 0.02
SPREAD_HIGH = 0.08


# ── Pairwise computation ───────────────────────────────────────────────────────

def resolve_class_cols(class_names: List[str], class_a: str, class_b: str) -> Tuple[int, int]:
    if class_a not in class_names:
        raise KeyError(f"class '{class_a}' not in {class_names}")
    if class_b not in class_names:
        raise KeyError(f"class '{class_b}' not in {class_names}")
    return class_names.index(class_a), class_names.index(class_b)


def centred_diff_at_x(
    model,
    x_vals: np.ndarray,
    concept_idx: int,
    col_a: int,
    col_b: int,
) -> np.ndarray:
    """Per-class-mean-centred difference f_k,A(x) - f_k,B(x) at x_vals."""
    x_mean = float(np.mean(x_vals))
    with torch.no_grad():
        out = model.concept_contributions(x_vals, concept_idx).cpu().numpy()
        f_at_mean = model.concept_contributions(
            np.array([x_mean], dtype=np.float32), concept_idx
        ).cpu().numpy()[0]
    fc = out - f_at_mean
    return fc[:, col_a] - fc[:, col_b]


def compute_pairwise_diff(
    model,
    scaler,
    X_train_final: np.ndarray,
    concept_idx: int,
    col_a: int,
    col_b: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (x_grid, x_col_scaled, diff_centred_grid, r2_empirical)."""
    x_grid, x_col, fc = compute_shape_fn(model, scaler, X_train_final, concept_idx)
    diff_grid = (fc[:, col_a] - fc[:, col_b]).astype(np.float32)

    x_mean = float(x_col.mean())
    # Explicit re-centre difference on grid so diff(mean(x)) == 0 exactly.
    diff_at_mean = float(np.interp(x_mean, x_grid, diff_grid))
    diff_grid = (diff_grid - diff_at_mean).astype(np.float32)

    diff_emp = centred_diff_at_x(model, x_col, concept_idx, col_a, col_b)

    if np.std(diff_emp) < 1e-12:
        r2 = 1.0
    else:
        r2 = float(
            LinearRegression().fit(x_col.reshape(-1, 1), diff_emp).score(
                x_col.reshape(-1, 1), diff_emp
            )
        )
    return x_grid, x_col, diff_grid, r2


def diff_spread(
    seed_data: List[Tuple[np.ndarray, np.ndarray]],
    common_grid: np.ndarray,
) -> float:
    """Mean per-grid-point |seed_curve − mean_curve| for difference curves."""
    arr = np.array([np.interp(common_grid, xg, dg) for xg, dg in seed_data])
    return float(np.mean(np.abs(arr - arr.mean(axis=0))))


def spread_label(spread: float, median_spread: float) -> str:
    if spread <= SPREAD_LOW or spread <= 0.5 * median_spread:
        return "low"
    if spread >= SPREAD_HIGH or spread >= 1.5 * median_spread:
        return "high"
    return "medium"


def pair_dir_name(class_a: str, class_b: str) -> str:
    return f"{class_a}_vs_{class_b}"


def dataset_display(ds: str) -> str:
    return "HAM10000" if ds == "ham10000" else "Chest X-ray"


def condition_label(ds: str) -> str:
    return "sparsity_concurvity" if ds == "ham10000" else "sparsity_conc"


# ── Panel drawing ──────────────────────────────────────────────────────────────

def draw_pairwise_panel(
    ax_main,
    ax_hist,
    concept_name: str,
    class_a: str,
    class_b: str,
    seed_data: List[Tuple[np.ndarray, np.ndarray]],
    x_hist: np.ndarray,
    mean_r2: float,
    surviving_seeds: List[int],
    small: bool = False,
) -> np.ndarray:
    """Draw one pairwise difference panel. Returns mean_diff on common_grid."""
    lw_s = 0.7 if small else SEED_LW
    lw_m = 1.8 if small else MEAN_LW
    lw_l = 1.2 if small else LINREF_LW
    fs_t = 8 if small else 11
    fs_a = 7 if small else 10
    fs_k = 7 if small else 9
    fs_r = 6 if small else 8

    common_grid = seed_data[0][0]
    ax_main.axhline(0, color="grey", lw=YREF_LW, alpha=YREF_ALPHA, zorder=1)

    on_common = []
    for xg, dg in seed_data:
        interp = np.interp(common_grid, xg, dg)
        on_common.append(interp)
        ax_main.plot(common_grid, interp, color=PAIR_COLOR,
                     alpha=SEED_ALPHA, lw=lw_s, zorder=2)

    arr = np.array(on_common)
    mean_diff = arr.mean(axis=0)
    ax_main.plot(common_grid, mean_diff, color=PAIR_COLOR,
                 lw=lw_m, alpha=1.0, zorder=3)

    slope, intercept = np.polyfit(common_grid, mean_diff, 1)
    ax_main.plot(
        common_grid, slope * common_grid + intercept,
        color=PAIR_COLOR, lw=lw_l, alpha=LINREF_ALPHA, linestyle="--", zorder=2,
    )

    ymax = float(np.max(np.abs(mean_diff)))
    if ymax > 1e-9:
        margin = ymax * 0.08
        ax_main.set_ylim(-ymax - margin, ymax + margin)

    if not small:
        ax_main.text(
            0.97, 0.97, f"R²={mean_r2:.2f}",
            transform=ax_main.transAxes, fontsize=fs_r, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="lightgrey"),
        )
        ax_main.set_ylabel(f"log P({class_a}) / P({class_b}) — contribution", fontsize=fs_a)

    n_s = len(surviving_seeds)
    title = concept_name.replace("_", " ")
    if n_s < len(SEEDS):
        title += f" [n={n_s}/{len(SEEDS)}]"
    ax_main.set_title(title, fontsize=fs_t)
    ax_main.tick_params(labelsize=fs_k)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    bins = 30 if small else 40
    ax_hist.hist(x_hist, bins=bins, color="#404040", alpha=HIST_ALPHA,
                 range=(common_grid[0], common_grid[-1]))
    ax_main.set_xlim(common_grid[0], common_grid[-1])
    xlabel = concept_name.replace("_", " ") + ("" if small else " (scaled)")
    ax_hist.set_xlabel(xlabel, fontsize=fs_a)
    ax_hist.set_ylabel("n", fontsize=fs_r if small else 8)
    ax_hist.tick_params(labelsize=fs_k)

    return mean_diff


def make_individual_fig(
    concept_name: str,
    class_a: str,
    class_b: str,
    seed_data: List[Tuple[np.ndarray, np.ndarray]],
    x_hist: np.ndarray,
    mean_r2: float,
    surviving_seeds: List[int],
) -> Tuple[plt.Figure, np.ndarray]:
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=ax)
    mean_diff = draw_pairwise_panel(
        ax, axh, concept_name, class_a, class_b,
        seed_data, x_hist, mean_r2, surviving_seeds, small=False,
    )
    fig.tight_layout()
    return fig, mean_diff


def make_grid_fig(panel_records: List[dict], n_cols: int = 3) -> plt.Figure:
    n = len(panel_records)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig = plt.figure(figsize=(4.5 * n_cols, 3.8 * n_rows))
    outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.60, wspace=0.38)
    for pi, rec in enumerate(panel_records):
        row, col_i = divmod(pi, n_cols)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col_i], height_ratios=[5, 1], hspace=0.10,
        )
        ax = fig.add_subplot(inner[0])
        axh = fig.add_subplot(inner[1], sharex=ax)
        draw_pairwise_panel(
            ax, axh, rec["concept_name"], rec["class_a"], rec["class_b"],
            rec["seed_data"], rec["x_hist"], rec["mean_r2"],
            rec["surviving_seeds"], small=True,
        )
    return fig


# ── Interpretation template ────────────────────────────────────────────────────

def make_section(rec: dict) -> str:
    concept = rec["concept_name"]
    class_a = rec["class_a"]
    class_b = rec["class_b"]
    ds = rec["dataset"]
    n_s = len(rec["surviving_seeds"])
    mean_r2 = rec["mean_r2"]
    mean_diff = rec["mean_diff"]
    common_grid = rec["common_grid"]
    x_col = rec["x_hist"]
    spread = rec["seed_spread"]
    spread_lbl = rec["spread_label"]

    mono = monotonicity_label(mean_diff, common_grid, x_col)
    peak_pct, trough_pct = peak_trough_pct(mean_diff, common_grid, x_col)
    rng = float(mean_diff.max() - mean_diff.min())
    peak_loc = peak_pct if abs(mean_diff.max()) >= abs(mean_diff.min()) else trough_pct

    lines = [
        f"## {dataset_display(ds)}: {class_a} vs {class_b} — {concept.replace('_', ' ')}",
        "",
        f"**Pairing**: log P({class_a}) / P({class_b})",
        f"**Concept**: {concept}",
        f"**Survival**: {n_s}/{len(SEEDS)} seeds "
        f"(concept appeared in K=10 {condition_label(ds)} primary for {n_s} of {len(SEEDS)} seeds)",
        "",
        f"**Linear fit R²** (difference vs linear): {mean_r2:.2f} (mean across seeds)",
        "",
        "**Auto-derived characteristics**:",
        f"- Monotonic over [pct5, pct95]: {mono}",
        f"- Peak location: at x = {peak_loc:.0f} (= {peak_loc:.0f}-th percentile of empirical distribution)",
        f"- Range (max − min of mean difference): {rng:.2f}",
        f"- Seed spread: {spread:.2f} ({spread_lbl})",
        "",
        "**Interpretation** (TO FILL): "
        "[explain what the pairwise log-odds difference shows — "
        f"at low {concept.replace('_', ' ')} the model prefers X over Y; "
        f"at high {concept.replace('_', ' ')} it prefers Y over X; etc.]",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


# ── Sanity checks ──────────────────────────────────────────────────────────────

def sanity_checks(cfg_list: List[dict]) -> bool:
    ok = True
    sep = "=" * 60
    print(f"\n{sep}\nPAIRWISE SANITY CHECKS\n{sep}")

    # [1] Checkpoint files
    for cfg in cfg_list:
        print(f"\n[1] {cfg['name']} — checkpoint inventory:")
        for seed in SEEDS:
            sd = pathlib.Path(cfg["checkpoint_dir"]) / f"seed_{seed}"
            for fn in ("model.pt", "scaler.pkl", "meta.json"):
                p = sd / fn
                tag = "OK     " if p.exists() else "MISSING"
                if not p.exists():
                    ok = False
                print(f"  {tag}: {p.relative_to(_ROOT)}")

    # [2] Class labels for pairings
    for cfg in cfg_list:
        ds = cfg["name"]
        print(f"\n[2] {ds} — class label mapping:")
        try:
            data = load_dataset(cfg)
        except Exception as e:
            print(f"  FAIL loading data: {e}")
            ok = False
            continue
        class_names = data["class_names"]
        print(f"  class_names: {class_names}")
        for class_a, class_b in PAIRINGS[ds]:
            try:
                col_a, col_b = resolve_class_cols(class_names, class_a, class_b)
                print(f"  pairing ({class_a}, {class_b}) -> columns ({col_a}, {col_b}) OK")
            except KeyError as e:
                print(f"  FAIL pairing ({class_a}, {class_b}): {e}")
                ok = False

    # [3+4] Forward pass shape + difference mean-centring
    print(f"\n[3+4] Forward pass + difference mean-centring:")
    for cfg in cfg_list:
        ds = cfg["name"]
        try:
            data = load_dataset(cfg)
            model, scaler, meta = load_seed(cfg, 42)
        except Exception as e:
            print(f"  [{ds}] FAIL: {e}")
            ok = False
            continue

        X_tf = data["X_train_final"]
        concept_names = data["concept_names"]
        class_names = data["class_names"]
        class_a, class_b = PAIRINGS[ds][0]
        col_a, col_b = resolve_class_cols(class_names, class_a, class_b)

        sc0 = meta["surviving_concepts"][0]
        if sc0 not in concept_names:
            print(f"  [{ds}] WARNING: '{sc0}' not in concept_names")
            continue
        ci0 = concept_names.index(sc0)

        xg, x_col, diff_grid, _ = compute_pairwise_diff(
            model, scaler, X_tf, ci0, col_a, col_b,
        )
        if diff_grid.shape != (N_GRID,):
            print(f"  [{ds}] FAIL: diff_grid shape {diff_grid.shape} != ({N_GRID},)")
            ok = False
        else:
            print(f"  [{ds}] concept_contributions grid: ({N_GRID}, {cfg['num_classes']}) OK")

        x_mean = float(x_col.mean())
        diff_at_mean = centred_diff_at_x(
            model, np.array([x_mean], dtype=np.float32), ci0, col_a, col_b,
        )[0]
        max_dev = float(np.abs(diff_at_mean))
        status = "OK" if max_dev < CENTRING_TOL else "WARN"
        print(
            f"  [{ds}] mean-centring '{sc0}' ({class_a} vs {class_b}): "
            f"|diff| at mean(x) = {max_dev:.6f} (tol={CENTRING_TOL}) -> {status}"
        )
        if max_dev >= CENTRING_TOL:
            ok = False

    # [5] Existing multiclass plots untouched
    print(f"\n[5] Existing multiclass output directory:")
    if MULTICLASS_OUT_DIR.exists():
        print(f"  OK: {MULTICLASS_OUT_DIR.relative_to(_ROOT)} exists (not modified)")
    else:
        print(f"  WARN: {MULTICLASS_OUT_DIR.relative_to(_ROOT)} not found")

    print(f"\n{'All checks passed.' if ok else 'Some checks FAILED - fix before plotting.'}")
    return ok


# ── Process one (dataset, pairing) ─────────────────────────────────────────────

def process_pairing(
    cfg: dict,
    data: dict,
    class_a: str,
    class_b: str,
    seed_models: dict,
    seed_scalers: dict,
    seed_metas: dict,
    union_concepts: List[str],
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    ds = cfg["name"]
    concept_names = data["concept_names"]
    class_names = data["class_names"]
    X_tf = data["X_train_final"]
    col_a, col_b = resolve_class_cols(class_names, class_a, class_b)

    pair_name = pair_dir_name(class_a, class_b)
    indiv_dir = OUT_DIR / cfg["out_subdir"] / pair_name / "individual"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    data_rows: List[dict] = []
    r2_rows: List[dict] = []
    panel_records: List[dict] = []
    summary_rows: List[dict] = []

    for concept in union_concepts:
        if concept not in concept_names:
            print(f"  WARNING: '{concept}' not in concept_names — skipping")
            continue

        cidx = concept_names.index(concept)
        surv_seeds = [s for s in SEEDS if concept in seed_metas[s]["surviving_concepts"]]
        print(f"  {concept}: {len(surv_seeds)}/{len(SEEDS)} seeds", flush=True)

        x_grids: List[np.ndarray] = []
        diff_grids: List[np.ndarray] = []
        r2_vals: List[float] = []
        x_hist: np.ndarray | None = None

        for si, seed in enumerate(surv_seeds):
            xg, xc, diff_g, r2 = compute_pairwise_diff(
                seed_models[seed], seed_scalers[seed], X_tf, cidx, col_a, col_b,
            )
            x_grids.append(xg)
            diff_grids.append(diff_g)
            r2_vals.append(r2)
            if si == 0:
                x_hist = xc

            for gi in range(N_GRID):
                data_rows.append({
                    "dataset": ds,
                    "pair_A": class_a,
                    "pair_B": class_b,
                    "concept": concept,
                    "seed": seed,
                    "x_grid_value": float(xg[gi]),
                    "diff_value": float(diff_g[gi]),
                })
            r2_rows.append({
                "dataset": ds,
                "pair_A": class_a,
                "pair_B": class_b,
                "concept": concept,
                "seed": seed,
                "r2": r2,
            })

        seed_data = list(zip(x_grids, diff_grids))
        mean_r2 = float(np.mean(r2_vals))
        common_grid = x_grids[0]

        fig, mean_diff = make_individual_fig(
            concept, class_a, class_b, seed_data, x_hist, mean_r2, surv_seeds,
        )
        safe = concept.replace(" ", "_")
        fig.savefig(indiv_dir / f"{safe}.png", dpi=150, bbox_inches="tight")
        fig.savefig(indiv_dir / f"{safe}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"    -> {pair_name}/individual/{safe}.png  .pdf")

        spread = diff_spread(seed_data, common_grid)
        rng = float(mean_diff.max() - mean_diff.min())

        rec = {
            "dataset": ds,
            "class_a": class_a,
            "class_b": class_b,
            "pair_name": pair_name,
            "concept_name": concept,
            "seed_data": seed_data,
            "x_hist": x_hist,
            "mean_r2": mean_r2,
            "surviving_seeds": surv_seeds,
            "common_grid": common_grid,
            "mean_diff": mean_diff,
            "seed_spread": spread,
            "range": rng,
        }
        panel_records.append(rec)
        summary_rows.append(rec)

    if panel_records:
        spreads = [r["seed_spread"] for r in panel_records]
        med_spread = float(np.median(spreads))
        for rec in panel_records:
            rec["spread_label"] = spread_label(rec["seed_spread"], med_spread)

        grid_fig = make_grid_fig(panel_records)
        pair_root = OUT_DIR / cfg["out_subdir"] / pair_name
        grid_fig.savefig(pair_root / "grid_all.png", dpi=120, bbox_inches="tight")
        grid_fig.savefig(pair_root / "grid_all.pdf", bbox_inches="tight")
        plt.close(grid_fig)
        print(f"  Saved {cfg['out_subdir']}/{pair_name}/grid_all.png  .pdf")

    return data_rows, r2_rows, panel_records, summary_rows


def process_dataset(cfg: dict, data: dict) -> Tuple[List[dict], List[dict], List[dict]]:
    seed_models = {}
    seed_scalers = {}
    seed_metas = {}
    for seed in SEEDS:
        m, s, meta = load_seed(cfg, seed)
        seed_models[seed] = m
        seed_scalers[seed] = s
        seed_metas[seed] = meta

    seen: set = set()
    union: List[str] = []
    for seed in SEEDS:
        for cn in seed_metas[seed]["surviving_concepts"]:
            if cn not in seen:
                union.append(cn)
                seen.add(cn)
    print(f"\n  Union ({len(union)} concepts): {union}")

    all_data: List[dict] = []
    all_r2: List[dict] = []
    all_panels: List[dict] = []

    for class_a, class_b in PAIRINGS[cfg["name"]]:
        print(f"\n  Pairing: {class_a} vs {class_b}")
        data_rows, r2_rows, panels, _ = process_pairing(
            cfg, data, class_a, class_b,
            seed_models, seed_scalers, seed_metas, union,
        )
        all_data.extend(data_rows)
        all_r2.extend(r2_rows)
        all_panels.extend(panels)

    return all_data, all_r2, all_panels


# ── Reporting ──────────────────────────────────────────────────────────────────

def report_pairing(panels: List[dict], class_a: str, class_b: str) -> None:
    pair_panels = [
        p for p in panels
        if p["class_a"] == class_a and p["class_b"] == class_b
    ]
    if not pair_panels:
        return

    ds = pair_panels[0]["dataset"]
    pair_name = pair_dir_name(class_a, class_b)
    print(f"\n{dataset_display(ds)} / {pair_name}: {len(pair_panels)} panels")

    r2_vals = [p["mean_r2"] for p in pair_panels]
    print(
        f"  R² distribution (mean across seeds): "
        f"min={min(r2_vals):.3f}, median={np.median(r2_vals):.3f}, max={max(r2_vals):.3f}"
    )

    med_spread = float(np.median([p["seed_spread"] for p in pair_panels]))
    candidates = []
    for p in pair_panels:
        if p["mean_r2"] < 0.5 and p["seed_spread"] < med_spread and p["range"] > 0.15:
            score = (-p["range"], p["mean_r2"], p["seed_spread"])
            candidates.append((p["concept_name"], p["mean_r2"], p["seed_spread"], p["range"], score))

    candidates.sort(key=lambda x: x[4])
    print("  Top 3 striking panels (low R², low spread, meaningful range):")
    if candidates:
        for cn, r2, sp, rng, _ in candidates[:3]:
            print(f"    {cn}: mean_R²={r2:.2f}, spread={sp:.4f}, range={rng:.3f}")
    else:
        print("    (none meeting all criteria)")


def report_cross_pairing(all_panels: List[dict]) -> None:
    concept_hits: Dict[str, List[str]] = {}
    med_spread_global = float(np.median([p["seed_spread"] for p in all_panels])) if all_panels else 0.0

    for p in all_panels:
        if p["mean_r2"] < 0.5 and p["seed_spread"] < med_spread_global and p["range"] > 0.15:
            key = f"{p['dataset']}:{p['concept_name']}"
            pair = pair_dir_name(p["class_a"], p["class_b"])
            concept_hits.setdefault(key, []).append(pair)

    multi = {k: v for k, v in concept_hits.items() if len(v) > 1}
    print("\nCross-pairing striking concepts:")
    if multi:
        for key, pairs in sorted(multi.items()):
            print(f"  {key}: striking in {', '.join(pairs)}")
    else:
        print("  (none appearing striking in multiple pairings)")


def list_output_files() -> None:
    print("\nOutput files:")
    if not OUT_DIR.exists():
        print("  (output directory not created)")
        return
    files = sorted(OUT_DIR.rglob("*"))
    for p in files:
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"  {p.relative_to(_ROOT)}  ({size_kb:.1f} KB)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sanity_only", action="store_true",
                    help="Run sanity checks and exit without generating plots.")
    ap.add_argument("--dataset", choices=["ham10000", "chestxray"],
                    help="Process only one dataset (default: both).")
    args = ap.parse_args()

    ds_names = [args.dataset] if args.dataset else list(PAIRINGS.keys())
    cfg_list = [DATASET_CFGS[n] for n in ds_names]

    ok = sanity_checks(cfg_list)
    if not ok:
        sys.exit(1)
    if args.sanity_only:
        print("\n--sanity_only: exiting before plot generation.")
        return

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_data_rows: List[dict] = []
    all_r2_rows: List[dict] = []
    all_panels: List[dict] = []
    all_sections: List[str] = ["# Pairwise Shape Function Interpretation Template\n"]

    for cfg in cfg_list:
        ds = cfg["name"]
        print(f"\n{'='*60}\n{ds}\n{'='*60}")
        data = load_dataset(cfg)
        data_rows, r2_rows, panels = process_dataset(cfg, data)
        all_data_rows.extend(data_rows)
        all_r2_rows.extend(r2_rows)
        all_panels.extend(panels)

        for class_a, class_b in PAIRINGS[ds]:
            for rec in panels:
                if rec["class_a"] == class_a and rec["class_b"] == class_b:
                    all_sections.append(make_section(rec))

    pd.DataFrame(all_data_rows).to_csv(OUT_DIR / "pairwise_data.csv", index=False)
    print(f"\nSaved pairwise_data.csv ({len(all_data_rows)} rows)")

    pd.DataFrame(all_r2_rows).to_csv(OUT_DIR / "pairwise_r2.csv", index=False)
    print(f"Saved pairwise_r2.csv ({len(all_r2_rows)} rows)")

    (OUT_DIR / "interpretation_template.md").write_text(
        "\n".join(all_sections), encoding="utf-8",
    )
    print("Saved interpretation_template.md")

    elapsed = round(time.time() - t0, 1)
    run_cfg = {
        "datasets": ds_names,
        "pairings": {k: PAIRINGS[k] for k in ds_names},
        "seeds": SEEDS,
        "n_grid": N_GRID,
        "grid_pctlo": GRID_PCTLO,
        "grid_pcthi": GRID_PCTHI,
        "pair_color": PAIR_COLOR,
        "centring_tol": CENTRING_TOL,
        "wall_clock_s": elapsed,
    }
    (OUT_DIR / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2), encoding="utf-8",
    )
    print("Saved run_config.json")

    print(f"\n{'='*60}\nREPORT\n{'='*60}")
    for cfg in cfg_list:
        ds = cfg["name"]
        for class_a, class_b in PAIRINGS[ds]:
            report_pairing(all_panels, class_a, class_b)

    report_cross_pairing(all_panels)
    list_output_files()
    print(f"\nTotal wall-clock: {elapsed}s")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
