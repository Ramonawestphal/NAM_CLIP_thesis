#!/usr/bin/env python
"""
Shape function plots for primary K=10 NAM models.

HAM10000 (sparsity_concurvity) and chest X-ray (sparsity_conc), 5 seeds each.
Reads existing checkpoints — no retraining.

Usage (from project root):
    python scripts/analysis/plot_shape_functions.py --sanity_only
    python scripts/analysis/plot_shape_functions.py
    python scripts/analysis/plot_shape_functions.py --dataset chestxray
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.nam_multiclass import NAMMulticlass
from scripts.HAM10000._common import make_fixed_val_split

# ── Constants ──────────────────────────────────────────────────────────────────
SEEDS      = [42, 43, 44, 45, 46]
OUT_DIR    = _ROOT / "results" / "analysis" / "shape_function_plots"
R2_CSV     = _ROOT / "results" / "analysis" / "nonlinearity" / "shape_function_r2.csv"
N_GRID     = 200
GRID_PCTLO = 2.0
GRID_PCTHI = 98.0

SEED_ALPHA   = 0.30;  SEED_LW    = 1.0
MEAN_LW      = 2.50
LINREF_ALPHA = 0.70;  LINREF_LW  = 1.5
YREF_ALPHA   = 0.50;  YREF_LW    = 0.5
HIST_ALPHA   = 0.60

CX_COLORS = {"normal": "#5078A0", "bacteria": "#E07B39", "virus": "#2E8B7A"}

DATASET_CFGS: Dict[str, dict] = {
    "ham10000": {
        "name":           "ham10000",
        "features_path":  _ROOT / "data/features/biomedclip/ham10000_concept_scores_v6.npz",
        "splits_path":    _ROOT / "data/splits/train_test_lesion_split.npz",
        "checkpoint_dir": _ROOT / "results/HAM10000/primary_checkpoints",
        "n_features":     24,
        "num_classes":    7,
        "hidden_dims":    (64, 32),
        "dropout":        0.1,
        "out_subdir":     "ham10000",
        "r2_key":         "ham10000",
    },
    "chestxray": {
        "name":           "chestxray",
        "features_path":  _ROOT / "data/features/biomedclip/chestxray_concept_scores_v4.npz",
        "splits_path":    _ROOT / "data/splits/chestxray_outer_split.npz",
        "label_map_path": _ROOT / "results/chestxray/architecture_selection/label_mapping.json",
        "checkpoint_dir": _ROOT / "results/chestxray/primary_checkpoints",
        "n_features":     17,
        "num_classes":    3,
        "hidden_dims":    (64, 32),
        "dropout":        0.1,
        "out_subdir":     "chestxray",
        "r2_key":         "chestxray",
        "class_names":    ["normal", "bacteria", "virus"],
        "class_colors":   CX_COLORS,
    },
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ham10000(cfg: dict) -> dict:
    feat          = np.load(cfg["features_path"], allow_pickle=True)
    scores        = feat["scores"].astype(np.float32)
    labels        = feat["labels"]
    lesion_ids    = feat["lesion_ids"]
    concept_names = feat["concept_ids"].tolist()          # key is concept_ids for HAM10000
    class_names   = sorted(np.unique(labels).tolist())

    split     = np.load(cfg["splits_path"])
    train_idx = split["train_idx"]

    X_pool = scores[train_idx]
    y_pool = labels[train_idx]
    groups = lesion_ids[train_idx]
    vsplit = make_fixed_val_split(X_pool, y_pool, groups, class_names, val_random_state=42)
    X_tf   = X_pool[vsplit["train_rel"]]

    tab10  = plt.get_cmap("tab10")
    colors = {cn: mcolors.to_hex(tab10(i)) for i, cn in enumerate(class_names)}

    return {"concept_names": concept_names, "class_names": class_names,
            "class_colors": colors, "X_train_final": X_tf}


def load_chestxray(cfg: dict) -> dict:
    with open(cfg["label_map_path"]) as f:
        sub2int = json.load(f)

    feat          = np.load(cfg["features_path"], allow_pickle=True)
    scores        = feat["scores"].astype(np.float32)
    concept_names = feat["concept_names"].tolist()

    split          = np.load(cfg["splits_path"], allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]
    labels_subtype = split["labels_subtype"]
    patient_ids    = split["patient_ids"]

    labels_int = np.array([sub2int[s] for s in labels_subtype], dtype=np.int64)
    X_pool     = scores[train_pool_idx]
    y_pool_str = labels_int[train_pool_idx].astype(str)
    groups     = patient_ids[train_pool_idx]
    vsplit     = make_fixed_val_split(X_pool, y_pool_str, groups, ["0", "1", "2"],
                                      val_random_state=42)
    X_tf       = X_pool[vsplit["train_rel"]]

    return {"concept_names": concept_names, "class_names": cfg["class_names"],
            "class_colors": cfg["class_colors"], "X_train_final": X_tf}


def load_dataset(cfg: dict) -> dict:
    return load_ham10000(cfg) if cfg["name"] == "ham10000" else load_chestxray(cfg)


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def load_seed(cfg: dict, seed: int) -> Tuple[NAMMulticlass, object, dict]:
    sd = pathlib.Path(cfg["checkpoint_dir"]) / f"seed_{seed}"
    with open(sd / "meta.json") as f:
        meta = json.load(f)
    with open(sd / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = NAMMulticlass(n_features=cfg["n_features"], num_classes=cfg["num_classes"],
                          hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"])
    try:
        state = torch.load(sd / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(sd / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, scaler, meta


# ── Shape function computation ─────────────────────────────────────────────────

def compute_shape_fn(
    model: NAMMulticlass,
    scaler,
    X_train_final: np.ndarray,
    concept_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x_grid, x_col_scaled, f_centred).
    x_grid     : (N_GRID,) in scaled coordinates
    x_col_scaled: (N_train,) all training values for histogram
    f_centred  : (N_GRID, C) mean-centred shape function
    """
    X_sc   = scaler.transform(X_train_final.astype(np.float32))
    x_col  = X_sc[:, concept_idx]
    lo, hi = np.percentile(x_col, [GRID_PCTLO, GRID_PCTHI])
    x_grid = np.linspace(lo, hi, N_GRID).astype(np.float32)

    with torch.no_grad():
        f_grid    = model.concept_contributions(x_grid, concept_idx).cpu().numpy()
        x_mean_pt = np.array([float(x_col.mean())], dtype=np.float32)
        f_at_mean = model.concept_contributions(x_mean_pt, concept_idx).cpu().numpy()

    return x_grid, x_col.astype(np.float32), (f_grid - f_at_mean).astype(np.float32)


# ── R² helpers ─────────────────────────────────────────────────────────────────

def load_r2_df() -> pd.DataFrame:
    return pd.read_csv(R2_CSV)


def r2_dict_for(r2_df: pd.DataFrame, ds_key: str, concept: str,
                class_names: List[str]) -> Dict[str, float]:
    sub = r2_df[(r2_df["dataset"] == ds_key) & (r2_df["concept"] == concept)]
    return {cn: float(sub.loc[sub["class"] == cn, "r2"].mean())
            if len(sub[sub["class"] == cn]) > 0 else float("nan")
            for cn in class_names}


# ── Auto-derived characteristics ───────────────────────────────────────────────

def monotonicity_label(curve: np.ndarray, x_grid: np.ndarray, x_col: np.ndarray) -> str:
    p5, p95 = np.percentile(x_col, [5, 95])
    sub = curve[(x_grid >= p5) & (x_grid <= p95)]
    if len(sub) < 2:
        return "N/A"
    d    = np.diff(sub)
    n_sc = int(np.sum((d[:-1] * d[1:]) < 0)) if len(d) > 1 else 0
    return "Yes" if n_sc == 0 else ("Mostly" if n_sc == 1 else "No")


def peak_trough_pct(curve: np.ndarray, x_grid: np.ndarray,
                    x_col: np.ndarray) -> Tuple[float, float]:
    px = x_grid[np.argmax(curve)]
    tx = x_grid[np.argmin(curve)]
    return float(np.mean(x_col <= px) * 100), float(np.mean(x_col <= tx) * 100)


def class_spread(seed_data: List[Tuple[np.ndarray, np.ndarray]],
                 common_grid: np.ndarray, ci: int) -> float:
    arr = np.array([np.interp(common_grid, xg, fg[:, ci]) for xg, fg in seed_data])
    return float(np.mean(np.abs(arr - arr.mean(axis=0))))


# ── Panel drawing ──────────────────────────────────────────────────────────────

def draw_panel(
    ax_main, ax_hist,
    concept_name: str,
    class_names: List[str],
    colors: Dict[str, str],
    seed_data: List[Tuple[np.ndarray, np.ndarray]],
    x_hist: np.ndarray,
    r2_dict: Dict[str, float],
    surviving_seeds: List[int],
    small: bool = False,
) -> np.ndarray:
    """Draw concept panel into existing axes. Returns mean_curves (N_GRID, C)."""
    lw_s = 0.7 if small else SEED_LW
    lw_m = 1.8 if small else MEAN_LW
    lw_l = 1.2 if small else LINREF_LW
    fs_t = 8   if small else 11
    fs_a = 7   if small else 10
    fs_k = 7   if small else  9
    fs_r = 6   if small else  8

    common_grid = seed_data[0][0]
    ax_main.axhline(0, color="grey", lw=YREF_LW, alpha=YREF_ALPHA, zorder=1)

    mean_curves = np.zeros((N_GRID, len(class_names)))
    for ci, cname in enumerate(class_names):
        col = colors[cname]
        on_common = []
        for xg, fg in seed_data:
            fon = np.interp(common_grid, xg, fg[:, ci])
            on_common.append(fon)
            ax_main.plot(common_grid, fon, color=col, alpha=SEED_ALPHA, lw=lw_s, zorder=2)
        arr    = np.array(on_common)
        mean_c = arr.mean(axis=0)
        mean_curves[:, ci] = mean_c

        ax_main.plot(common_grid, mean_c, color=col, lw=lw_m, alpha=1.0, zorder=3,
                     label=cname if not small else None)
        slope, intercept = np.polyfit(common_grid, mean_c, 1)
        ax_main.plot(common_grid, slope * common_grid + intercept,
                     color=col, lw=lw_l, alpha=LINREF_ALPHA, linestyle="--", zorder=2)

    if not small:
        r2_txt = "\n".join(
            f"{cn}: R²={r2_dict.get(cn, float('nan')):.2f}" for cn in class_names
        )
        ax_main.text(0.97, 0.97, r2_txt, transform=ax_main.transAxes,
                     fontsize=fs_r, va="top", ha="right",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="lightgrey"))
        ax_main.legend(fontsize=fs_r, loc="upper left", framealpha=0.8)

    n_s   = len(surviving_seeds)
    title = concept_name.replace("_", " ")
    if n_s < len(SEEDS):
        title += f" [n={n_s}/{len(SEEDS)}]"
    ax_main.set_title(title, fontsize=fs_t)
    if not small:
        ax_main.set_ylabel("Shape function (logit)", fontsize=fs_a)
    ax_main.tick_params(labelsize=fs_k)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    bins    = 30 if small else 40
    ax_hist.hist(x_hist, bins=bins, color="#404040", alpha=HIST_ALPHA,
                 range=(common_grid[0], common_grid[-1]))
    ax_main.set_xlim(common_grid[0], common_grid[-1])
    xlabel  = concept_name.replace("_", " ") + ("" if small else " (scaled)")
    ax_hist.set_xlabel(xlabel, fontsize=fs_a)
    ax_hist.set_ylabel("n", fontsize=fs_r if small else 8)
    ax_hist.tick_params(labelsize=fs_k)

    return mean_curves


def make_individual_fig(concept_name, class_names, colors, seed_data,
                        x_hist, r2_dict, surviving_seeds) -> Tuple[plt.Figure, np.ndarray]:
    fig = plt.figure(figsize=(7.0, 5.0))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.08)
    ax  = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=ax)
    mean_curves = draw_panel(ax, axh, concept_name, class_names, colors,
                             seed_data, x_hist, r2_dict, surviving_seeds, small=False)
    fig.tight_layout()
    return fig, mean_curves


def make_grid_fig(panel_records: List[dict], class_names: List[str],
                  colors: Dict[str, str], n_cols: int = 3) -> plt.Figure:
    n      = len(panel_records)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig    = plt.figure(figsize=(4.5 * n_cols, 3.8 * n_rows))
    outer  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.60, wspace=0.38)
    for pi, rec in enumerate(panel_records):
        row, col_i = divmod(pi, n_cols)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col_i], height_ratios=[5, 1], hspace=0.10
        )
        ax  = fig.add_subplot(inner[0])
        axh = fig.add_subplot(inner[1], sharex=ax)
        draw_panel(ax, axh, rec["concept_name"], class_names, colors,
                   rec["seed_data"], rec["x_hist"], rec["r2_dict"],
                   rec["surviving_seeds"], small=True)
    return fig


# ── Interpretation template ────────────────────────────────────────────────────

def make_section(rec: dict, dataset_display: str) -> str:
    concept     = rec["concept_name"]
    class_names = rec["class_names"]
    n_s         = len(rec["surviving_seeds"])
    r2_dict     = rec["r2_dict"]
    mean_curves = rec["mean_curves"]
    common_grid = rec["common_grid"]
    x_col       = rec["x_hist"]
    seed_data   = rec["seed_data"]

    lines = [f"## {dataset_display}: {concept.replace('_', ' ')}",
             "",
             f"**Survival**: {n_s}/{len(SEEDS)} seeds at K=10 sparsity_conc",
             "",
             "**R² (linear fit) per class** (from results/analysis/nonlinearity/shape_function_r2.csv):"]
    for cn in class_names:
        lines.append(f"- {cn}: {r2_dict.get(cn, float('nan')):.3f}")
    lines += ["", "**Auto-derived characteristics**:"]

    for ci, cn in enumerate(class_names):
        curve  = mean_curves[:, ci]
        mono   = monotonicity_label(curve, common_grid, x_col)
        pp, tp = peak_trough_pct(curve, common_grid, x_col)
        rng    = float(curve.max() - curve.min())
        spread = class_spread(seed_data, common_grid, ci)
        lines.append(f"- *{cn}*: monotonic={mono}, peak_pct={pp:.0f}, "
                     f"trough_pct={tp:.0f}, range={rng:.3f}, seed_spread={spread:.4f}")

    lines += ["", "**Interpretation** (TO FILL):", "", "---", ""]
    return "\n".join(lines)


# ── Sanity checks ──────────────────────────────────────────────────────────────

def sanity_checks(cfg_list: List[dict]) -> bool:
    ok = True
    sep = "=" * 60
    print(f"\n{sep}\nSANITY CHECKS\n{sep}")

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

    # [2] surviving_concepts in meta.json
    for cfg in cfg_list:
        print(f"\n[2] {cfg['name']} — surviving_concepts:")
        union: set = set()
        for seed in SEEDS:
            mp = pathlib.Path(cfg["checkpoint_dir"]) / f"seed_{seed}" / "meta.json"
            with open(mp) as f:
                meta = json.load(f)
            if "surviving_concepts" not in meta:
                print(f"  FAIL seed_{seed}: missing surviving_concepts key")
                ok = False
                continue
            sc = meta["surviving_concepts"]
            union.update(sc)
            print(f"  seed_{seed}: {len(sc)} -> {sc}")
        print(f"  Union: {len(union)} -> {sorted(union)}")

    # [5] R² CSV
    print(f"\n[5] R² CSV:")
    if not R2_CSV.exists():
        print(f"  FAIL: {R2_CSV} not found")
        ok = False
    else:
        r2df = pd.read_csv(R2_CSV)
        print(f"  OK: {len(r2df)} rows, datasets={r2df['dataset'].unique().tolist()}")

    # [3+4] Model load, forward pass, mean-centring
    print(f"\n[3+4] Model load + mean-centring:")
    for cfg in cfg_list:
        ds = cfg["name"]
        try:
            data = load_dataset(cfg)
        except Exception as e:
            print(f"  [{ds}] FAIL loading data: {e}")
            ok = False
            continue

        X_tf          = data["X_train_final"]
        concept_names = data["concept_names"]

        try:
            model, scaler, meta = load_seed(cfg, 42)
        except Exception as e:
            print(f"  [{ds}] FAIL loading seed_42: {e}")
            ok = False
            continue

        with torch.no_grad():
            X_sc4 = scaler.transform(X_tf[:4].astype(np.float32))
            out   = model(torch.tensor(X_sc4))
        exp = (4, cfg["num_classes"])
        if tuple(out.shape) != exp:
            print(f"  [{ds}] FAIL: forward pass shape {tuple(out.shape)} != {exp}")
            ok = False
        else:
            print(f"  [{ds}] forward pass: {tuple(out.shape)} OK")

        # Mean-centring: f_centred at nearest grid point to x_mean should be ~0
        sc0 = meta["surviving_concepts"][0]
        if sc0 not in concept_names:
            print(f"  [{ds}] WARNING: '{sc0}' not in concept_names")
            continue
        ci0 = concept_names.index(sc0)
        xg, x_col, fc = compute_shape_fn(model, scaler, X_tf, ci0)
        x_mean = float(x_col.mean())
        nearest_i = int(np.argmin(np.abs(xg - x_mean)))
        max_dev   = float(np.abs(fc[nearest_i]).max())
        tol       = 0.10   # tolerance at nearest grid point (not exact x_mean)
        status    = "OK" if max_dev < tol else f"WARN"
        print(f"  [{ds}] mean-centring '{sc0}': |f_cen| at grid nearest to mean "
              f"= {max_dev:.4f} (tol={tol}) -> {status}")
        if max_dev >= tol:
            print(f"         (large deviation may indicate centring not applied)")

    print(f"\n{'All checks passed.' if ok else 'Some checks FAILED - fix before plotting.'}")
    return ok


# ── Process one dataset ────────────────────────────────────────────────────────

def process_dataset(cfg: dict, data: dict,
                    r2_df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    ds_key        = cfg["r2_key"]
    concept_names = data["concept_names"]
    class_names   = data["class_names"]
    colors        = data["class_colors"]
    X_tf          = data["X_train_final"]
    indiv_dir     = OUT_DIR / cfg["out_subdir"] / "individual"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    seed_models  = {}
    seed_scalers = {}
    seed_metas   = {}
    for seed in SEEDS:
        m, s, meta         = load_seed(cfg, seed)
        seed_models[seed]  = m
        seed_scalers[seed] = s
        seed_metas[seed]   = meta

    # Union of surviving concepts in first-appearance order
    seen:  set  = set()
    union: List[str] = []
    for seed in SEEDS:
        for cn in seed_metas[seed]["surviving_concepts"]:
            if cn not in seen:
                union.append(cn)
                seen.add(cn)
    print(f"\n  Union ({len(union)} concepts): {union}")

    shape_rows:    List[dict] = []
    panel_records: List[dict] = []

    for concept in union:
        if concept not in concept_names:
            print(f"  WARNING: '{concept}' not in concept_names — skipping")
            continue

        cidx            = concept_names.index(concept)
        surv_seeds      = [s for s in SEEDS
                           if concept in seed_metas[s]["surviving_concepts"]]
        print(f"  {concept}: {len(surv_seeds)}/{len(SEEDS)} seeds", flush=True)

        x_grids:    List[np.ndarray] = []
        f_centreds: List[np.ndarray] = []
        x_hist:     np.ndarray       = None  # type: ignore[assignment]

        for si, seed in enumerate(surv_seeds):
            xg, xc, fc = compute_shape_fn(seed_models[seed], seed_scalers[seed], X_tf, cidx)
            x_grids.append(xg)
            f_centreds.append(fc)
            if si == 0:
                x_hist = xc

            for ci, cname in enumerate(class_names):
                for gi in range(N_GRID):
                    shape_rows.append({
                        "dataset": ds_key, "concept": concept, "class": cname,
                        "seed": seed, "x_grid_value": float(xg[gi]),
                        "f_value": float(fc[gi, ci]),
                    })

        seed_data = list(zip(x_grids, f_centreds))
        r2_dict   = r2_dict_for(r2_df, ds_key, concept, class_names)

        fig, mean_curves = make_individual_fig(
            concept, class_names, colors, seed_data, x_hist, r2_dict, surv_seeds
        )
        safe = concept.replace(" ", "_")
        fig.savefig(indiv_dir / f"{safe}.png", dpi=150, bbox_inches="tight")
        fig.savefig(indiv_dir / f"{safe}.pdf",            bbox_inches="tight")
        plt.close(fig)
        print(f"    -> individual/{safe}.png  individual/{safe}.pdf")

        panel_records.append({
            "concept_name":  concept,
            "class_names":   class_names,
            "seed_data":     seed_data,
            "x_hist":        x_hist,
            "r2_dict":       r2_dict,
            "surviving_seeds": surv_seeds,
            "common_grid":   x_grids[0],
            "mean_curves":   mean_curves,
        })

    return shape_rows, panel_records


# ── Reporting helpers ──────────────────────────────────────────────────────────

def report_dataset(ds_name: str, panel_records: List[dict]) -> None:
    display = "HAM10000" if ds_name == "ham10000" else "Chest X-ray"
    print(f"\n{display}: {len(panel_records)} panels")
    for rec in panel_records:
        print(f"  {rec['concept_name']}: {len(rec['surviving_seeds'])}/{len(SEEDS)} seeds")

    spread_map: Dict[str, float] = {}
    for rec in panel_records:
        nc  = len(rec["class_names"])
        spr = [class_spread(rec["seed_data"], rec["common_grid"], ci) for ci in range(nc)]
        spread_map[rec["concept_name"]] = float(np.mean(spr))

    top3 = sorted(spread_map, key=spread_map.get, reverse=True)[:3]  # type: ignore[arg-type]
    print(f"\n  High-disagreement (top 3 by seed spread):")
    for cn in top3:
        print(f"    {cn}: spread={spread_map[cn]:.4f}")

    med_spread = float(np.median(list(spread_map.values()))) if spread_map else 0.0
    print(f"\n  Striking panels (mean R²<0.5, spread<median, range>0.15):")
    candidates = []
    for rec in panel_records:
        cn      = rec["concept_name"]
        r2_vals = [v for v in rec["r2_dict"].values() if not np.isnan(v)]
        mean_r2 = float(np.mean(r2_vals)) if r2_vals else float("nan")
        rng     = float(rec["mean_curves"].max() - rec["mean_curves"].min())
        spread  = spread_map[cn]
        if mean_r2 < 0.5 and spread < med_spread and rng > 0.15:
            candidates.append((cn, mean_r2, spread, rng))
    candidates.sort(key=lambda x: (-x[3], x[1]))
    if candidates:
        for cn, r2, sp, rng in candidates:
            print(f"    {cn}: mean_R²={r2:.2f}, spread={sp:.4f}, range={rng:.3f}")
    else:
        print("    (none meeting all criteria)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
         formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sanity_only", action="store_true",
                    help="Run sanity checks and exit without generating plots.")
    ap.add_argument("--dataset", choices=["ham10000", "chestxray"],
                    help="Process only one dataset (default: both).")
    args = ap.parse_args()

    ds_names = [args.dataset] if args.dataset else ["ham10000", "chestxray"]
    cfg_list = [DATASET_CFGS[n] for n in ds_names]

    ok = sanity_checks(cfg_list)
    if not ok:
        sys.exit(1)
    if args.sanity_only:
        print("\n--sanity_only: exiting before plot generation.")
        return

    t0    = time.time()
    r2_df = load_r2_df()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_shape_rows:   List[dict]             = []
    all_panels_by_ds: Dict[str, List[dict]]  = {}

    for cfg in cfg_list:
        ds = cfg["name"]
        print(f"\n{'='*60}\n{ds}\n{'='*60}")
        data = load_dataset(cfg)
        shape_rows, panel_records = process_dataset(cfg, data, r2_df)
        all_shape_rows.extend(shape_rows)
        all_panels_by_ds[ds] = panel_records

        out_root = OUT_DIR / cfg["out_subdir"]
        grid_fig = make_grid_fig(panel_records, data["class_names"], data["class_colors"])
        grid_fig.savefig(out_root / "grid_all.png", dpi=120, bbox_inches="tight")
        grid_fig.savefig(out_root / "grid_all.pdf",            bbox_inches="tight")
        plt.close(grid_fig)
        print(f"\n  Saved {cfg['out_subdir']}/grid_all.png  {cfg['out_subdir']}/grid_all.pdf")

    # shape_function_data.csv
    csv_path = OUT_DIR / "shape_function_data.csv"
    pd.DataFrame(all_shape_rows).to_csv(csv_path, index=False)
    print(f"\nSaved shape_function_data.csv ({len(all_shape_rows)} rows)")

    # interpretation_template.md
    sections = ["# Shape Function Interpretation Template\n"]
    for ds, recs in all_panels_by_ds.items():
        display = "HAM10000" if ds == "ham10000" else "Chest X-ray"
        for rec in recs:
            sections.append(make_section(rec, display))
    (OUT_DIR / "interpretation_template.md").write_text(
        "\n".join(sections), encoding="utf-8"
    )
    print("Saved interpretation_template.md")

    # run_config.json
    elapsed = round(time.time() - t0, 1)
    run_cfg = {
        "datasets":      ds_names,
        "seeds":         SEEDS,
        "n_grid":        N_GRID,
        "grid_pctlo":    GRID_PCTLO,
        "grid_pcthi":    GRID_PCTHI,
        "r2_source":     str(R2_CSV.relative_to(_ROOT)),
        "wall_clock_s":  elapsed,
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print("Saved run_config.json")

    # Final report
    print(f"\n{'='*60}\nREPORT\n{'='*60}")
    for ds, recs in all_panels_by_ds.items():
        report_dataset(ds, recs)

    print(f"\nTotal wall-clock: {elapsed}s")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
