#!/usr/bin/env python
"""
Curated appendix grid for Chest X-ray multiclass shape function plots.

Uses the full per-class shape function panels from plot_shape_functions.py
(not pairwise differences). Run from project root:
    python scripts/analysis/plot_shape_functions_chestxray_selection.py
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.analysis.plot_shape_functions import (
    DATASET_CFGS,
    OUT_DIR,
    SEEDS,
    compute_shape_fn,
    draw_panel,
    load_dataset,
    load_r2_df,
    load_seed,
    r2_dict_for,
)

CHESTXRAY_CONCEPTS: List[str] = [
    "focal_opacity",
    "patchy_infiltrate",
    "parapneumonic_effusion",
    "peribronchial_cuffing",
    "pleural_effusion",
    "clear_lung_fields",
    "symmetric_lung_aeration",
    "round_pneumonia",
]


def build_panel_record(
    concept: str,
    data: dict,
    cfg: dict,
    r2_df,
    seed_models: dict,
    seed_scalers: dict,
    seed_metas: dict,
) -> dict:
    concept_names = data["concept_names"]
    X_tf = data["X_train_final"]

    if concept not in concept_names:
        raise KeyError(f"Concept '{concept}' not in concept_names")

    cidx = concept_names.index(concept)
    surv_seeds = [s for s in SEEDS if concept in seed_metas[s]["surviving_concepts"]]
    plot_seeds = surv_seeds if surv_seeds else list(SEEDS)

    x_grids = []
    f_centreds = []
    x_hist = None
    for si, seed in enumerate(plot_seeds):
        xg, xc, fc = compute_shape_fn(seed_models[seed], seed_scalers[seed], X_tf, cidx)
        x_grids.append(xg)
        f_centreds.append(fc)
        if si == 0:
            x_hist = xc

    return {
        "concept_name": concept,
        "seed_data": list(zip(x_grids, f_centreds)),
        "x_hist": x_hist,
        "r2_dict": r2_dict_for(r2_df, cfg["r2_key"], concept, data["class_names"]),
        "surviving_seeds": surv_seeds,
        "plot_seeds": plot_seeds,
    }


def _fix_y_axis(ax) -> None:
    """Avoid matplotlib offset text (e.g. 1e-5) on near-zero shape functions."""
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.get_offset_text().set_visible(False)
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span < 0.08:
        mid = 0.5 * (ymin + ymax)
        half = max(0.04, 0.5 * span + 0.02)
        ax.set_ylim(mid - half, mid + half)


def make_selection_grid_fig(
    panel_records: List[dict],
    class_names: List[str],
    colors: Dict[str, str],
    n_cols: int = 3,
) -> plt.Figure:
    n = len(panel_records)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig = plt.figure(figsize=(4.5 * n_cols, 3.8 * n_rows + 0.6))
    fig.suptitle(
        "Chest X-ray — multiclass shape function contributions (normal, bacteria, virus)",
        fontsize=14,
        y=0.995,
    )
    outer = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.60, wspace=0.38,
        top=0.94, bottom=0.05,
    )
    for pi, rec in enumerate(panel_records):
        row, col_i = divmod(pi, n_cols)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col_i], height_ratios=[5, 1], hspace=0.10,
        )
        ax = fig.add_subplot(inner[0])
        axh = fig.add_subplot(inner[1], sharex=ax)
        draw_panel(
            ax, axh, rec["concept_name"], class_names, colors,
            rec["seed_data"], rec["x_hist"], rec["r2_dict"],
            rec["surviving_seeds"], small=True,
        )
        n_s = len(rec["surviving_seeds"])
        title = f"{rec['concept_name'].replace('_', ' ')} [n={n_s}/{len(SEEDS)}]"
        ax.set_title(title, fontsize=8)
        _fix_y_axis(ax)
    return fig


def main() -> None:
    cfg = DATASET_CFGS["chestxray"]
    data = load_dataset(cfg)
    r2_df = load_r2_df()

    seed_models = {}
    seed_scalers = {}
    seed_metas = {}
    for seed in SEEDS:
        m, s, meta = load_seed(cfg, seed)
        seed_models[seed] = m
        seed_scalers[seed] = s
        seed_metas[seed] = meta

    panels = []
    print(f"Building Chest X-ray selection grid ({len(CHESTXRAY_CONCEPTS)} concepts)")
    for concept in CHESTXRAY_CONCEPTS:
        rec = build_panel_record(
            concept, data, cfg, r2_df, seed_models, seed_scalers, seed_metas,
        )
        panels.append(rec)
        n_surv = len(rec["surviving_seeds"])
        flag = "" if n_surv else " [not in K=10 for any seed — plotted from all seeds]"
        print(f"  {concept}: {n_surv}/{len(SEEDS)} seeds at K=10{flag}")

    fig = make_selection_grid_fig(
        panels, data["class_names"], data["class_colors"],
    )
    out_dir = OUT_DIR / cfg["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "grid_selection_appendix"
    fig.savefig(out_dir / f"{stem}.png", dpi=120, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_dir.relative_to(_ROOT)}/{stem}.png")
    print(f"Saved {out_dir.relative_to(_ROOT)}/{stem}.pdf")


if __name__ == "__main__":
    main()
