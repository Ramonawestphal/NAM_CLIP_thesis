#!/usr/bin/env python
"""
Curated appendix grids for HAM10000 pairwise shape function plots.

Reuses computation/rendering from plot_shape_functions_pairwise.py.
Run from project root:
    python scripts/analysis/plot_shape_functions_pairwise_selection.py
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.analysis.plot_shape_functions_pairwise import (
    DATASET_CFGS,
    OUT_DIR,
    SEEDS,
    compute_pairwise_diff,
    diff_spread,
    draw_pairwise_panel,
    load_dataset,
    load_seed,
    pair_dir_name,
    resolve_class_cols,
)

SELECTIONS: Dict[str, Dict[str, List[str]]] = {
    "mel_vs_nv": {
        "class_a": "mel",
        "class_b": "nv",
        "concepts": [
            "atypical_pigment_network",
            "asymmetry",
            "central_white_patch",
            "red_lacunae",
            "scaly_surface",
        ],
    },
    "bcc_vs_nv": {
        "class_a": "bcc",
        "class_b": "nv",
        "concepts": [
            "asymmetry",
            "scaly_surface",
            "milia_like_cysts",
            "central_white_patch",
            "red_lacunae",
        ],
    },
}

MIN_SURVIVING_SEEDS = 2


def build_panel_record(
    concept: str,
    class_a: str,
    class_b: str,
    data: dict,
    seed_models: dict,
    seed_scalers: dict,
    seed_metas: dict,
) -> dict:
    concept_names = data["concept_names"]
    class_names = data["class_names"]
    X_tf = data["X_train_final"]
    col_a, col_b = resolve_class_cols(class_names, class_a, class_b)

    if concept not in concept_names:
        raise KeyError(f"Concept '{concept}' not in concept_names")

    cidx = concept_names.index(concept)
    surv_seeds = [s for s in SEEDS if concept in seed_metas[s]["surviving_concepts"]]

    x_grids: List = []
    diff_grids: List = []
    r2_vals: List[float] = []
    x_hist = None

    for si, seed in enumerate(surv_seeds):
        xg, xc, diff_g, r2 = compute_pairwise_diff(
            seed_models[seed], seed_scalers[seed], X_tf, cidx, col_a, col_b,
        )
        x_grids.append(xg)
        diff_grids.append(diff_g)
        r2_vals.append(r2)
        if si == 0:
            x_hist = xc

    seed_data = list(zip(x_grids, diff_grids))
    common_grid = x_grids[0]
    mean_r2 = float(sum(r2_vals) / len(r2_vals)) if r2_vals else float("nan")

    on_common = [np.interp(common_grid, xg, dg) for xg, dg in seed_data]
    mean_diff = np.array(on_common).mean(axis=0)

    return {
        "dataset": "ham10000",
        "class_a": class_a,
        "class_b": class_b,
        "concept_name": concept,
        "seed_data": seed_data,
        "x_hist": x_hist,
        "mean_r2": mean_r2,
        "surviving_seeds": surv_seeds,
        "common_grid": common_grid,
        "mean_diff": mean_diff,
        "seed_spread": diff_spread(seed_data, common_grid),
    }


def make_selection_grid_fig(
    panel_records: List[dict],
    class_a: str,
    class_b: str,
    n_cols: int = 3,
) -> plt.Figure:
    n = len(panel_records)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig = plt.figure(figsize=(4.5 * n_cols, 3.8 * n_rows + 0.6))
    fig.suptitle(
        f"Pairwise comparison: log P({class_a}) / P({class_b})",
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
        draw_pairwise_panel(
            ax, axh, rec["concept_name"], rec["class_a"], rec["class_b"],
            rec["seed_data"], rec["x_hist"], rec["mean_r2"],
            rec["surviving_seeds"], small=True,
        )
        n_s = len(rec["surviving_seeds"])
        title = f"{rec['concept_name'].replace('_', ' ')} [n={n_s}/{len(SEEDS)}]"
        ax.set_title(title, fontsize=8)
    return fig


def main() -> None:
    cfg = DATASET_CFGS["ham10000"]
    data = load_dataset(cfg)

    seed_models = {}
    seed_scalers = {}
    seed_metas = {}
    for seed in SEEDS:
        m, s, meta = load_seed(cfg, seed)
        seed_models[seed] = m
        seed_scalers[seed] = s
        seed_metas[seed] = meta

    for key, spec in SELECTIONS.items():
        class_a = spec["class_a"]
        class_b = spec["class_b"]
        pair_name = pair_dir_name(class_a, class_b)
        print(f"\nBuilding selection grid: {pair_name} ({len(spec['concepts'])} concepts)")

        panels = []
        for concept in spec["concepts"]:
            rec = build_panel_record(
                concept, class_a, class_b, data,
                seed_models, seed_scalers, seed_metas,
            )
            n_surv = len(rec["surviving_seeds"])
            if n_surv < MIN_SURVIVING_SEEDS:
                print(f"  {concept}: {n_surv}/{len(SEEDS)} seeds — skipped (<{MIN_SURVIVING_SEEDS})")
                continue
            panels.append(rec)
            print(
                f"  {concept}: {n_surv}/{len(SEEDS)} seeds, "
                f"mean R²={rec['mean_r2']:.2f}"
            )

        fig = make_selection_grid_fig(panels, class_a, class_b)
        out_dir = OUT_DIR / cfg["out_subdir"] / pair_name
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"grid_selection_appendix_{pair_name}"
        fig.savefig(out_dir / f"{stem}.png", dpi=120, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir.relative_to(_ROOT)}/{stem}.png")
        print(f"  Saved {out_dir.relative_to(_ROOT)}/{stem}.pdf")


if __name__ == "__main__":
    main()
