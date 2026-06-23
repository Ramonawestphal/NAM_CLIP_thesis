"""
Mean-across-seeds shape function plots for NAM ablation conditions.

Usage (from project root):
  python scripts/plot_shape_functions_final.py \
      --config_name=plain_nam \
      --config_dir=reports/nam/v6_sparsity_sweep/lam_0.0 \
      --out_dir=results/final_models/plain_nam

  python scripts/plot_shape_functions_final.py \
      --config_name=concurvity_only_lc1 \
      --config_dir=reports/nam/v6_final \
      --out_dir=results/final_models/concurvity_only_lc1

  python scripts/plot_shape_functions_final.py \
      --config_name=sparsity_only_lc0 \
      --config_dir=results/final_models/sparsity_only_lc0 \
      --out_dir=results/final_models/sparsity_only_lc0

  python scripts/plot_shape_functions_final.py \
      --config_name=sparsity_conc_lc1 \
      --config_dir=results/final_models/sparsity_conc_lc1 \
      --out_dir=results/final_models/sparsity_conc_lc1

Output: {out_dir}/shape_functions_mean_seeds.png
        {out_dir}/active_concepts.txt
"""

from __future__ import annotations

import argparse
import ast
import os
import pickle
import pathlib
import sys
import warnings
import glob

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models.nam_multiclass import NAMMulticlass

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES_PATH = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH   = "data/splits/train_test_lesion_split.npz"
SEEDS         = [42, 43, 44, 45, 46]
N_FEATURES    = 24
N_CLASSES     = 7
GRID_POINTS   = 200
MAJORITY      = 3   # out of 5 seeds to call a concept "active"
ZERO_THR      = 1e-8

# Color palette from inspect_shape_functions_v6_final.py — shared across all conditions.
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_COLORS = {
    "akiec": "#E69F00", "bcc": "#56B4E9", "bkl": "#009E73",
    "df":    "#F0E442", "mel": "#D55E00", "nv":  "#0072B2", "vasc": "#CC79A7",
}


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config_name", required=True,
                   help="Short label for this condition (e.g. plain_nam)")
    p.add_argument("--config_dir",  required=True,
                   help="Directory that contains seed checkpoints")
    p.add_argument("--out_dir",     default=None,
                   help="Output directory for plots (defaults to --config_dir)")
    return p.parse_args()


# ── Checkpoint discovery ───────────────────────────────────────────────────────
def find_checkpoint(config_dir: str, seed: int) -> str | None:
    """Try both old (seed_{s}) and new (seed{s}/seed_{s}) path patterns."""
    candidates = [
        os.path.join(config_dir, f"seed_{seed}", "best_model.pt"),
        os.path.join(config_dir, f"seed{seed}", f"seed_{seed}", "best_model.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_norms_csv(config_dir: str, seed: int) -> str | None:
    candidates = [
        os.path.join(config_dir, f"seed_{seed}", "feature_group_norms.csv"),
        os.path.join(config_dir, f"seed{seed}", f"seed_{seed}", "feature_group_norms.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_config_csv(config_dir: str, seed: int) -> str | None:
    """winning_config.csv may be at config_dir root or per-seed subdirectory."""
    candidates = [
        os.path.join(config_dir, "winning_config.csv"),
        os.path.join(config_dir, f"seed{seed}", "winning_config.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_scaler(config_dir: str, seed: int) -> str | None:
    candidates = [
        os.path.join(config_dir, "scaler.pkl"),
        os.path.join(config_dir, f"seed{seed}", "scaler.pkl"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ── Helpers ────────────────────────────────────────────────────────────────────
def group_norm_from_model(model: NAMMulticlass) -> dict[str, float]:
    """Compute L2 group norm per sub-network (same logic as sparsity.py)."""
    norms: dict[str, float] = {}
    for name, subnet in model.feature_subnetworks():
        params = [p for p in subnet.parameters() if p.requires_grad]
        sq = sum(p.pow(2).sum().item() for p in params)
        norms[name] = float(sq ** 0.5)
    return norms


def load_model(ckpt_path: str, hidden_dims: tuple, dropout: float,
               concept_names: list[str]) -> NAMMulticlass:
    model = NAMMulticlass(
        n_features=N_FEATURES, num_classes=N_CLASSES,
        hidden_dims=hidden_dims, dropout=dropout,
        concept_names=concept_names,
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    config_dir  = args.config_dir
    config_name = args.config_name
    out_dir     = args.out_dir if args.out_dir else config_dir
    os.makedirs(out_dir, exist_ok=True)

    # -- Load features and training data
    print("Loading features and splits...")
    feat            = np.load(FEATURES_PATH, allow_pickle=True)
    scores_raw      = feat["scores"]
    concept_names   = list(feat["concept_ids"])
    train_idx       = np.load(SPLITS_PATH)["train_idx"]
    X_train_raw     = scores_raw[train_idx]

    # -- Discover architecture from winning_config.csv
    cfg_path = find_config_csv(config_dir, SEEDS[0])
    if cfg_path is None:
        raise FileNotFoundError(
            f"winning_config.csv not found under {config_dir}. "
            "Run training first."
        )
    cfg = pd.read_csv(cfg_path).iloc[0]
    hidden_dims = tuple(ast.literal_eval(cfg["hidden_dims"]))
    dropout     = float(cfg["dropout"])
    print(f"Architecture: hidden={list(hidden_dims)}, dropout={dropout}")

    # -- Load scaler (all seeds use the same val split, so scaler is identical)
    scaler_path = find_scaler(config_dir, SEEDS[0])
    if scaler_path is None:
        raise FileNotFoundError(f"scaler.pkl not found under {config_dir}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_train_sc = scaler.transform(X_train_raw).astype(np.float32)
    print(f"Scaler loaded from: {scaler_path}")

    # -- Load each seed's checkpoint and compute active-concept status
    models: list[NAMMulticlass] = []
    active_per_seed: list[list[bool]] = []   # [seed_idx][concept_idx] → bool

    for seed in SEEDS:
        ckpt = find_checkpoint(config_dir, seed)
        if ckpt is None:
            raise FileNotFoundError(
                f"best_model.pt not found for seed {seed} under {config_dir}"
            )
        model = load_model(ckpt, hidden_dims, dropout, concept_names)
        models.append(model)

        # Prefer saved norms CSV; fall back to computing from loaded model.
        norms_path = find_norms_csv(config_dir, seed)
        if norms_path and os.path.exists(norms_path):
            norms_df = pd.read_csv(norms_path)
            norms = dict(zip(norms_df["concept_name"], norms_df["group_norm"]))
        else:
            norms = group_norm_from_model(model)

        seed_active = [norms.get(c, 0.0) > ZERO_THR for c in concept_names]
        active_per_seed.append(seed_active)
        n_act = sum(seed_active)
        print(f"  seed {seed}: {n_act}/{N_FEATURES} active  ({ckpt})")

    # -- Majority-rule active concepts (active in >= 3/5 seeds)
    active_majority = [
        sum(active_per_seed[si][ci] for si in range(len(SEEDS))) >= MAJORITY
        for ci in range(N_FEATURES)
    ]
    active_names     = [concept_names[i] for i in range(N_FEATURES) if     active_majority[i]]
    eliminated_names = [concept_names[i] for i in range(N_FEATURES) if not active_majority[i]]
    print(f"\nMajority-active ({MAJORITY}/5 seeds): {len(active_names)} concepts")
    print(f"  Active     : {active_names}")
    print(f"  Eliminated : {eliminated_names}")

    # Save active list
    with open(os.path.join(out_dir, "active_concepts.txt"), "w") as f:
        f.write(f"# Config: {config_name}\n")
        f.write(f"# Majority threshold: >= {MAJORITY}/5 seeds with group_norm > {ZERO_THR}\n\n")
        f.write("ACTIVE:\n")
        for n in active_names:
            f.write(f"  {n}\n")
        f.write("\nELIMINATED:\n")
        for n in eliminated_names:
            f.write(f"  {n}\n")
        f.write(f"\nPer-seed active counts:\n")
        for si, seed in enumerate(SEEDS):
            cnt = sum(active_per_seed[si])
            f.write(f"  seed {seed}: {cnt}/{N_FEATURES}\n")

    # -- Compute per-seed shape functions on shared grids
    # grids[concept_idx] = (200,) float32 array
    # shapes[concept_idx] = (5, 200, 7) float32 array — seeds × grid × classes
    grids:  list[np.ndarray] = []
    shapes: list[np.ndarray] = []

    print("\nComputing shape functions across 5 seeds...")
    for ci, cname in enumerate(concept_names):
        lo, hi = np.percentile(X_train_sc[:, ci], [1, 99])
        grid   = np.linspace(lo, hi, GRID_POINTS).astype(np.float32)
        seed_fns = []
        for model in models:
            fn = model.concept_contributions(grid, ci).numpy()  # (200, 7)
            seed_fns.append(fn)
        grids.append(grid)
        shapes.append(np.stack(seed_fns, axis=0))  # (5, 200, 7)

    # ── Build figure ──────────────────────────────────────────────────────────
    n_active     = len(active_names)
    n_eliminated = len(eliminated_names)

    # Layout: active concepts in 4-column grid, then eliminated below
    n_act_rows   = (n_active + 3) // 4       # rows for active section
    n_elim_rows  = (n_eliminated + 3) // 4   # rows for eliminated section (may be 0)
    n_rows_total = n_act_rows + n_elim_rows + (1 if n_eliminated > 0 else 0)

    COLS    = 4
    FIG_W   = 5.5 * COLS
    FIG_H   = 3.5 * n_act_rows + 3.0 * n_elim_rows + 1.5   # extra for legend
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Title
    fig.suptitle(
        f"{config_name}  —  Shape functions (mean ± 95% CI, n=5 seeds)\n"
        f"Active: {n_active}  |  Eliminated: {n_eliminated}  |  "
        f"hidden={list(hidden_dims)}, dropout={dropout}",
        fontsize=13, y=1.0,
    )

    # --- Active concept panels ---
    act_axes: list[plt.Axes] = []
    for rank, cname in enumerate(active_names):
        ci   = concept_names.index(cname)
        row  = rank // COLS
        col  = rank % COLS
        ax   = fig.add_subplot(n_rows_total, COLS,
                               row * COLS + col + 1)
        grid = grids[ci]
        shp  = shapes[ci]  # (5, 200, 7)

        mean = shp.mean(axis=0)      # (200, 7)
        std  = shp.std(axis=0)       # (200, 7)
        se   = std / np.sqrt(len(SEEDS))
        ci95 = 1.96 * se

        ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)

        for cls_idx, cls in enumerate(CLASS_NAMES):
            color = CLASS_COLORS[cls]
            m     = mean[:, cls_idx]
            lo_b  = m - ci95[:, cls_idx]
            hi_b  = m + ci95[:, cls_idx]
            ax.plot(grid, m, color=color, linewidth=1.5, label=cls)
            ax.fill_between(grid, lo_b, hi_b, color=color, alpha=0.15, linewidth=0)

        # active-in-how-many-seeds indicator
        n_seeds_active = sum(active_per_seed[si][ci] for si in range(len(SEEDS)))
        ax.set_title(f"{cname}\n({n_seeds_active}/5 seeds)", fontsize=8)
        ax.set_xlabel(f"{cname} (z-scored)", fontsize=7)
        ax.set_ylabel("Logit contribution", fontsize=7)
        ax.tick_params(labelsize=6)
        act_axes.append(ax)

    # Blank any leftover cells in the active section
    n_act_cells = n_act_rows * COLS
    for blank_pos in range(n_active, n_act_cells):
        row = blank_pos // COLS
        col = blank_pos % COLS
        ax = fig.add_subplot(n_rows_total, COLS, row * COLS + col + 1)
        ax.set_visible(False)

    # --- Divider label row (if there are eliminated concepts) ---
    if n_eliminated > 0:
        divider_row = n_act_rows
        ax_div = fig.add_subplot(n_rows_total, 1, divider_row + 1)
        ax_div.set_facecolor("#f0f0f0")
        ax_div.text(0.5, 0.5, "Eliminated (not used by model)",
                    ha="center", va="center", fontsize=10,
                    style="italic", color="#555555",
                    transform=ax_div.transAxes)
        ax_div.set_xticks([])
        ax_div.set_yticks([])
        for spine in ax_div.spines.values():
            spine.set_visible(False)

        # --- Eliminated concept panels (grayed out, horizontal y=0 line) ---
        for rank, cname in enumerate(eliminated_names):
            ci   = concept_names.index(cname)
            row  = n_act_rows + 1 + rank // COLS
            col  = rank % COLS
            panel_pos = row * COLS + col + 1

            # Subplot index in the full grid counts divider as one row
            ax = fig.add_subplot(n_rows_total, COLS, panel_pos)
            grid = grids[ci]

            ax.axhline(0, color="#aaaaaa", linewidth=1.2)
            ax.text(0.5, 0.5, "ELIMINATED",
                    ha="center", va="center", fontsize=8,
                    color="#888888", transform=ax.transAxes, style="italic")
            ax.set_facecolor("#f9f9f9")
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
            ax.set_title(cname, fontsize=8, color="#666666")
            ax.set_xlabel(f"{cname} (z-scored)", fontsize=7, color="#888888")
            ax.set_ylabel("Logit contribution", fontsize=7, color="#888888")
            ax.tick_params(labelsize=6, colors="#aaaaaa")

        # Blank leftover eliminated cells
        n_elim_cells = n_elim_rows * COLS
        for blank_pos in range(n_eliminated, n_elim_cells):
            row = n_act_rows + 1 + blank_pos // COLS
            col = blank_pos % COLS
            ax = fig.add_subplot(n_rows_total, COLS, row * COLS + col + 1)
            ax.set_visible(False)

    # --- Shared legend at top ---
    handles = [
        plt.Line2D([0], [0], color=CLASS_COLORS[cls], linewidth=2.0, label=cls)
        for cls in CLASS_NAMES
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=len(CLASS_NAMES),
        fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, 1.0),
        title="Class",
        title_fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(out_dir, "shape_functions_mean_seeds.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_png}")

    # -- Summary text
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  Active (majority rule, >= {MAJORITY}/5 seeds): {len(active_names)}")
    for n in active_names:
        ci = concept_names.index(n)
        cnt = sum(active_per_seed[si][ci] for si in range(len(SEEDS)))
        print(f"    {n:35s} ({cnt}/5 seeds)")
    if eliminated_names:
        print(f"  Eliminated: {len(eliminated_names)}")
        for n in eliminated_names:
            ci = concept_names.index(n)
            cnt = sum(active_per_seed[si][ci] for si in range(len(SEEDS)))
            print(f"    {n:35s} ({cnt}/5 seeds)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
