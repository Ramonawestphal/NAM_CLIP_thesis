"""Plot NAM shape functions in the style of Agarwal et al. (2021), Figures 4–8, 10–11.

Each panel shows ensemble member curves (thin blue), the ensemble mean (solid line),
and training-set data density (pink background). Shape functions are mean-centred per
member over ``X_train`` before plotting (paper Section 2.3).

Requires trained ``NAM`` models exposing ``feature_forward(k, x)`` — see ``src.nam.nam.NAM``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.nam.nam import NAM


def evaluate_shape_function(
    ensemble: list[NAM],
    k: int,
    x_grid: np.ndarray,
    X_train: np.ndarray,
) -> np.ndarray:
    """Centred f_k on ``x_grid`` for each ensemble member; shape ``(M, len(x_grid))``."""
    M = len(ensemble)
    G = len(x_grid)
    out = np.zeros((M, G), dtype=np.float32)

    x_train_k = X_train[:, k]

    for m, model in enumerate(ensemble):
        f_grid = model.feature_forward(k, x_grid).detach().cpu().numpy()
        f_train = model.feature_forward(k, x_train_k).detach().cpu().numpy()
        out[m] = f_grid - f_train.mean()

    return out


def feature_importance(ensemble: list[NAM], X_train: np.ndarray) -> np.ndarray:
    """Mean |centred f_k| over training rows, averaged across ensemble members; shape ``(K,)``."""
    _, K = X_train.shape
    imp = np.zeros(K, dtype=np.float64)
    for k in range(K):
        x_k = X_train[:, k]
        per_model = []
        for model in ensemble:
            f = model.feature_forward(k, x_k).detach().cpu().numpy()
            f_centred = f - f.mean()
            per_model.append(np.mean(np.abs(f_centred)))
        imp[k] = np.mean(per_model)
    return imp


def infer_binary_feature_indices(
    X_train: np.ndarray,
    max_unique: int = 2,
) -> set[int]:
    """Column indices with at most ``max_unique`` distinct training values (e.g. OHE slots)."""
    cats: set[int] = set()
    for k in range(X_train.shape[1]):
        u = np.unique(X_train[:, k])
        if len(u) <= max_unique:
            cats.add(k)
    return cats


def compas_categorical_indices(feature_names: list[str]) -> set[int]:
    """Indices of race/sex OHE columns after ``CompasEncoder`` (names ``race_*``, ``sex_*``)."""
    return {
        i
        for i, name in enumerate(feature_names)
        if name.startswith("race_") or name.startswith("sex_")
    }


def draw_density_background(
    ax,
    x_train_k: np.ndarray,
    n_bins: int = 30,
    is_categorical: bool = False,
) -> None:
    """Pink vertical bars; alpha scales with bin count (drawn behind shape curves)."""
    pink = "#E89BB0"

    if is_categorical:
        vals, counts = np.unique(x_train_k, return_counts=True)
        max_c = counts.max()
        if len(vals) > 1:
            half = 0.5 * np.min(np.diff(np.sort(vals.astype(float))))
        else:
            half = 0.5
        for v, c in zip(vals, counts):
            ax.axvspan(
                float(v) - half,
                float(v) + half,
                ymin=0,
                ymax=1,
                color=pink,
                alpha=0.15 + 0.75 * (c / max_c),
                linewidth=0,
                zorder=0,
            )
    else:
        counts, edges = np.histogram(x_train_k, bins=n_bins)
        max_c = max(int(counts.max()), 1)
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            if c == 0:
                continue
            ax.axvspan(
                lo,
                hi,
                ymin=0,
                ymax=1,
                color=pink,
                alpha=0.15 + 0.75 * (c / max_c),
                linewidth=0,
                zorder=0,
            )


def plot_shape_function_panel(
    ax,
    f_ensemble: np.ndarray,
    x_grid: np.ndarray,
    x_train_k: np.ndarray,
    feature_name: str,
    y_label: str,
    is_categorical: bool = False,
    y_clip: tuple[float, float] | None = None,
) -> None:
    """One feature panel: density background, ensemble lines, mean curve, zero reference."""
    mean_curve = f_ensemble.mean(axis=0)

    if y_clip is not None:
        y_lo, y_hi = y_clip
    else:
        lo, hi = np.percentile(f_ensemble, [1, 99])
        pad = 0.1 * (hi - lo + 1e-6)
        y_lo, y_hi = lo - pad, hi + pad
    ax.set_ylim(y_lo, y_hi)

    draw_density_background(ax, x_train_k, is_categorical=is_categorical)

    if is_categorical:
        vals = np.unique(x_train_k)
        for m in range(f_ensemble.shape[0]):
            ax.scatter(
                vals,
                f_ensemble[m],
                s=8,
                color="steelblue",
                alpha=0.15,
                zorder=2,
            )
        ax.scatter(
            vals,
            mean_curve,
            s=40,
            color="C0",
            zorder=3,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_xticks(vals)
    else:
        for m in range(f_ensemble.shape[0]):
            ax.plot(
                x_grid,
                f_ensemble[m],
                color="steelblue",
                alpha=0.15,
                linewidth=0.8,
                zorder=2,
            )
        ax.plot(x_grid, mean_curve, color="C0", linewidth=2.0, zorder=3)

    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4, zorder=1)
    ax.set_xlabel(feature_name)
    ax.set_ylabel(y_label)
    ax.set_xlim(float(x_grid.min()), float(x_grid.max()))


def plot_all_shape_functions(
    ensemble: list[NAM],
    X_train: np.ndarray,
    feature_names: list[str],
    task: str = "binary",
    top_k: int | None = None,
    n_cols: int = 4,
    grid_size: int = 200,
    categorical_features: set[int] | None = None,
    shared_ylim: bool = True,
    save_path: str | None = None,
):
    """
    Grid of shape-function panels plus a feature-importance table.

    Parameters
    ----------
    ensemble
        List of trained ``NAM`` models (e.g. 20–100 CV resamples).
    X_train
        Training matrix ``(N, K)`` in the same encoding used at train time.
    feature_names
        Human-readable names, length ``K``.
    task
        ``"binary"`` → y-label “Log-odds contribution”; ``"regression"`` → “Target contribution”.
    top_k
        If set, plot only the top-k features by importance.
    categorical_features
        Column indices drawn as discrete scatter panels (OHE slots). If ``None``, columns
        with ≤2 unique training values are treated as categorical.
  """
    _, K = X_train.shape
    if categorical_features is None:
        categorical_features = infer_binary_feature_indices(X_train)

    imp = feature_importance(ensemble, X_train)
    order = np.argsort(-imp)
    if top_k is not None:
        order = order[:top_k]

    importance_df = pd.DataFrame(
        {
            "feature": [feature_names[i] for i in np.argsort(-imp)],
            "importance": imp[np.argsort(-imp)],
        }
    )

    panels: list[dict] = []
    for k in order:
        is_cat = k in categorical_features
        x_k = X_train[:, k]
        if is_cat:
            x_grid = np.unique(x_k)
        else:
            lo, hi = float(np.min(x_k)), float(np.max(x_k))
            x_grid = np.linspace(lo, hi, grid_size)
        f_ens = evaluate_shape_function(ensemble, k, x_grid, X_train)
        panels.append(
            dict(k=k, x_grid=x_grid, f_ens=f_ens, x_train_k=x_k, is_cat=is_cat)
        )

    if shared_ylim and panels:
        all_vals = np.concatenate([p["f_ens"].ravel() for p in panels])
        lo, hi = np.percentile(all_vals, [1, 99])
        pad = 0.1 * (hi - lo + 1e-6)
        y_clip = (lo - pad, hi + pad)
    else:
        y_clip = None

    n_panels = len(panels)
    n_rows = int(np.ceil(n_panels / n_cols)) if n_panels else 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )

    y_label = (
        "Log-odds contribution" if task == "binary" else "Target contribution"
    )

    for ax, p in zip(axes.ravel(), panels):
        plot_shape_function_panel(
            ax=ax,
            f_ensemble=p["f_ens"],
            x_grid=p["x_grid"],
            x_train_k=p["x_train_k"],
            feature_name=feature_names[p["k"]],
            y_label=y_label,
            is_categorical=p["is_cat"],
            y_clip=y_clip,
        )

    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, importance_df
