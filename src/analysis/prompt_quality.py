"""Five prompt quality analyses for HAM10000 CLIP concept scores.

Each function takes numpy arrays from the training split and returns a
pandas DataFrame. Figures are saved as side effects where noted.

Re-run Analysis 3 (OvR AUC) on the v4 score matrix from the repo root::

    python -m src.analysis.prompt_quality --version v4

Or import :func:`run_analysis3_ovr_auc` with a custom ``scores_npz`` path.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

_ROOT = pathlib.Path(__file__).resolve().parents[2]

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Default score bundles (written by scripts/extract_features.py / iterate_prompts.py)
SCORES_V1_NPZ = _ROOT / "data/features/ham10000_concept_scores.npz"
SCORES_V4_NPZ = _ROOT / "data/features/ham10000_concept_scores_v4.npz"
DEFAULT_SPLIT_NPZ = _ROOT / "data/splits/train_test_lesion_split.npz"
DEFAULT_REPORT_DIR = _ROOT / "reports/prompt_analysis"

# Paired contrasts for analysis 5 (positive_concept, negative_concept, target_class).
# Use "__all_others__" to compare against the mean of every other t1 prompt.
_CONTRASTS = [
    ("asymmetry",              "symmetric_uniform",       "mel"),
    ("border_irregularity",    "symmetric_uniform",       "mel"),
    ("colour_variation",       "symmetric_uniform",       "mel"),
    ("atypical_pigment_network", "reticular_network",     "mel"),
    ("red_lacunae",            "__all_others__",          "vasc"),
    ("arborizing_vessels",     "reticular_network",       "bcc"),
    ("scaly_surface",          "symmetric_uniform",       "akiec"),
    ("milia_like_cysts",       "atypical_pigment_network","bkl"),
]


def _versioned_stem(stem: str, version: str | None) -> str:
    return f"{stem}_{version}" if version else stem


def build_prompt_meta_from_npz(data: np.lib.npyio.NpzFile) -> pd.DataFrame:
    """Build the 72-row prompt metadata table from an NPZ written by extract/iterate."""
    concept_ids = list(data["concept_ids"])
    prompts = list(data["prompts"])
    prompt_concept_idx = data["prompt_concept_idx"]
    prompt_template_idx = data["prompt_template_idx"]
    tiers = data["tiers"]
    tmpl_name = {0: "t1", 1: "t2", 2: "t3"}
    return pd.DataFrame({
        "prompt_idx": np.arange(len(prompts)),
        "concept_id": [concept_ids[i] for i in prompt_concept_idx],
        "template": [tmpl_name[i] for i in prompt_template_idx],
        "prompt": prompts,
        "concept_idx": list(prompt_concept_idx),
        "tier": [int(tiers[i]) for i in prompt_concept_idx],
    })


def load_train_scores_and_meta(
    scores_npz: str | pathlib.Path,
    split_npz: str | pathlib.Path = DEFAULT_SPLIT_NPZ,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load full score matrix and return training-split ``scores``, ``labels``, ``meta``."""
    scores_npz = pathlib.Path(scores_npz)
    split_npz = pathlib.Path(split_npz)
    if not scores_npz.is_file():
        raise FileNotFoundError(f"Scores NPZ not found: {scores_npz}")
    if not split_npz.is_file():
        raise FileNotFoundError(
            f"Split NPZ not found: {split_npz}\n"
            "Run scripts/analyze_prompts.py once to create the lesion split."
        )

    data = np.load(scores_npz, allow_pickle=True)
    split = np.load(split_npz)
    train_idx = split["train_idx"]

    scores = data["scores"][train_idx]
    labels = data["labels"][train_idx]
    meta = build_prompt_meta_from_npz(data)
    return scores, labels, meta


def run_analysis3_ovr_auc(
    scores_npz: str | pathlib.Path | None = None,
    *,
    version: str | None = "v4",
    split_npz: str | pathlib.Path = DEFAULT_SPLIT_NPZ,
    output_dir: str | pathlib.Path = DEFAULT_REPORT_DIR,
) -> pd.DataFrame:
    """Load a score NPZ, run Analysis 3 on the training split, save CSV + bar plot.

    Parameters
    ----------
    scores_npz
        Path to ``ham10000_concept_scores*.npz``. Defaults to v4 when
        ``version='v4'``, else the v1 bundle.
    version
        Suffix for outputs (e.g. ``prompt_ovr_auc_v4.csv``). Pass ``None`` for
        unversioned filenames (same as the original analyze_prompts run).
    """
    if scores_npz is None:
        scores_npz = SCORES_V4_NPZ if version == "v4" else SCORES_V1_NPZ

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_train, labels_train, meta = load_train_scores_and_meta(scores_npz, split_npz)
    print(
        f"Analysis 3: OvR AUC on training split "
        f"({len(scores_train):,} images, {scores_train.shape[1]} prompts)"
    )
    print(f"  scores: {pathlib.Path(scores_npz).relative_to(_ROOT)}")

    return analysis3_ovr_auc(
        scores_train,
        labels_train,
        meta,
        output_dir,
        version=version,
    )


# ---------------------------------------------------------------------------
# Analysis 1: Score range and distribution per prompt
# ---------------------------------------------------------------------------

def analysis1_score_distribution(
    scores: np.ndarray,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Min/max/mean/std/p05/p95 per prompt; flag low-variation prompts."""
    rows = []
    for _, row in meta.iterrows():
        col = scores[:, row["prompt_idx"]]
        p05, p95 = np.percentile(col, [5, 95])
        rows.append({
            "prompt_idx":   row["prompt_idx"],
            "concept_id":   row["concept_id"],
            "template":     row["template"],
            "prompt":       row["prompt"],
            "min":          float(col.min()),
            "max":          float(col.max()),
            "mean":         float(col.mean()),
            "std":          float(col.std()),
            "p05":          float(p05),
            "p95":          float(p95),
            "flagged_dead": bool((p95 - p05) < 0.02),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 2: Per-class mean scores + heatmap
# ---------------------------------------------------------------------------

def analysis2_class_means(
    scores: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    output_dir: pathlib.Path,
) -> pd.DataFrame:
    """Mean cosine similarity per prompt per class; saves heatmap PNG."""
    rows = []
    for _, row in meta.iterrows():
        col = scores[:, row["prompt_idx"]]
        entry = {
            "prompt_idx": row["prompt_idx"],
            "concept_id": row["concept_id"],
            "template":   row["template"],
            "prompt":     row["prompt"],
        }
        for cls in CLASSES:
            mask = labels == cls
            entry[f"mean_{cls}"] = float(col[mask].mean()) if mask.any() else np.nan
        rows.append(entry)

    df = pd.DataFrame(rows)

    # --- Heatmap -------------------------------------------------------
    class_cols = [f"mean_{c}" for c in CLASSES]
    heat = df[class_cols].values.astype(float)          # (72, 7)

    # Row-wise z-score so cross-class patterns are visible
    mu  = heat.mean(axis=1, keepdims=True)
    sig = heat.std(axis=1, keepdims=True)
    sig[sig == 0] = 1.0
    z = (heat - mu) / sig

    y_labels = (df["concept_id"] + " " + df["template"]).tolist()

    # Concept group boundaries for horizontal separators
    boundaries: List[int] = []
    prev = None
    for i, cid in enumerate(df["concept_id"]):
        if cid != prev and prev is not None:
            boundaries.append(i)
        prev = cid

    fig, ax = plt.subplots(figsize=(9, 22))
    sns.heatmap(
        z,
        ax=ax,
        xticklabels=CLASSES,
        yticklabels=y_labels,
        cmap="RdBu_r",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        cbar_kws={"label": "z-score (row-wise)", "shrink": 0.4},
        linewidths=0,
    )
    if boundaries:
        ax.hlines(boundaries, xmin=0, xmax=len(CLASSES), colors="white", linewidths=1.0)
    ax.set_title("Per-prompt class mean scores (z-scored)", pad=10)
    ax.set_xlabel("Diagnostic class")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fig.savefig(output_dir / "prompt_class_means_heatmap.png", dpi=150)
    plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# Analysis 3: One-vs-rest AUC per prompt per class + bar plot
# ---------------------------------------------------------------------------

def analysis3_ovr_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    output_dir: pathlib.Path,
    version: str | None = None,
) -> pd.DataFrame:
    """OvR AUC for each prompt × class; saves top-20 bar plot PNG.

    When ``version`` is set (e.g. ``"v4"``), writes
    ``prompt_ovr_auc_v4.csv`` and ``top20_prompts_auc_v4.png`` instead of
    overwriting the v1 report files.
    """
    rows = []
    for _, row in meta.iterrows():
        col = scores[:, row["prompt_idx"]]
        entry = {
            "prompt_idx": row["prompt_idx"],
            "concept_id": row["concept_id"],
            "template":   row["template"],
            "prompt":     row["prompt"],
        }
        auc_by_class: dict[str, float] = {}
        for cls in CLASSES:
            y_bin = (labels == cls).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                auc_by_class[cls] = np.nan
            else:
                try:
                    auc_by_class[cls] = float(roc_auc_score(y_bin, col))
                except Exception:
                    auc_by_class[cls] = np.nan
            entry[f"auc_{cls}"] = auc_by_class[cls]

        valid = {c: v for c, v in auc_by_class.items() if not np.isnan(v)}
        if valid:
            best_cls = max(valid, key=valid.__getitem__)
            entry["auc_max"]          = valid[best_cls]
            entry["auc_argmax_class"] = best_cls
        else:
            entry["auc_max"]          = np.nan
            entry["auc_argmax_class"] = None
        rows.append(entry)

    df = pd.DataFrame(rows)

    # --- Top-20 bar plot -----------------------------------------------
    top20 = df.nlargest(20, "auc_max").iloc[::-1]   # ascending so best is at top
    bar_labels = (top20["concept_id"] + " " + top20["template"]).tolist()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top20)), top20["auc_max"].values, color="steelblue", height=0.7)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(bar_labels, fontsize=8)
    ax.axvline(0.5, color="tomato", linestyle="--", linewidth=0.9, label="random (0.5)")
    ax.set_xlabel("AUC-max (best OvR class)")
    ax.set_title("Top 20 prompts by AUC-max")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0.45)
    plt.tight_layout()
    plot_stem = _versioned_stem("top20_prompts_auc", version)
    fig.savefig(output_dir / f"{plot_stem}.png", dpi=150)
    plt.close(fig)

    if version is not None:
        csv_stem = _versioned_stem("prompt_ovr_auc", version)
        df.to_csv(output_dir / f"{csv_stem}.csv", index=False)

    return df


# ---------------------------------------------------------------------------
# Analysis 4: Template comparison (uses output of analysis 3)
# ---------------------------------------------------------------------------

def analysis4_template_comparison(
    auc_df: pd.DataFrame,
    concept_ids: List[str],
    tiers: np.ndarray,
) -> pd.DataFrame:
    """Compare t1/t2/t3 templates per concept by auc_max."""
    rows = []
    for c_idx, concept_id in enumerate(concept_ids):
        subset = auc_df[auc_df["concept_id"] == concept_id]

        def _get(tmpl: str) -> float:
            r = subset[subset["template"] == tmpl]
            return float(r["auc_max"].iloc[0]) if len(r) else np.nan

        t1, t2, t3 = _get("t1"), _get("t2"), _get("t3")
        vals = {"t1": t1, "t2": t2, "t3": t3}
        valid = {k: v for k, v in vals.items() if not np.isnan(v)}
        best_tmpl = max(valid, key=valid.__getitem__) if valid else None
        best_auc  = valid[best_tmpl] if best_tmpl else np.nan

        rows.append({
            "concept_id":   concept_id,
            "tier":         int(tiers[c_idx]),
            "t1_auc_max":   t1,
            "t2_auc_max":   t2,
            "t3_auc_max":   t3,
            "best_template": best_tmpl,
            "best_auc":     best_auc,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 5: Paired-contrast sanity checks
# ---------------------------------------------------------------------------

def analysis5_paired_contrasts(
    scores: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Check whether 'positive' concept outscores 'negative' on target class."""
    # Build concept_id -> t1 column index
    t1_lookup: dict[str, int] = {
        row["concept_id"]: row["prompt_idx"]
        for _, row in meta[meta["template"] == "t1"].iterrows()
    }
    all_t1_cols = list(t1_lookup.values())

    rows = []
    for pos_concept, neg_concept, target_class in _CONTRASTS:
        class_mask = labels == target_class
        pos_col    = scores[class_mask, t1_lookup[pos_concept]]
        pos_mean   = float(pos_col.mean())

        if neg_concept == "__all_others__":
            other_cols = [c for cid, c in t1_lookup.items() if cid != pos_concept]
            neg_mean   = float(scores[class_mask][:, other_cols].mean())
            neg_label  = "all_others"
        else:
            neg_col  = scores[class_mask, t1_lookup[neg_concept]]
            neg_mean = float(neg_col.mean())
            neg_label = neg_concept

        diff   = pos_mean - neg_mean
        status = "PASS" if diff >= 0.005 else "FAIL"

        rows.append({
            "positive_concept": pos_concept,
            "negative_concept": neg_label,
            "target_class":     target_class,
            "pos_mean":         round(pos_mean, 5),
            "neg_mean":         round(neg_mean, 5),
            "diff":             round(diff, 5),
            "status":           status,
        })
    return pd.DataFrame(rows)


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Run prompt quality Analysis 3 (per-prompt OvR AUC) on a score NPZ.",
    )
    p.add_argument(
        "--version",
        default="v4",
        help="Output suffix and default scores file (default: v4). Use 'none' for v1 paths.",
    )
    p.add_argument(
        "--scores",
        type=pathlib.Path,
        default=None,
        help="Override path to ham10000_concept_scores_*.npz",
    )
    p.add_argument(
        "--split",
        type=pathlib.Path,
        default=DEFAULT_SPLIT_NPZ,
        help="Train/test image index split NPZ",
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for CSV and PNG outputs",
    )
    args = p.parse_args()
    version = None if str(args.version).lower() in ("none", "", "v1") else args.version

    df3 = run_analysis3_ovr_auc(
        args.scores,
        version=version,
        split_npz=args.split,
        output_dir=args.output_dir,
    )

    mean_auc = float(df3["auc_max"].mean())
    csv_name = _versioned_stem("prompt_ovr_auc", version) + ".csv"
    plot_name = _versioned_stem("top20_prompts_auc", version) + ".png"
    print(f"\nMean auc_max: {mean_auc:.4f}")
    print(f"Saved {args.output_dir / csv_name}")
    print(f"Saved {args.output_dir / plot_name}")
    print("\nTop 5 prompts by auc_max:")
    for _, r in df3.nlargest(5, "auc_max").iterrows():
        print(
            f"  [{r['concept_id']} {r['template']}]  "
            f"auc_max={r['auc_max']:.4f}  best_class={r['auc_argmax_class']}"
        )


if __name__ == "__main__":
    _cli()
