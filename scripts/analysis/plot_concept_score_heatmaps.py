"""
Generate three appendix figures:
  1. HAM10000 per-class concept score heatmap
  2. Chest X-ray per-class concept score heatmap
  3. Mel vs. nv prototype horizontal bar chart

Outputs are written to results/analysis/figures/.
Nothing in data/ or results/ is modified except by writing new figures.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import os

OUT_DIR = Path("results/analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def pretty_name(s):
    return s.replace("_", " ").replace(" like ", "-like ").replace(" grey ", "-grey ")

CLASS_ORDER_HAM = ["mel", "bcc", "akiec", "bkl", "df", "vasc", "nv"]
CLASS_LABELS_HAM = {
    "mel":   "Melanoma",
    "bcc":   "Basal cell\ncarcinoma",
    "akiec": "Actinic\nkeratosis",
    "bkl":   "Benign\nkeratosis",
    "df":    "Dermatofibroma",
    "vasc":  "Vascular\nlesion",
    "nv":    "Melanocytic\nnevus",
}

CLASS_ORDER_CX  = ["bacteria", "virus", "normal"]
CLASS_LABELS_CX = {
    "bacteria": "Bacterial\npneumonia",
    "virus":    "Viral\npneumonia",
    "normal":   "Normal",
}

# Tier groupings and separators (0-indexed concept positions within each dataset)
# HAM: 0-3 = ABCD (tier 1), 4-10 = 7-point (tier 2), 11-23 = class-anchors (tier 3)
# CX:  0-5 = general (tier 1), 6-12 = subtype (tier 2), 13-16 = normal-anchor (tier 3)

TIER_NAMES_HAM = ["ABCD rule", "7-point checklist", "Class-anchor concepts"]
TIER_BOUNDS_HAM = [(0, 4), (4, 11), (11, 24)]   # [start, end)

TIER_NAMES_CX  = ["General pneumonia signs", "Subtype-discriminative", "Normal anchors"]
TIER_BOUNDS_CX  = [(0, 6), (6, 13), (13, 17)]

TIER_COLORS = ["#d4e8f7", "#fde8c8", "#d6f0d6"]  # light blue, orange, green


def load_ham10000():
    d = np.load("data/features/biomedclip/ham10000_concept_scores_v6.npz",
                allow_pickle=True)
    scores     = d["scores"]                         # (10015, 24)
    labels     = np.array([str(x) for x in d["labels"]])
    image_ids  = np.array([str(x) for x in d["image_ids"]])
    concept_ids = [str(x) for x in d["concept_ids"]]
    return scores, labels, image_ids, concept_ids


def load_chestxray():
    d = np.load("data/features/biomedclip/chestxray_concept_scores_v4.npz",
                allow_pickle=True)
    scores     = d["scores"]                         # (5856, 17)
    image_paths = np.array([str(x) for x in d["image_paths"]])
    concept_names = [str(x) for x in d["concept_names"]]
    labels = []
    for p in image_paths:
        pu = p.upper()
        if "NORMAL"   in pu: labels.append("normal")
        elif "BACTERIA" in pu: labels.append("bacteria")
        elif "VIRUS"    in pu: labels.append("virus")
        else: labels.append("unknown")
    return scores, np.array(labels), image_paths, concept_names


def class_mean_matrix(scores, labels, class_order, concept_ids):
    """Return (n_classes × n_concepts) DataFrame of per-class means."""
    df = pd.DataFrame(scores, columns=concept_ids)
    df["label"] = labels
    means = df.groupby("label")[concept_ids].mean()
    return means.loc[class_order]          # reorder rows


def deviation_matrix(means_df):
    """Return signed deviation from per-concept mean (column-centred)."""
    return means_df - means_df.mean(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 & 2: Heatmaps
# ──────────────────────────────────────────────────────────────────────────────

def draw_heatmap(
    means_df,
    class_labels_map,
    tier_bounds,
    tier_names,
    tier_colors,
    title,
    out_stem,
):
    dev = deviation_matrix(means_df)
    concepts = list(means_df.columns)
    class_keys = list(means_df.index)

    n_classes  = len(class_keys)
    n_concepts = len(concepts)

    # Layout: wide figure, one column per concept + colour bar
    fig_w = max(14, n_concepts * 0.62)
    fig_h = n_classes * 0.8 + 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Symmetric colour scale capped at ±0.025
    vmax = 0.025
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdYlGn          # red = below average, green = above average

    # Draw cells
    im = ax.imshow(
        dev.values,
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )

    # Annotate each cell with the signed deviation from per-concept mean
    for r, cls in enumerate(class_keys):
        for c, concept in enumerate(concepts):
            dev_val  = dev.loc[cls, concept]
            # Use white text on dark cells for readability
            text_col = "white" if abs(dev_val) > 0.015 else "black"
            ax.text(c, r, f"{dev_val:+.3f}", ha="center", va="center",
                    fontsize=6.2, color=text_col, fontweight="normal")

    # Axis ticks
    ax.set_xticks(range(n_concepts))
    ax.set_xticklabels(
        [pretty_name(c) for c in concepts],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(
        [class_labels_map[k] for k in class_keys],
        fontsize=9,
    )

    # Vertical tier separators on the heatmap
    for (t_start, t_end) in tier_bounds:
        if t_end < n_concepts:
            ax.axvline(x=t_end - 0.5, color="white", linewidth=2.5, zorder=5)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Deviation from per-concept mean", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.tick_params(axis="both", which="both", length=0)

    # Reserve top 16% of figure height for tier header bands + title; this
    # must happen BEFORE we query ax.get_position() so tight_layout has run.
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    # Place tier header bands at their post-layout positions
    ax_pos  = ax.get_position()
    tier_h  = 0.048                    # band height in figure fraction
    tier_y  = ax_pos.y1 + 0.012       # gap above heatmap top edge

    for (t_start, t_end), t_name, t_col in zip(tier_bounds, tier_names, tier_colors):
        left  = ax_pos.x0 + (t_start / n_concepts) * ax_pos.width
        width = ((t_end - t_start) / n_concepts) * ax_pos.width
        tier_ax = fig.add_axes([left, tier_y, width, tier_h])
        tier_ax.set_facecolor(t_col)
        tier_ax.set_xticks([])
        tier_ax.set_yticks([])
        for spine in tier_ax.spines.values():
            spine.set_visible(False)
        tier_ax.text(
            0.5, 0.5, t_name,
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            transform=tier_ax.transAxes,
        )

    # Main title sits above the tier bands with a small gap
    fig.suptitle(title, y=tier_y + tier_h + 0.025,
                 fontsize=10, fontweight="bold", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{out_stem}.{ext}", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_stem}.pdf/.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Prototype bar chart (mel vs. nv)
# ──────────────────────────────────────────────────────────────────────────────

def find_prototype(scores, labels, image_ids, target_class, other_class):
    """
    Return the image in target_class that is most discriminative against other_class.
    Uses the projection of each candidate onto the (target_mean - other_mean) vector,
    so the selected image best represents the features that separate target from other.
    Also constrained to be within the 40th percentile distance to its own centroid
    (so it is still representative, not just an outlier on one feature).
    """
    t_mask   = labels == target_class
    o_mask   = labels == other_class
    t_idx    = np.where(t_mask)[0]

    t_mean   = scores[t_mask].mean(axis=0)
    o_mean   = scores[o_mask].mean(axis=0)
    diff_vec = t_mean - o_mean
    diff_norm = np.linalg.norm(diff_vec)
    if diff_norm < 1e-9:
        diff_vec = np.ones_like(t_mean)
        diff_norm = np.linalg.norm(diff_vec)

    # Projection of each target-class image onto the discriminant axis
    projs    = scores[t_idx] @ diff_vec / diff_norm

    # Also compute distance to centroid to filter out outliers
    dists_to_cent = np.linalg.norm(scores[t_idx] - t_mean, axis=1)
    percentile_40 = np.percentile(dists_to_cent, 40)

    # Among images close enough to the centroid, pick the one with max projection
    eligible = dists_to_cent <= percentile_40
    if eligible.sum() == 0:
        eligible = np.ones(len(t_idx), dtype=bool)

    best_local = np.argmax(np.where(eligible, projs, -np.inf))
    best       = t_idx[best_local]
    return image_ids[best], scores[best], dists_to_cent[best_local]


def draw_prototype_bar(
    mel_scores, nv_scores, mel_id, nv_id,
    concept_ids, mel_class_mean, nv_class_mean, out_stem,
):
    concepts = [pretty_name(c) for c in concept_ids]
    n = len(concepts)

    # Sort by mel − nv difference of the INDIVIDUAL image scores
    diff = mel_scores - nv_scores
    order = np.argsort(diff)[::-1]   # descending: most mel-favouring first

    mel_sorted      = mel_scores[order]
    nv_sorted       = nv_scores[order]
    mel_mean_sorted = mel_class_mean[order]
    nv_mean_sorted  = nv_class_mean[order]
    labels_sorted   = [concepts[i] for i in order]
    diff_sorted     = diff[order]

    fig, ax = plt.subplots(figsize=(9.5, 8.5))

    y      = np.arange(n)
    height = 0.38

    ax.barh(
        y + height / 2, mel_sorted, height=height,
        color="#c0392b", alpha=0.85,
        label=f"Melanoma image  ({mel_id})",
        edgecolor="white", linewidth=0.4,
    )
    ax.barh(
        y - height / 2, nv_sorted, height=height,
        color="#2980b9", alpha=0.85,
        label=f"Melanocytic nevus image  ({nv_id})",
        edgecolor="white", linewidth=0.4,
    )

    # Overlay class-mean markers as vertical tick marks
    for i in range(n):
        ax.plot(mel_mean_sorted[i], y[i] + height / 2,
                marker="|", markersize=9, markeredgewidth=1.8,
                color="#7b241c", zorder=5)
        ax.plot(nv_mean_sorted[i], y[i] - height / 2,
                marker="|", markersize=9, markeredgewidth=1.8,
                color="#1a5276", zorder=5)

    # Annotate difference for concepts where |diff| > 0.008
    for i, (m, nval, d) in enumerate(zip(mel_sorted, nv_sorted, diff_sorted)):
        if abs(d) > 0.008:
            sign = "+" if d > 0 else "−"
            ax.text(
                max(m, nval) + 0.003, y[i],
                f"{sign}{abs(d):.3f}",
                va="center", ha="left", fontsize=7,
                color="#c0392b" if d > 0 else "#2980b9",
                fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels_sorted, fontsize=8.2)
    ax.set_xlabel("BiomedCLIP cosine similarity score", fontsize=9)
    ax.set_title(
        "Concept score profiles of two representative HAM10000 images\n"
        "(melanoma vs. melanocytic nevus; sorted by image-level mel−nv difference;\n"
        "tick marks show the class mean for each concept)",
        fontsize=9.5, fontweight="bold",
    )

    # Custom legend with class-mean markers
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor="#c0392b", alpha=0.85,
                       label=f"Melanoma image  ({mel_id})"),
        mpatches.Patch(facecolor="#2980b9", alpha=0.85,
                       label=f"Melanocytic nevus image  ({nv_id})"),
        Line2D([0], [0], marker="|", color="#7b241c", markersize=9,
               markeredgewidth=2, linestyle="None",
               label="Melanoma class mean"),
        Line2D([0], [0], marker="|", color="#1a5276", markersize=9,
               markeredgewidth=2, linestyle="None",
               label="Melanocytic nevus class mean"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_xlim(0.32, 0.57)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{out_stem}.{ext}", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_stem}.pdf/.png")

    return order


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── HAM10000 ──────────────────────────────────────────────────────────────
    print("Loading HAM10000 concept scores …")
    ham_scores, ham_labels, ham_ids, ham_concepts = load_ham10000()

    means_ham = class_mean_matrix(ham_scores, ham_labels, CLASS_ORDER_HAM, ham_concepts)

    draw_heatmap(
        means_ham,
        CLASS_LABELS_HAM,
        TIER_BOUNDS_HAM,
        TIER_NAMES_HAM,
        TIER_COLORS,
        title=(
            "HAM10000: Mean BiomedCLIP cosine similarity by class and concept\n"
            "(cell values show signed deviation from per-concept mean; green = above average, red = below average)"
        ),
        out_stem="heatmap_concept_scores_ham10000",
    )

    # ── Chest X-ray ───────────────────────────────────────────────────────────
    print("Loading chest X-ray concept scores …")
    cx_scores, cx_labels, cx_paths, cx_concepts = load_chestxray()

    means_cx = class_mean_matrix(cx_scores, cx_labels, CLASS_ORDER_CX, cx_concepts)

    draw_heatmap(
        means_cx,
        CLASS_LABELS_CX,
        TIER_BOUNDS_CX,
        TIER_NAMES_CX,
        TIER_COLORS,
        title=(
            "Chest X-ray: Mean BiomedCLIP cosine similarity by class and concept\n"
            "(cell values show signed deviation from per-concept mean; green = above average, red = below average)"
        ),
        out_stem="heatmap_concept_scores_chestxray",
    )

    # ── Prototype bar chart ───────────────────────────────────────────────────
    print("\nFinding prototype images …")

    mel_id,  mel_s,  mel_d  = find_prototype(ham_scores, ham_labels, ham_ids, "mel", "nv")
    nv_id,   nv_s,   nv_d   = find_prototype(ham_scores, ham_labels, ham_ids, "nv",  "mel")

    print(f"  Mel prototype : {mel_id}  (L2 dist to centroid = {mel_d:.4f})")
    print(f"  Nv  prototype : {nv_id}   (L2 dist to centroid = {nv_d:.4f})")

    # Locate image files
    for img_id in [mel_id, nv_id]:
        for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
            candidate = Path("data/ham10000") / part / f"{img_id}.jpg"
            if candidate.exists():
                print(f"  Image file: {candidate}")
                break

    # Class means for the tick-mark overlay
    mel_class_mean = np.array(means_ham.loc["mel"])
    nv_class_mean  = np.array(means_ham.loc["nv"])

    draw_prototype_bar(
        mel_s, nv_s, mel_id, nv_id, ham_concepts,
        mel_class_mean, nv_class_mean,
        out_stem="prototype_mel_nv_barchart",
    )

    # ── Summary of prototype concept scores ──────────────────────────────────
    print("\nTop concept differences (mel − nv):")
    diff = mel_s - nv_s
    order = np.argsort(np.abs(diff))[::-1]
    for i in order[:10]:
        sign = "+" if diff[i] > 0 else ""
        print(f"  {ham_concepts[i]:30s}  mel={mel_s[i]:.4f}  nv={nv_s[i]:.4f}  diff={sign}{diff[i]:.4f}")


if __name__ == "__main__":
    main()
