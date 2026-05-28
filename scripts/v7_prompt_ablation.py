"""
v7 Prompt Ablation Diagnostic — 2×2 ablation on top-K most-correlated concept pairs.

Ablation axes
  A) Disease anchor   : original disease name vs generic "a skin lesion"
  B) Feature descriptor: original v6 wording vs structurally-rewritten phrasing
     that emphasises geometric / countable / spatial properties instead of
     generic atypia adjectives ("irregular", "atypical", "variegated").

Pipeline reused
  - src/features/clip_loader.load_biomedclip()  — model + tokeniser
  - src/features/encode_text.encode_prompts()   — L2-normalised text embeddings
  - src/analysis/concept_targets.CONCEPT_TARGET_CLASS — intended-class AUC
  - data/features/biomedclip/ham10000_image_embeddings.npy — cached image
    embeddings (10015 × 512), encoded once, shared across all 4 variants

Correlation matrices: train_final split (GroupShuffleSplit 80/20, random_state=42)
                      on 6429 images — identical to concept_correlation_diagnostic.py
AUC computation:      train_idx split (8020 images) — identical to analyze_prompts_v6.py

Run from project root:
    python scripts/v7_prompt_ablation.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import types

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Torchvision compatibility shim ────────────────────────────────────────────
# torch 2.11 (nightly CPU) + torchvision 0.26.0+cu126 crash at import because
# _meta_registrations.py uses @torch.library.register_fake("torchvision::nms")
# but the torchvision::nms custom op was never registered in the CPU-only build.
# torch._C._dispatch_has_kernel_for_dispatch_key() raises RuntimeError instead
# of returning False when the operator is unknown.
# Fix: patch that C function to return False gracefully for missing ops.
import torch as _torch
import torch._C as _torch_C

_orig_has_kernel = _torch_C._dispatch_has_kernel_for_dispatch_key

def _safe_has_kernel(qualname: str, dispatch_key: str) -> bool:
    try:
        return _orig_has_kernel(qualname, dispatch_key)
    except RuntimeError:
        return False

_torch_C._dispatch_has_kernel_for_dispatch_key = _safe_has_kernel

try:
    import torchvision as _tv  # noqa: F401
    print(f"NOTE: torchvision {_tv.__version__} loaded (with _dispatch_has_kernel patch).")
except Exception as _e:
    print(f"WARNING: torchvision still failed after patch: {_e}")

import configparser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.features.clip_loader import load_biomedclip
from src.features.encode_text import encode_prompts
from src.analysis.concept_targets import CONCEPT_TARGET_CLASS

# ── Paths ─────────────────────────────────────────────────────────────────────
CORR_CSV       = "reports/nam/diagnostics/concept_correlation.csv"
FEATURES_NPZ   = "data/features/biomedclip/ham10000_concept_scores_v6.npz"
SPLITS_PATH    = "data/splits/train_test_lesion_split.npz"
IMAGE_EMB_PATH = "data/features/biomedclip/ham10000_image_embeddings.npy"
PROMPT_FILE    = "src/features/prompts/ham10000_prompts_v6_biomedclip.txt"
OUT_DIR        = "reports/nam/diagnostics"
ABLATION_JSON  = "data/prompts/ham10000_prompts_v7_ablation.json"

TOP_K          = 8         # number of top-correlated pairs to target
FINAL_R_PERP   = 0.13     # R_perp_val of regularized NAM at lambda=1.0

# ── Descriptor rewrites for targeted concepts ─────────────────────────────────
# Rules: emphasise geometric, countable, or spatial properties.
# Avoid: "irregular", "atypical", "variegated", "abnormal" (the adjectives
# that the CLIP encoder tends to share across melanoma concepts).
DESCRIPTOR_REWRITES: dict[str, str] = {
    # ABCD rule
    "asymmetry": (
        "a lesion lacking bilateral symmetry when bisected along any axis"
    ),
    "border_irregularity": (
        "a border with notches, indentations, and angulated projections"
    ),
    "colour_variation": (
        "three or more distinct colours including brown, black, red, and white regions"
    ),
    "diameter_large": (
        "a lesion spanning more than six millimetres at its widest point"
    ),
    # 7-point checklist
    "irregular_pigmentation": (
        "pigment distributed non-uniformly in discrete blotches across the lesion surface"
    ),
    "irregular_dots_globules": (
        "dots and globules of varied size and spacing distributed unevenly across the lesion"
    ),
    "irregular_streaks": (
        "radial projections and finger-like pseudopods extending from the lesion periphery"
    ),
    "atypical_vascular_pattern": (
        "polymorphous vessels including dotted, looped, and corkscrew-shaped blood vessels"
    ),
    "regression_structures": (
        "white scar-like areas and blue-grey peppering replacing prior pigmented tissue"
    ),
    # BKL
    "milia_like_cysts": (
        "multiple white round structures resembling small keratin-filled cysts"
    ),
    "comedo_like_openings": (
        "round to oval keratin-filled crypts with dark central plugs"
    ),
    # Additional concepts that might appear in top-K under different seeds
    "atypical_pigment_network": (
        "a meshwork of pigmented lines with variable width and irregular branching"
    ),
    "blue_white_veil": (
        "a diffuse blue-white haze overlying an area of confluent pigmentation"
    ),
    "symmetric_uniform": (
        "a lesion with identical halves when bisected and homogeneous pigmentation throughout"
    ),
    "reticular_network": (
        "a regular honeycomb meshwork of thin pigmented lines with uniform spacing"
    ),
    "arborizing_vessels": (
        "large-calibre blood vessels branching like a tree with no anastomoses"
    ),
    "blue_grey_ovoid_nests": (
        "oval aggregates of blue-grey pigment larger than globules within the dermis"
    ),
    "ulceration": (
        "a disrupted surface with loss of overlying epidermis and exposed dermis"
    ),
    "scaly_surface": (
        "a surface covered in adherent white to yellowish keratotic scales"
    ),
    "strawberry_pattern": (
        "a red background with unfocused follicular openings surrounded by a white halo"
    ),
    "cerebriform_surface": (
        "gyri and sulci forming a convoluted brain-like ridged surface pattern"
    ),
    "central_white_patch": (
        "a central area of dense white fibrosis surrounded by a peripheral pigmented ring"
    ),
    "red_lacunae": (
        "discrete round to oval red or dark-red lacunae separated by white septae"
    ),
    "healthy_skin": (
        "no visible lesion, homogeneous skin-coloured background with follicular openings"
    ),
}

GENERIC_ANCHOR = "a skin lesion"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_v6_prompts(path: str) -> dict[str, str]:
    """Return {concept_id: prompt_text} preserving concept order from file."""
    parser = configparser.ConfigParser()
    with open(path, encoding="utf-8") as fh:
        # Strip comment lines (start with #) so configparser doesn't choke
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    parser.read_string("".join(lines))
    return {sec: parser[sec]["t1"].strip() for sec in parser.sections()}


def _parse_anchor_descriptor(prompt: str) -> tuple[str, str]:
    """Split 'Dermoscopy of {anchor} showing {descriptor}' → (anchor, descriptor)."""
    prefix, sep = "Dermoscopy of ", " showing "
    if not prompt.startswith(prefix):
        raise ValueError(f"Unexpected prompt format: {prompt!r}")
    rest = prompt[len(prefix):]
    idx = rest.index(sep)
    return rest[:idx], rest[idx + len(sep):]


def _build_variant(
    concept_ids: list[str],
    v6_prompts: dict[str, str],
    targeted: set[str],
    swap_anchor: bool,
    swap_descriptor: bool,
) -> list[str]:
    """Build a 24-element prompt list for one variant configuration."""
    out = []
    for cid in concept_ids:
        original = v6_prompts[cid]
        if cid not in targeted:
            out.append(original)
            continue
        anchor, descriptor = _parse_anchor_descriptor(original)
        a = GENERIC_ANCHOR if swap_anchor else anchor
        d = DESCRIPTOR_REWRITES[cid] if swap_descriptor else descriptor
        out.append(f"Dermoscopy of {a} showing {d}")
    return out


def _compute_corr(scores_subset: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix; scale-invariant, no z-scoring needed."""
    return np.corrcoef(scores_subset, rowvar=False)   # (24, 24)


def _mean_abs_off_diag(corr: np.ndarray) -> float:
    n = corr.shape[0]
    i, j = np.triu_indices(n, k=1)
    return float(np.mean(np.abs(corr[i, j])))


def _compute_aucs(
    scores_train: np.ndarray,       # (N_train_idx, 24)
    labels_train: np.ndarray,       # (N_train_idx,)
    concept_ids: list[str],
) -> dict[str, float]:
    """Intended-class AUC per concept on train_idx split (mirrors analyze_prompts_v6.py)."""
    aucs: dict[str, float] = {}
    for col, cid in enumerate(concept_ids):
        target_cls = CONCEPT_TARGET_CLASS.get(cid)
        if target_cls is None:
            aucs[cid] = float("nan")
            continue
        y_bin = (labels_train == target_cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            aucs[cid] = float("nan")
            continue
        aucs[cid] = float(roc_auc_score(y_bin, scores_train[:, col]))
    return aucs


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load correlation CSV and select top-K pairs
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1 — selecting top-K concept pairs")
print("=" * 70)

corr_df = pd.read_csv(CORR_CSV, index_col=0)
concept_ids_ordered: list[str] = list(corr_df.columns)
n_concepts = len(concept_ids_ordered)
corr_mat_ref = corr_df.values  # (24, 24)

idx_i, idx_j = np.triu_indices(n_concepts, k=1)
abs_r_vals = np.abs(corr_mat_ref[idx_i, idx_j])
top_k_pos  = np.argsort(-abs_r_vals)[:TOP_K]

top_pairs: list[tuple[str, str, float]] = []
for k in top_k_pos:
    ci = concept_ids_ordered[idx_i[k]]
    cj = concept_ids_ordered[idx_j[k]]
    r  = float(corr_mat_ref[idx_i[k], idx_j[k]])
    top_pairs.append((ci, cj, r))

print(f"\nTop {TOP_K} most-correlated concept pairs (from {CORR_CSV}):")
print(f"  {'Rank':<5} {'Concept A':<35} {'Concept B':<35} {'r':>8}")
print("  " + "-" * 83)
for rank, (ci, cj, r) in enumerate(top_pairs, 1):
    print(f"  {rank:<5} {ci:<35} {cj:<35} {r:+.4f}")

# Deduplicate targeted concepts
targeted: set[str] = set()
for ci, cj, _ in top_pairs:
    targeted.add(ci)
    targeted.add(cj)
targeted_sorted = sorted(targeted)
print(f"\nUnique targeted concepts ({len(targeted)}): {targeted_sorted}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parse v6 prompts and build 4 variant sets
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2 — building prompt variants")
print("=" * 70)

v6_prompts = _read_v6_prompts(PROMPT_FILE)

# Verify all targeted concepts have a DESCRIPTOR_REWRITE
missing_rewrites = [c for c in targeted if c not in DESCRIPTOR_REWRITES]
if missing_rewrites:
    raise RuntimeError(
        f"No descriptor rewrite defined for: {missing_rewrites}. "
        "Add entries to DESCRIPTOR_REWRITES before running."
    )

print("\n[AUDIT] Descriptor rewrites for targeted concepts:")
print(f"  {'Concept':<35} {'Original descriptor':<55} {'Rewritten descriptor'}")
print("  " + "-" * 130)
for cid in concept_ids_ordered:
    if cid not in targeted:
        continue
    _, orig_desc = _parse_anchor_descriptor(v6_prompts[cid])
    rewrite = DESCRIPTOR_REWRITES[cid]
    print(f"  {cid:<35} {orig_desc:<55} {rewrite}")

VARIANTS: dict[str, list[str]] = {
    "original":              _build_variant(concept_ids_ordered, v6_prompts, targeted,
                                            swap_anchor=False, swap_descriptor=False),
    "anchor_swapped":        _build_variant(concept_ids_ordered, v6_prompts, targeted,
                                            swap_anchor=True,  swap_descriptor=False),
    "descriptor_rewritten":  _build_variant(concept_ids_ordered, v6_prompts, targeted,
                                            swap_anchor=False, swap_descriptor=True),
    "both_changed":          _build_variant(concept_ids_ordered, v6_prompts, targeted,
                                            swap_anchor=True,  swap_descriptor=True),
}
VARIANT_LABELS = {
    "original":             "Original (v6)",
    "anchor_swapped":       "Anchor-swapped",
    "descriptor_rewritten": "Descriptor-rewritten",
    "both_changed":         "Both changed",
}

# Save prompt variants to JSON
os.makedirs(os.path.dirname(ABLATION_JSON), exist_ok=True)
json_out: dict[str, dict[str, str]] = {}
for vname, prompts in VARIANTS.items():
    json_out[vname] = {cid: p for cid, p in zip(concept_ids_ordered, prompts)}
with open(ABLATION_JSON, "w", encoding="utf-8") as fh:
    json.dump(json_out, fh, indent=2, ensure_ascii=False)
print(f"\nVariant prompts saved to: {ABLATION_JSON}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Load data: image embeddings, labels, splits
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3 — loading data & model")
print("=" * 70)

print("Loading cached image embeddings...")
image_emb = np.load(IMAGE_EMB_PATH)          # (10015, 512) L2-normalised
assert image_emb.shape == (10015, 512), f"Unexpected shape: {image_emb.shape}"
image_t = torch.tensor(image_emb, dtype=torch.float32)

print("Loading features / labels / splits...")
feat       = np.load(FEATURES_NPZ, allow_pickle=True)
labels     = feat["labels"]                  # (10015,) string class labels
lesion_ids = feat["lesion_ids"]             # (10015,) for GroupShuffleSplit

split     = np.load(SPLITS_PATH)
train_idx = split["train_idx"]               # 8020 — used for AUC (matches analyze_prompts_v6.py)
X_all_train      = image_emb[train_idx]
lesion_ids_train = lesion_ids[train_idx]
labels_train     = labels[train_idx]
y_all_train      = labels_train             # string labels

# train_final split (same GroupShuffleSplit as train_nam_v6_final.py) — for correlation
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_final_rel, _ = next(
    gss.split(X_all_train, y_all_train, groups=lesion_ids_train)
)
train_final_abs = train_idx[train_final_rel]   # absolute indices into (10015,)
print(f"  train_idx (AUC)     : {len(train_idx):5d} images")
print(f"  train_final (corr)  : {len(train_final_abs):5d} images")

print("Loading BiomedCLIP model...")
model, _, tokenizer, device = load_biomedclip()
print(f"  Device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Encode text + compute scores + correlation + AUC for each variant
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — running 4 encoding passes")
print("=" * 70)

corr_matrices: dict[str, np.ndarray] = {}
auc_results:   dict[str, dict[str, float]] = {}

for vname, prompts in VARIANTS.items():
    print(f"\n  Variant: {VARIANT_LABELS[vname]}")
    assert len(prompts) == 24, f"Expected 24 prompts, got {len(prompts)}"

    # Encode text (BiomedCLIP PubMedBERT, L2-normalised, on CPU)
    text_emb = encode_prompts(model, tokenizer, prompts, device)  # (24, 512)

    # Cosine similarity: image_emb @ text_emb.T (both L2-normalised → dot = cosine)
    with torch.no_grad():
        scores_all = (image_t @ text_emb.T).numpy()   # (10015, 24)

    # Correlation on train_final (mirrors concept_correlation_diagnostic.py)
    scores_tf = scores_all[train_final_abs]            # (6429, 24)
    corr_matrices[vname] = _compute_corr(scores_tf)

    # AUC on train_idx (mirrors analyze_prompts_v6.py)
    scores_tr = scores_all[train_idx]                  # (8020, 24)
    auc_results[vname] = _compute_aucs(scores_tr, labels_train, concept_ids_ordered)

    mean_abs = _mean_abs_off_diag(corr_matrices[vname])
    mean_auc = float(np.nanmean(list(auc_results[vname].values())))
    print(f"    mean |r| off-diag : {mean_abs:.4f}")
    print(f"    mean intended AUC : {mean_auc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Build per-pair results table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5 — per-pair results")
print("=" * 70)

rows = []
for ci, cj, _ in top_pairs:
    ii = concept_ids_ordered.index(ci)
    jj = concept_ids_ordered.index(cj)
    r_orig = float(corr_matrices["original"][ii, jj])
    r_anch = float(corr_matrices["anchor_swapped"][ii, jj])
    r_desc = float(corr_matrices["descriptor_rewritten"][ii, jj])
    r_both = float(corr_matrices["both_changed"][ii, jj])
    d_anch = r_anch - r_orig
    d_desc = r_desc - r_orig
    d_both = r_both - r_orig
    deltas = {"anchor": d_anch, "descriptor": d_desc, "both": d_both}
    best   = min(deltas, key=lambda k: deltas[k])   # most negative = best reduction
    rows.append({
        "concept_i":             ci,
        "concept_j":             cj,
        "r_original":            round(r_orig, 4),
        "r_anchor_swapped":      round(r_anch, 4),
        "r_descriptor_rewritten": round(r_desc, 4),
        "r_both_changed":        round(r_both, 4),
        "delta_anchor":          round(d_anch, 4),
        "delta_descriptor":      round(d_desc, 4),
        "delta_both":            round(d_both, 4),
        "best_intervention":     best,
    })

results_df = pd.DataFrame(rows).sort_values("r_original", key=abs, ascending=False)
os.makedirs(OUT_DIR, exist_ok=True)
results_csv = os.path.join(OUT_DIR, "v7_ablation_results.csv")
results_df.to_csv(results_csv, index=False)

print(f"\nPer-pair results (sorted by |r_original|):")
cols = ["concept_i", "concept_j", "r_original", "r_anchor_swapped",
        "r_descriptor_rewritten", "r_both_changed", "best_intervention"]
print(results_df[cols].to_string(index=False))
print(f"\nSaved to: {results_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-concept AUC table
# ─────────────────────────────────────────────────────────────────────────────
auc_rows = []
for cid in concept_ids_ordered:
    auc_rows.append({
        "concept_id":            cid,
        "target_class":          CONCEPT_TARGET_CLASS.get(cid, "?"),
        "auc_original":          round(auc_results["original"][cid], 4),
        "auc_anchor_swapped":    round(auc_results["anchor_swapped"][cid], 4),
        "auc_descriptor_rewritten": round(auc_results["descriptor_rewritten"][cid], 4),
        "auc_both_changed":      round(auc_results["both_changed"][cid], 4),
    })
auc_df = pd.DataFrame(auc_rows)
auc_csv = os.path.join(OUT_DIR, "v7_ablation_aucs.csv")
auc_df.to_csv(auc_csv, index=False)

print(f"\nPer-concept intended-class AUC (train_idx, N=8020):")
print(auc_df.to_string(index=False))
print(f"\nSaved to: {auc_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

ref_mean = _mean_abs_off_diag(corr_matrices["original"])
for vname, label in VARIANT_LABELS.items():
    m = _mean_abs_off_diag(corr_matrices[vname])
    d = m - ref_mean
    mean_auc = float(np.nanmean(list(auc_results[vname].values())))
    sign = "+" if d >= 0 else ""
    if vname == "original":
        print(f"\n  {label:<38}: mean |r| = {m:.4f}   mean AUC = {mean_auc:.4f}")
    else:
        print(f"  {label:<38}: mean |r| = {m:.4f}  (delta {sign}{d:.4f})"
              f"   mean AUC = {mean_auc:.4f}")

print(f"\n  Reference R_perp_val (regularized NAM, lambda=1.0): {FINAL_R_PERP:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. 2×2 heatmap comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating 2×2 heatmap figure...")

fig, axes = plt.subplots(2, 2, figsize=(22, 20))
variant_order = ["original", "anchor_swapped", "descriptor_rewritten", "both_changed"]

for ax, vname in zip(axes.flat, variant_order):
    corr = corr_matrices[vname]
    m    = _mean_abs_off_diag(corr)
    n    = corr.shape[0]

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_title(f"{VARIANT_LABELS[vname]}\nmean |r| = {m:.4f}", fontsize=11, pad=8)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(concept_ids_ordered, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(concept_ids_ordered, fontsize=6.5)

    for row in range(n):
        for col in range(n):
            r_val = corr[row, col]
            if abs(r_val) > 0.35:
                tc = "white" if abs(r_val) > 0.65 else "black"
                ax.text(col, row, f"{r_val:.2f}", ha="center", va="center",
                        fontsize=4.5, color=tc)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Pearson r", fontsize=9)

fig.suptitle(
    "BiomedCLIP concept correlation: 2×2 prompt ablation\n"
    "(top-left = v6 baseline; top-right = anchor-swapped; "
    "bottom-left = descriptor-rewritten; bottom-right = both)",
    fontsize=12, y=1.01
)
plt.tight_layout()

png_path = os.path.join(OUT_DIR, "v7_ablation_heatmaps.png")
pdf_path = os.path.join(OUT_DIR, "v7_ablation_heatmaps.pdf")
fig.savefig(png_path, dpi=130, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)
print(f"Heatmaps saved to:\n  {png_path}\n  {pdf_path}")

print("\nDone.")
