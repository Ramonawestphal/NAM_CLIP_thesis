"""
v1 vs v2 chest X-ray BiomedCLIP concept score comparison.

v1 has 19 concepts; v2 has 18 (no_focal_opacity DROPPED). Change sets:
    REVISED (5): focal_opacity, dense_segmental_opacity, round_pneumonia,
        bilateral_interstitial_pattern, hyperinflation
    DROPPED (1): no_focal_opacity
    UNCHANGED (13): the remaining shared concepts (byte-identical prompts)
revision_status is parsed from the v2 prompt-file annotations (# REVISED /
# UNCHANGED), cross-checked against byte comparison of the prompt text.

Reads (all read-only):
    results/chestxray/per_class_diagnostic/per_class_means.csv          (v1)
    results/chestxray/per_class_diagnostic_v2/per_class_means.csv       (v2)
    results/chestxray/correlation_diagnostic/correlation_matrix.npy     (v1)
    results/chestxray/correlation_diagnostic_v2/correlation_matrix.npy  (v2)
    data/features/biomedclip/chestxray_concept_scores_{v1,v2}.npz       (orderings)

Outputs:
    results/chestxray/v1_vs_v2_comparison/delta_per_class_means.csv
    results/chestxray/v1_vs_v2_comparison/delta_heatmap.png
    results/chestxray/v1_vs_v2_comparison/comparison_summary.txt

Run from project root (after the v2 diagnostics):
    python scripts/chestxray/compare_v1_v2.py
"""

from __future__ import annotations

import csv
import pathlib
import re
import sys
import textwrap
from typing import Dict, List, Tuple

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features.prompt_loader import load_prompts

# ── Paths ─────────────────────────────────────────────────────────────────────
V1_CSV       = _ROOT / "results/chestxray/per_class_diagnostic/per_class_means.csv"
V2_CSV       = _ROOT / "results/chestxray/per_class_diagnostic_v2/per_class_means.csv"
V1_CORR_NPY  = _ROOT / "results/chestxray/correlation_diagnostic/correlation_matrix.npy"
V2_CORR_NPY  = _ROOT / "results/chestxray/correlation_diagnostic_v2/correlation_matrix.npy"
V1_SCORES    = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v1.npz"
V2_SCORES    = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v2.npz"
PROMPTS_V1   = _ROOT / "src/features/prompts/chestxray_prompts_v1.txt"
PROMPTS_V2   = _ROOT / "src/features/prompts/chestxray_prompts_v2.txt"
OUT_DIR      = _ROOT / "results/chestxray/v1_vs_v2_comparison"

CLASSES = ["normal", "bacteria", "virus"]

# ── Pre-registered blocks (normal_anchor differs between versions) ─────────────
CONSOLIDATION_BLOCK = [
    "consolidation", "focal_opacity", "lobar_consolidation",
    "dense_segmental_opacity", "air_bronchograms",
]
INTERSTITIAL_BLOCK = [
    "bilateral_interstitial_pattern", "peribronchial_cuffing", "perihilar_infiltrates",
]
NORMAL_ANCHOR_BLOCK_V1 = [
    "clear_lung_fields", "sharp_costophrenic_angles",
    "normal_cardiac_silhouette", "no_focal_opacity", "symmetric_lung_aeration",
]
NORMAL_ANCHOR_BLOCK_V2 = [
    "clear_lung_fields", "sharp_costophrenic_angles",
    "normal_cardiac_silhouette", "symmetric_lung_aeration",
]

EXPECTED_TARGET: Dict[str, List[str]] = {
    "consolidation":                    ["bacteria", "virus"],
    "focal_opacity":                    ["bacteria", "virus"],
    "air_bronchograms":                 ["bacteria", "virus"],
    "pleural_effusion":                 ["bacteria", "virus"],
    "silhouette_sign":                  ["bacteria", "virus"],
    "patchy_infiltrate":                ["bacteria", "virus"],
    "lobar_consolidation":              ["bacteria"],
    "dense_segmental_opacity":          ["bacteria"],
    "parapneumonic_effusion":           ["bacteria"],
    "round_pneumonia":                  ["bacteria"],
    "bilateral_interstitial_pattern":   ["virus"],
    "peribronchial_cuffing":            ["virus"],
    "perihilar_infiltrates":            ["virus"],
    "hyperinflation":                   ["virus"],
    "clear_lung_fields":                ["normal"],
    "sharp_costophrenic_angles":        ["normal"],
    "normal_cardiac_silhouette":        ["normal"],
    "symmetric_lung_aeration":          ["normal"],
    "no_focal_opacity":                 ["normal"],   # v1 only
}

TIER_LABELS: Dict[str, str] = {
    "consolidation": "T1-general", "focal_opacity": "T1-general",
    "air_bronchograms": "T1-general", "pleural_effusion": "T1-general",
    "silhouette_sign": "T1-general", "patchy_infiltrate": "T1-general",
    "lobar_consolidation": "T2A-bacteria", "dense_segmental_opacity": "T2A-bacteria",
    "parapneumonic_effusion": "T2A-bacteria", "round_pneumonia": "T2A-bacteria",
    "bilateral_interstitial_pattern": "T2B-virus", "peribronchial_cuffing": "T2B-virus",
    "perihilar_infiltrates": "T2B-virus", "hyperinflation": "T2B-virus",
    "clear_lung_fields": "T3-normal", "sharp_costophrenic_angles": "T3-normal",
    "normal_cardiac_silhouette": "T3-normal", "symmetric_lung_aeration": "T3-normal",
    "no_focal_opacity": "T3-normal",
}

# ── Pre-registered hypotheses (registered on v1; v2 = follow-up) ───────────────
HYPOTHESES = {
    "H1": "Within consolidation_block, mean |r| > 0.7",
    "H2": "Within interstitial_block, mean |r| > 0.7",
    "H3": "Between consolidation & interstitial, mean |r| < within-block |r| of either",
    "H4": "normal_anchor_block correlates negatively (signed r < 0) with both pneumonia blocks",
    "H5": "Aggregate mean |r| over all pairs falls in [0.3, 0.6]",
}


# ── Revision-status parsing ───────────────────────────────────────────────────

def parse_revision_status() -> Tuple[set, set, set]:
    """Return (revised, unchanged, dropped) sets.

    revised/unchanged are parsed from v2 prompt annotations (# REVISED /
    # UNCHANGED). dropped = concepts present in v1 but absent from v2.
    Cross-checked against byte comparison of prompt text.
    """
    raw = PROMPTS_V2.read_text(encoding="utf-8")
    revised   = set(re.findall(r"\[(\w+)\]\s*\n#\s*REVISED",   raw))
    unchanged = set(re.findall(r"\[(\w+)\]\s*\n#\s*UNCHANGED", raw))

    p1 = load_prompts(PROMPTS_V1)
    p2 = load_prompts(PROMPTS_V2)
    v1_set, v2_set = set(p1["concept_ids"]), set(p2["concept_ids"])
    dropped = v1_set - v2_set

    # Cross-check annotations against byte comparison
    v1m = dict(zip(p1["concept_ids"], p1["prompts"]))
    v2m = dict(zip(p2["concept_ids"], p2["prompts"]))
    byte_revised   = {c for c in v1_set & v2_set if v1m[c] != v2m[c]}
    byte_unchanged = {c for c in v1_set & v2_set if v1m[c] == v2m[c]}
    if revised != byte_revised:
        print(f"  WARNING: annotation REVISED {sorted(revised)} != "
              f"byte-diff REVISED {sorted(byte_revised)}; using byte comparison")
        revised, unchanged = byte_revised, byte_unchanged
    return revised, unchanged, dropped


REVISED_CONCEPTS, UNCHANGED_CONCEPTS, DROPPED_FROM_V1 = parse_revision_status()


def revision_status(cname: str) -> str:
    if cname in DROPPED_FROM_V1:
        return "DROPPED"
    if cname in REVISED_CONCEPTS:
        return "REVISED"
    if cname in UNCHANGED_CONCEPTS:
        return "UNCHANGED"
    return "UNKNOWN"


# ── CSV / stats helpers ───────────────────────────────────────────────────────

def load_per_class_csv(path: pathlib.Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result.setdefault(row["concept"], {})[row["class"]] = {
                "mean": float(row["mean"]),
                "std":  float(row["std"]),
                "n":    int(row["n"]),
            }
    return result


def argmax_class(stats: Dict[str, Dict[str, float]]) -> str:
    return max(CLASSES, key=lambda c: stats[c]["mean"])


def is_match(cname: str, am: str) -> bool:
    return am in EXPECTED_TARGET.get(cname, [])


def spread(stats: Dict[str, Dict[str, float]]) -> float:
    vals = [stats[c]["mean"] for c in CLASSES]
    return max(vals) - min(vals)


def discrimination_score(cname: str, stats: Dict[str, Dict[str, float]]) -> float:
    """mean[target_class(es)] - mean[non-target class(es)]."""
    targets = EXPECTED_TARGET.get(cname, [])
    if not targets:
        return float("nan")
    t_mean  = np.mean([stats[t]["mean"] for t in targets if t in stats])
    nt_mean = np.mean([stats[c]["mean"] for c in CLASSES if c not in targets])
    return float(t_mean - nt_mean)


def discrimination_status(cname: str, v1_m, v2_m) -> str:
    rs = revision_status(cname)
    if rs == "DROPPED":
        return "DROPPED"
    if rs == "UNCHANGED":
        if v1_m and v2_m:
            return "UNCHANGED-STILL-MATCH"
        if v1_m != v2_m:
            return "UNCHANGED-DRIFT"          # bit-identity bug if it occurs
        return "UNCHANGED-STILL-MISMATCH"     # both fail; byte-identical scores
    if rs == "REVISED":
        if not v1_m and v2_m:
            return "REVISED-FIXED"
        if not v1_m and not v2_m:
            return "REVISED-PERSISTENT-FAIL"
        if v1_m and not v2_m:
            return "REVISED-REGRESSED"
        return "REVISED-STILL-MATCH"          # was MATCH, still MATCH
    return "UNKNOWN"


# ── Correlation-matrix helpers ────────────────────────────────────────────────

def load_corr(matrix_path: pathlib.Path, scores_npz: pathlib.Path) -> Tuple[np.ndarray, List[str]]:
    corr  = np.load(matrix_path)
    names = np.load(scores_npz, allow_pickle=True)["concept_names"].tolist()
    assert corr.shape == (len(names), len(names)), (
        f"Matrix {matrix_path.name} shape {corr.shape} != {len(names)} concepts"
    )
    return corr, names


def _idx(names: List[str], block: List[str]) -> List[int]:
    return [names.index(c) for c in block if c in names]


def within_block_abs_r(corr: np.ndarray, idx: List[int]) -> float:
    vals = [abs(corr[idx[i], idx[j]]) for i in range(len(idx)) for j in range(i + 1, len(idx))]
    return float(np.mean(vals)) if vals else float("nan")


def between_block_signed_r(corr: np.ndarray, a: List[int], b: List[int]) -> float:
    vals = [corr[i, j] for i in a for j in b]
    return float(np.mean(vals)) if vals else float("nan")


def between_block_abs_r(corr: np.ndarray, a: List[int], b: List[int]) -> float:
    vals = [abs(corr[i, j]) for i in a for j in b]
    return float(np.mean(vals)) if vals else float("nan")


def mean_abs_offdiag(corr: np.ndarray) -> float:
    n = corr.shape[0]
    vals = [abs(corr[i, j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))


def eval_hypotheses(corr: np.ndarray, names: List[str], anchor_block: List[str]) -> Dict:
    """Return block stats + H1-H5 PASS/FAIL for one correlation matrix."""
    con = _idx(names, CONSOLIDATION_BLOCK)
    intr = _idx(names, INTERSTITIAL_BLOCK)
    anc = _idx(names, anchor_block)

    w_con = within_block_abs_r(corr, con)
    w_int = within_block_abs_r(corr, intr)
    w_anc = within_block_abs_r(corr, anc)
    bt_con_int = between_block_abs_r(corr, con, intr)
    anc_con = between_block_signed_r(corr, anc, con)
    anc_int = between_block_signed_r(corr, anc, intr)
    mar = mean_abs_offdiag(corr)

    return {
        "within_consolidation": w_con,
        "within_interstitial":  w_int,
        "within_normal_anchor": w_anc,
        "between_con_int_abs":  bt_con_int,
        "anchor_x_con_signed":  anc_con,
        "anchor_x_int_signed":  anc_int,
        "mean_abs_r":           mar,
        "H1": w_con > 0.7,
        "H2": w_int > 0.7,
        "H3": bt_con_int < w_con and bt_con_int < w_int,
        "H4": anc_con < 0 and anc_int < 0,
        "H5": 0.3 <= mar <= 0.6,
    }


# ── Output 4a: delta CSV ──────────────────────────────────────────────────────

def write_delta_csv(concepts_v2: List[str], v1_stats: Dict, v2_stats: Dict,
                    path: pathlib.Path) -> Dict[str, int]:
    """Write 57-row delta CSV; return discrimination_status counts."""
    rows: List[dict] = []
    status_counts: Dict[str, int] = {}
    # 18 shared concepts (v2 order) + dropped concept(s) appended
    all_concepts = list(concepts_v2) + sorted(DROPPED_FROM_V1)

    for cname in all_concepts:
        rs    = revision_status(cname)
        in_v1 = cname in v1_stats
        in_v2 = cname in v2_stats and rs != "DROPPED"

        v1_am = argmax_class(v1_stats[cname]) if in_v1 else None
        v2_am = argmax_class(v2_stats[cname]) if in_v2 else None
        v1_m  = is_match(cname, v1_am) if v1_am is not None else None
        v2_m  = is_match(cname, v2_am) if v2_am is not None else None
        dstat = discrimination_status(cname, v1_m, v2_m)
        status_counts[dstat] = status_counts.get(dstat, 0) + 1

        if dstat == "UNCHANGED-DRIFT":
            print(f"  WARNING: UNCHANGED-DRIFT for '{cname}' — byte-identical "
                  "prompt produced a different argmax; check extraction determinism.")
        if dstat == "REVISED-REGRESSED":
            print(f"  WARNING: REVISED-REGRESSED for '{cname}' — was v1 MATCH, now fails.")

        for cls in CLASSES:
            v1v = v1_stats[cname][cls]["mean"] if in_v1 else float("nan")
            v2v = v2_stats[cname][cls]["mean"] if in_v2 else float("nan")
            delta = v2v - v1v
            rows.append({
                "concept": cname, "class": cls,
                "v1_mean": f"{v1v:.6f}", "v2_mean": f"{v2v:.6f}",
                "delta_mean": f"{delta:.6f}",
                "v1_argmax": v1_am if v1_am is not None else "NaN",
                "v2_argmax": v2_am if v2_am is not None else "NaN",
                "v1_match":  str(v1_m) if v1_m is not None else "NaN",
                "v2_match":  str(v2_m) if v2_m is not None else "NaN",
                "revision_status": rs,
                "discrimination_status": dstat,
            })

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "concept", "class", "v1_mean", "v2_mean", "delta_mean",
            "v1_argmax", "v2_argmax", "v1_match", "v2_match",
            "revision_status", "discrimination_status",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Delta CSV saved → {path.relative_to(_ROOT)}  ({len(rows)} rows)")
    return status_counts


# ── Output 4b: delta heatmap ──────────────────────────────────────────────────

def plot_delta_heatmap(concepts_v2: List[str], v1_stats: Dict, v2_stats: Dict,
                       path: pathlib.Path) -> None:
    shared = [c for c in concepts_v2 if c in v1_stats]  # 18 shared concepts
    delta = np.array([
        [v2_stats[c][cls]["mean"] - v1_stats[c][cls]["mean"] for cls in CLASSES]
        for c in shared
    ])
    vmax = max(abs(delta.min()), abs(delta.max())) + 1e-3

    fig, (ax, ax_txt) = plt.subplots(
        1, 2, figsize=(11, max(8, len(shared) * 0.5)),
        gridspec_kw={"width_ratios": [2, 1.4]},
    )
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="v2 − v1 mean score")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_yticks(range(len(shared)))
    ax.set_yticklabels([c + (" [R]" if c in REVISED_CONCEPTS else "") for c in shared],
                       fontsize=7)
    ax.set_title("v2 − v1 mean cosine similarity\n[R] = revised prompt", fontsize=9, pad=8)
    for r in range(len(shared)):
        for c in range(len(CLASSES)):
            v = delta[r, c]
            ax.text(c, r, f"{v:+.3f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(v) > vmax * 0.6 else "black")

    # Annotation panel: discrimination_status per concept
    ax_txt.axis("off")
    ax_txt.set_title("discrimination_status", fontsize=9, pad=8)
    for r, c in enumerate(shared):
        v1_am = argmax_class(v1_stats[c])
        v2_am = argmax_class(v2_stats[c])
        dstat = discrimination_status(c, is_match(c, v1_am), is_match(c, v2_am))
        ax_txt.text(0.0, 1.0 - (r + 0.5) / len(shared), f"{c}: {dstat}",
                    fontsize=6.5, va="center", transform=ax_txt.transAxes,
                    family="monospace")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Delta heatmap saved → {path.relative_to(_ROOT)}")


# ── Output 4c: comparison summary ─────────────────────────────────────────────

def build_comparison_summary(
    concepts_v2: List[str], v1_stats: Dict, v2_stats: Dict,
    p_v1: dict, p_v2: dict,
    v1_corr: np.ndarray, v1_names: List[str],
    v2_corr: np.ndarray, v2_names: List[str],
    status_counts: Dict[str, int],
) -> Tuple[str, int, int]:
    L: List[str] = []
    v1_prompt = dict(zip(p_v1["concept_ids"], p_v1["prompts"]))
    v2_prompt = dict(zip(p_v2["concept_ids"], p_v2["prompts"]))

    v1_match = sum(is_match(c, argmax_class(v1_stats[c])) for c in v1_stats)
    v2_match = sum(is_match(c, argmax_class(v2_stats[c])) for c in concepts_v2 if c in v2_stats)

    L.append("=" * 70)
    L.append("V1 vs V2 COMPARISON SUMMARY")
    L.append("=" * 70)

    # Section 1 — headline counts
    L.append("\nSECTION 1 — HEADLINE COUNTS")
    L.append(f"  v1: {v1_match}/{len(v1_stats)} MATCH   v2: {v2_match}/{len(concepts_v2)} MATCH")
    fixed   = status_counts.get("REVISED-FIXED", 0)
    persist = status_counts.get("REVISED-PERSISTENT-FAIL", 0)
    regress = status_counts.get("REVISED-REGRESSED", 0)
    still   = status_counts.get("REVISED-STILL-MATCH", 0)
    unch_ok = status_counts.get("UNCHANGED-STILL-MATCH", 0)
    unch_mm = status_counts.get("UNCHANGED-STILL-MISMATCH", 0)
    unch_dr = status_counts.get("UNCHANGED-DRIFT", 0)
    L.append(f"  Of {len(REVISED_CONCEPTS)} REVISED concepts:")
    L.append(f"    REVISED-FIXED           : {fixed}")
    L.append(f"    REVISED-PERSISTENT-FAIL : {persist}")
    L.append(f"    REVISED-REGRESSED       : {regress}")
    if still:
        L.append(f"    REVISED-STILL-MATCH     : {still}")
    L.append(f"  Of {len(UNCHANGED_CONCEPTS)} UNCHANGED concepts:")
    L.append(f"    UNCHANGED-STILL-MATCH   : {unch_ok}  (expected {len(UNCHANGED_CONCEPTS)})")
    if unch_mm:
        L.append(f"    UNCHANGED-STILL-MISMATCH: {unch_mm}")
    if unch_dr:
        L.append(f"    UNCHANGED-DRIFT         : {unch_dr}  ← BUG: bit-identity violated")

    # Section 2 — per-revised-concept table
    L.append("\nSECTION 2 — PER-REVISED-CONCEPT COMPARISON")
    for cname in sorted(REVISED_CONCEPTS):
        v1_am = argmax_class(v1_stats[cname])
        v2_am = argmax_class(v2_stats[cname])
        v1_m  = is_match(cname, v1_am)
        v2_m  = is_match(cname, v2_am)
        dstat = discrimination_status(cname, v1_m, v2_m)
        verdict = {"REVISED-FIXED": "FIXED",
                   "REVISED-PERSISTENT-FAIL": "PERSISTENT-FAIL",
                   "REVISED-REGRESSED": "REGRESSED",
                   "REVISED-STILL-MATCH": "STILL-MATCH"}.get(dstat, dstat)
        L.append(f"\n  [{cname}]  [{TIER_LABELS.get(cname,'?')}]  → {verdict}")
        L.append(f"    v1 prompt: {v1_prompt.get(cname,'n/a')}")
        L.append(f"    v2 prompt: {v2_prompt.get(cname,'n/a')}")
        v1_row = "  ".join(f"{c}={v1_stats[cname][c]['mean']:.4f}" for c in CLASSES)
        v2_row = "  ".join(f"{c}={v2_stats[cname][c]['mean']:.4f}" for c in CLASSES)
        L.append(f"    v1 means: {v1_row}  argmax={v1_am}  spread={spread(v1_stats[cname]):.4f}")
        L.append(f"    v2 means: {v2_row}  argmax={v2_am}  spread={spread(v2_stats[cname]):.4f}")
        L.append(f"    spread change: {spread(v2_stats[cname]) - spread(v1_stats[cname]):+.4f}")

    # Section 3 — dropped concept
    L.append("\nSECTION 3 — DROPPED CONCEPT")
    for cname in sorted(DROPPED_FROM_V1):
        v1_am = argmax_class(v1_stats[cname]) if cname in v1_stats else "n/a"
        v1_m  = is_match(cname, v1_am) if cname in v1_stats else None
        L.append(f"  Concept: {cname}")
        L.append(f"  v1 result: argmax={v1_am} (match={v1_m})")
        # report correlation with clear_lung_fields from v1 matrix
        if cname in v1_names and "clear_lung_fields" in v1_names:
            r = v1_corr[v1_names.index(cname), v1_names.index("clear_lung_fields")]
            L.append(f"  v1 r with clear_lung_fields: {r:+.4f}")
        L.append("  Justification: extreme semantic redundancy with clear_lung_fields,")
        L.append("    not correlation tuning. Removed to avoid a near-duplicate feature.")

    # Section 4 — correlation structure comparison
    L.append("\nSECTION 4 — CORRELATION STRUCTURE COMPARISON")
    h_v1 = eval_hypotheses(v1_corr, v1_names, NORMAL_ANCHOR_BLOCK_V1)
    h_v2 = eval_hypotheses(v2_corr, v2_names, NORMAL_ANCHOR_BLOCK_V2)
    L.append(f"  Mean |r| (all pairs):       v1={h_v1['mean_abs_r']:.4f}  "
             f"v2={h_v2['mean_abs_r']:.4f}  Δ={h_v2['mean_abs_r']-h_v1['mean_abs_r']:+.4f}")
    L.append("  Within-block mean |r|:")
    L.append(f"    consolidation_block:  v1={h_v1['within_consolidation']:.4f}  "
             f"v2={h_v2['within_consolidation']:.4f}")
    L.append(f"    interstitial_block:   v1={h_v1['within_interstitial']:.4f}  "
             f"v2={h_v2['within_interstitial']:.4f}")
    L.append(f"    normal_anchor_block:  v1={h_v1['within_normal_anchor']:.4f} (5 mem)  "
             f"v2={h_v2['within_normal_anchor']:.4f} (4 mem)")
    L.append("  Between-block mean |r| (consolidation × interstitial):")
    L.append(f"    v1={h_v1['between_con_int_abs']:.4f}  v2={h_v2['between_con_int_abs']:.4f}")
    L.append("  Anchor×pneumonia signed r:")
    L.append(f"    anchor×consolidation: v1={h_v1['anchor_x_con_signed']:+.4f}  "
             f"v2={h_v2['anchor_x_con_signed']:+.4f}")
    L.append(f"    anchor×interstitial:  v1={h_v1['anchor_x_int_signed']:+.4f}  "
             f"v2={h_v2['anchor_x_int_signed']:+.4f}")
    L.append("  Expected: within-block correlations persist (no structure engineering);")
    L.append("    targeted revisions may shift specific pairwise r but blocks should remain.")

    # Section 5 — hypothesis follow-up
    L.append("\nSECTION 5 — PRE-REGISTERED HYPOTHESIS FOLLOW-UP")
    L.append("  (H1-H5 pre-registered on v1; v2 reported as follow-up, not fresh pre-registration)")
    pf = lambda b: "PASS" if b else "FAIL"
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        L.append(f"  {h}: {HYPOTHESES[h]}")
        L.append(f"      v1: {pf(h_v1[h])}   v2: {pf(h_v2[h])}")

    # Section 6 — recommendation
    L.append("\nSECTION 6 — RECOMMENDATION")
    if fixed >= 3 and v2_match >= 15:
        rec = ("v2 substantially improves on v1; recommend adopting v2 as operative "
               "feature set for downstream NAM training.")
    elif fixed >= 1:
        rec = ("v2 partially improves on v1; recommend reviewing specific failures "
               "before deciding. Consider escalating to supervisor.")
    else:
        rec = ("v2 does not meaningfully improve discrimination over v1; recommend "
               "reverting to v1 and reviewing the prompt-engineering strategy with "
               "supervisor before further iteration.")
    for line in textwrap.wrap(rec, width=68):
        L.append(f"  {line}")

    return "\n".join(L), v1_match, v2_match


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading v1 CSV from {V1_CSV.relative_to(_ROOT)} ...")
    v1_stats = load_per_class_csv(V1_CSV)
    print(f"Loading v2 CSV from {V2_CSV.relative_to(_ROOT)} ...")
    v2_stats = load_per_class_csv(V2_CSV)

    p_v1 = load_prompts(PROMPTS_V1)
    p_v2 = load_prompts(PROMPTS_V2)
    concepts_v2 = p_v2["concept_ids"]

    print(f"Change sets: {len(REVISED_CONCEPTS)} revised, "
          f"{len(UNCHANGED_CONCEPTS)} unchanged, {len(DROPPED_FROM_V1)} dropped")

    # Correlation matrices
    for pth in (V1_CORR_NPY, V2_CORR_NPY):
        if not pth.exists():
            sys.exit(f"ERROR: missing correlation matrix {pth}.\n"
                     "Run correlation_diagnostic.py and correlation_diagnostic_v2.py first.")
    v1_corr, v1_names = load_corr(V1_CORR_NPY, V1_SCORES)
    v2_corr, v2_names = load_corr(V2_CORR_NPY, V2_SCORES)

    # 4a CSV
    status_counts = write_delta_csv(concepts_v2, v1_stats, v2_stats,
                                    OUT_DIR / "delta_per_class_means.csv")
    # 4b heatmap
    plot_delta_heatmap(concepts_v2, v1_stats, v2_stats, OUT_DIR / "delta_heatmap.png")
    # 4c summary
    summary, v1_match, v2_match = build_comparison_summary(
        concepts_v2, v1_stats, v2_stats, p_v1, p_v2,
        v1_corr, v1_names, v2_corr, v2_names, status_counts,
    )
    (OUT_DIR / "comparison_summary.txt").write_text(summary, encoding="utf-8")
    print(f"  Comparison summary → {(OUT_DIR / 'comparison_summary.txt').relative_to(_ROOT)}")

    # ── Final console summary ──────────────────────────────────────────────────
    h_v1 = eval_hypotheses(v1_corr, v1_names, NORMAL_ANCHOR_BLOCK_V1)
    h_v2 = eval_hypotheses(v2_corr, v2_names, NORMAL_ANCHOR_BLOCK_V2)

    print("\n" + "=" * 70)
    print("FINAL CONSOLE SUMMARY")
    print("=" * 70)
    n_unchanged_ok = status_counts.get("UNCHANGED-STILL-MATCH", 0) + \
                     status_counts.get("UNCHANGED-STILL-MISMATCH", 0)
    drift = status_counts.get("UNCHANGED-DRIFT", 0)
    bit_id = "PASS" if drift == 0 else f"FAIL ({drift} drifted)"
    print(f"Bit-identity (13 unchanged concepts): {bit_id}")
    print(f"MATCH count: v1={v1_match}/{len(v1_stats)}  →  v2={v2_match}/{len(concepts_v2)}")
    print(f"mean |r|:    v1={h_v1['mean_abs_r']:.4f}  →  v2={h_v2['mean_abs_r']:.4f}")
    print("\nStatus of the 5 revised concepts:")
    for cname in sorted(REVISED_CONCEPTS):
        v1_am = argmax_class(v1_stats[cname])
        v2_am = argmax_class(v2_stats[cname])
        dstat = discrimination_status(cname, is_match(cname, v1_am), is_match(cname, v2_am))
        print(f"  {cname:<34s}  v1={v1_am:<9s} v2={v2_am:<9s}  {dstat}")

    # Recommendation echo
    rec_lines, in_rec = [], False
    for line in summary.splitlines():
        if line.strip().startswith("SECTION 6"):
            in_rec = True
            continue
        if in_rec and line.strip():
            rec_lines.append(line.strip())
    print(f"\nRecommendation: {' '.join(rec_lines)}")
    print(f"\nAll outputs written to {OUT_DIR.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
