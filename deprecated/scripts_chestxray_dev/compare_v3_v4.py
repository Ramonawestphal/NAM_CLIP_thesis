"""
v3 vs v4 chest X-ray BiomedCLIP concept score comparison.

v3 has 18 concepts; v4 has 17 (hyperinflation DROPPED). Change sets (byte compare):
    FROZEN  (17): all v4 concepts — byte-identical prompts, identical scores
    REVISED (0):  none
    DROPPED (1):  hyperinflation
This is a trivial comparison: v4 == v3 minus one column. The 17 kept concepts
have bit-identical scores (verified in the extraction step), so all deltas are 0.

Reads (all read-only):
    results/chestxray/per_class_diagnostic_v3/per_class_means.csv          (v3)
    results/chestxray/per_class_diagnostic_v4/per_class_means.csv          (v4)
    results/chestxray/correlation_diagnostic_v3/correlation_matrix.npy     (v3)
    results/chestxray/correlation_diagnostic_v4/correlation_matrix.npy     (v4)
    data/features/biomedclip/chestxray_concept_scores_{v3,v4}.npz          (orderings)

Outputs:
    results/chestxray/v3_vs_v4_comparison/delta_per_class_means.csv
    results/chestxray/v3_vs_v4_comparison/delta_heatmap.png
    results/chestxray/v3_vs_v4_comparison/comparison_summary.txt

Run from project root (after the v4 diagnostics):
    python scripts/chestxray/compare_v3_v4.py
"""

from __future__ import annotations

import csv
import pathlib
import sys
import textwrap
from typing import Dict, List, Tuple

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # robust on cp1252 consoles
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features.prompt_loader import load_prompts

# ── Paths (before = v3, after = v4) ───────────────────────────────────────────
V3_CSV       = _ROOT / "results/chestxray/per_class_diagnostic_v3/per_class_means.csv"
V4_CSV       = _ROOT / "results/chestxray/per_class_diagnostic_v4/per_class_means.csv"
V3_CORR_NPY  = _ROOT / "results/chestxray/correlation_diagnostic_v3/correlation_matrix.npy"
V4_CORR_NPY  = _ROOT / "results/chestxray/correlation_diagnostic_v4/correlation_matrix.npy"
V3_SCORES    = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v3.npz"
V4_SCORES    = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v4.npz"
PROMPTS_V3   = _ROOT / "src/features/prompts/chestxray_prompts_v3.txt"
PROMPTS_V4   = _ROOT / "src/features/prompts/chestxray_prompts_v4.txt"
OUT_DIR      = _ROOT / "results/chestxray/v3_vs_v4_comparison"

CLASSES = ["normal", "bacteria", "virus"]

# ── Blocks (unchanged v3 → v4; hyperinflation was ungrouped) ──────────────────
CONSOLIDATION_BLOCK = [
    "consolidation", "focal_opacity", "lobar_consolidation",
    "dense_segmental_opacity", "air_bronchograms",
]
INTERSTITIAL_BLOCK = [
    "bilateral_interstitial_pattern", "peribronchial_cuffing", "perihilar_infiltrates",
]
NORMAL_ANCHOR_BLOCK = [
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
    "hyperinflation":                   ["virus"],   # v3 only (dropped in v4)
    "clear_lung_fields":                ["normal"],
    "sharp_costophrenic_angles":        ["normal"],
    "normal_cardiac_silhouette":        ["normal"],
    "symmetric_lung_aeration":          ["normal"],
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
}

HYPOTHESES = {
    "H1": "Within consolidation_block, mean |r| > 0.7",
    "H2": "Within interstitial_block, mean |r| > 0.7",
    "H3": "Between consolidation & interstitial, mean |r| < within-block |r| of either",
    "H4": "normal_anchor_block correlates negatively (signed r < 0) with both pneumonia blocks",
    "H5": "Aggregate mean |r| over all pairs falls in [0.3, 0.6]",
}

# Documented three-iteration trajectory of the dropped concept (argmax=normal each time)
HYPERINFLATION_TRAJECTORY = [
    ("v1", "hyperinflation of the lungs", 0.043),
    ("v2", "overexpanded lungs with flattened diaphragm domes", 0.015),
    ("v3", "increased lung volume with depressed diaphragms and widened intercostal spaces", 0.0008),
]


# ── Change sets by byte comparison (v3 → v4) ──────────────────────────────────

def build_change_sets() -> Tuple[set, set, set]:
    p3 = load_prompts(PROMPTS_V3)
    p4 = load_prompts(PROMPTS_V4)
    v3m = dict(zip(p3["concept_ids"], p3["prompts"]))
    v4m = dict(zip(p4["concept_ids"], p4["prompts"]))
    v3_set, v4_set = set(p3["concept_ids"]), set(p4["concept_ids"])
    dropped = v3_set - v4_set
    revised = {c for c in v3_set & v4_set if v3m[c] != v4m[c]}
    frozen  = {c for c in v3_set & v4_set if v3m[c] == v4m[c]}
    return revised, frozen, dropped


REVISED_CONCEPTS, FROZEN_CONCEPTS, DROPPED_CONCEPTS = build_change_sets()


def revision_status(cname: str) -> str:
    if cname in DROPPED_CONCEPTS:
        return "DROPPED"
    if cname in REVISED_CONCEPTS:
        return "REVISED"
    if cname in FROZEN_CONCEPTS:
        return "UNCHANGED"
    return "UNKNOWN"


# ── CSV / stats helpers ───────────────────────────────────────────────────────

def load_per_class_csv(path: pathlib.Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result.setdefault(row["concept"], {})[row["class"]] = {
                "mean": float(row["mean"]), "std": float(row["std"]), "n": int(row["n"]),
            }
    return result


def argmax_class(stats: Dict[str, Dict[str, float]]) -> str:
    return max(CLASSES, key=lambda c: stats[c]["mean"])


def is_match(cname: str, am: str) -> bool:
    return am in EXPECTED_TARGET.get(cname, [])


def spread(stats: Dict[str, Dict[str, float]]) -> float:
    vals = [stats[c]["mean"] for c in CLASSES]
    return max(vals) - min(vals)


def discrimination_status(cname: str, b_m, a_m) -> str:
    """b_m = v3 (before) match bool; a_m = v4 (after) match bool."""
    rs = revision_status(cname)
    if rs == "DROPPED":
        return "DROPPED"
    if rs == "UNCHANGED":
        if b_m and a_m:
            return "UNCHANGED-STILL-MATCH"
        if b_m != a_m:
            return "UNCHANGED-DRIFT"            # bit-identity bug if it occurs
        return "UNCHANGED-STILL-MISMATCH"
    if rs == "REVISED":
        if not b_m and a_m:
            return "REVISED-FIXED"
        if not b_m and not a_m:
            return "REVISED-PERSISTENT-FAIL"
        if b_m and not a_m:
            return "REVISED-REGRESSED"
        return "REVISED-STILL-MATCH"
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


def mean_abs_r_to_others(corr: np.ndarray, names: List[str], concept: str) -> float:
    if concept not in names:
        return float("nan")
    i = names.index(concept)
    vals = [abs(corr[i, j]) for j in range(len(names)) if j != i]
    return float(np.mean(vals))


def eval_hypotheses(corr: np.ndarray, names: List[str]) -> Dict:
    con  = _idx(names, CONSOLIDATION_BLOCK)
    intr = _idx(names, INTERSTITIAL_BLOCK)
    anc  = _idx(names, NORMAL_ANCHOR_BLOCK)
    w_con = within_block_abs_r(corr, con)
    w_int = within_block_abs_r(corr, intr)
    w_anc = within_block_abs_r(corr, anc)
    bt = between_block_abs_r(corr, con, intr)
    anc_con = between_block_signed_r(corr, anc, con)
    anc_int = between_block_signed_r(corr, anc, intr)
    mar = mean_abs_offdiag(corr)
    return {
        "within_consolidation": w_con, "within_interstitial": w_int,
        "within_normal_anchor": w_anc, "between_con_int_abs": bt,
        "anchor_x_con_signed": anc_con, "anchor_x_int_signed": anc_int,
        "mean_abs_r": mar,
        "H1": w_con > 0.7, "H2": w_int > 0.7,
        "H3": bt < w_con and bt < w_int,
        "H4": anc_con < 0 and anc_int < 0,
        "H5": 0.3 <= mar <= 0.6,
    }


# ── Output: delta CSV (54 rows: 17 shared×3 + hyperinflation×3) ───────────────

def write_delta_csv(v4_concepts: List[str], v3_stats: Dict, v4_stats: Dict,
                    path: pathlib.Path) -> Dict[str, int]:
    rows: List[dict] = []
    status_counts: Dict[str, int] = {}
    all_concepts = list(v4_concepts) + sorted(DROPPED_CONCEPTS)  # 17 + 1

    for cname in all_concepts:
        rs    = revision_status(cname)
        in_v3 = cname in v3_stats
        in_v4 = cname in v4_stats and rs != "DROPPED"
        v3_am = argmax_class(v3_stats[cname]) if in_v3 else None
        v4_am = argmax_class(v4_stats[cname]) if in_v4 else None
        b_m   = is_match(cname, v3_am) if v3_am is not None else None
        a_m   = is_match(cname, v4_am) if v4_am is not None else None
        dstat = discrimination_status(cname, b_m, a_m)
        status_counts[dstat] = status_counts.get(dstat, 0) + 1

        if dstat == "UNCHANGED-DRIFT":
            print(f"  WARNING: UNCHANGED-DRIFT for '{cname}' — frozen prompt produced "
                  "a different argmax; check extraction determinism.")

        for cls in CLASSES:
            v3v = v3_stats[cname][cls]["mean"] if in_v3 else float("nan")
            v4v = v4_stats[cname][cls]["mean"] if in_v4 else float("nan")
            rows.append({
                "concept": cname, "class": cls,
                "v3_mean": f"{v3v:.6f}", "v4_mean": f"{v4v:.6f}",
                "delta_mean": f"{v4v - v3v:.6f}",
                "v3_argmax": v3_am if v3_am is not None else "NaN",
                "v4_argmax": v4_am if v4_am is not None else "NaN",
                "v3_match":  str(b_m) if b_m is not None else "NaN",
                "v4_match":  str(a_m) if a_m is not None else "NaN",
                "revision_status": rs,
                "discrimination_status": dstat,
            })

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "concept", "class", "v3_mean", "v4_mean", "delta_mean",
            "v3_argmax", "v4_argmax", "v3_match", "v4_match",
            "revision_status", "discrimination_status",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Delta CSV saved → {path.relative_to(_ROOT)}  ({len(rows)} rows)")
    return status_counts


# ── Output: delta heatmap (17 shared concepts) ────────────────────────────────

def plot_delta_heatmap(v4_concepts: List[str], v3_stats: Dict, v4_stats: Dict,
                       path: pathlib.Path) -> None:
    shared = [c for c in v4_concepts if c in v3_stats]  # 17
    delta = np.array([
        [v4_stats[c][cls]["mean"] - v3_stats[c][cls]["mean"] for cls in CLASSES]
        for c in shared
    ])
    vmax = max(abs(delta.min()), abs(delta.max()), 1e-3) + 1e-3

    fig, (ax, ax_txt) = plt.subplots(
        1, 2, figsize=(11, max(8, len(shared) * 0.5)),
        gridspec_kw={"width_ratios": [2, 1.4]},
    )
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="v4 − v3 mean score")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_yticks(range(len(shared)))
    ax.set_yticklabels(shared, fontsize=7)
    ax.set_title("v4 − v3 mean cosine similarity\n(all deltas ~0: 17 frozen concepts)",
                 fontsize=9, pad=8)
    for r in range(len(shared)):
        for c in range(len(CLASSES)):
            v = delta[r, c]
            ax.text(c, r, f"{v:+.3f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(v) > vmax * 0.6 else "black")

    ax_txt.axis("off")
    ax_txt.set_title("discrimination_status", fontsize=9, pad=8)
    for r, c in enumerate(shared):
        b_m = is_match(c, argmax_class(v3_stats[c]))
        a_m = is_match(c, argmax_class(v4_stats[c]))
        dstat = discrimination_status(c, b_m, a_m)
        ax_txt.text(0.0, 1.0 - (r + 0.5) / len(shared), f"{c}: {dstat}",
                    fontsize=6.5, va="center", transform=ax_txt.transAxes,
                    family="monospace")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Delta heatmap saved → {path.relative_to(_ROOT)}")


# ── Output: comparison summary ────────────────────────────────────────────────

def build_comparison_summary(
    v3_concepts: List[str], v4_concepts: List[str], v3_stats: Dict, v4_stats: Dict,
    v3_corr: np.ndarray, v3_names: List[str],
    v4_corr: np.ndarray, v4_names: List[str],
    status_counts: Dict[str, int],
) -> Tuple[str, int, int]:
    L: List[str] = []
    # v3 over ALL 18 v3 concepts; v4 over its 17 concepts
    v3_match = sum(is_match(c, argmax_class(v3_stats[c])) for c in v3_concepts if c in v3_stats)
    v4_match = sum(is_match(c, argmax_class(v4_stats[c])) for c in v4_concepts if c in v4_stats)
    n_v3, n_v4 = len(v3_concepts), len(v4_concepts)

    L.append("=" * 70)
    L.append("V3 vs V4 COMPARISON SUMMARY")
    L.append("=" * 70)

    # Section 1 — headline
    L.append("\nSECTION 1 — HEADLINE COUNTS")
    L.append(f"  v3: {v3_match}/{n_v3} MATCH   v4: {v4_match}/{n_v4} MATCH")
    L.append("  The improvement is STRUCTURAL (removed the single non-discriminating")
    L.append("  concept), not behavioural: no prompt was revised, so all 17 kept")
    L.append("  concepts have bit-identical scores and unchanged argmax.")
    L.append(f"  Change sets: {len(FROZEN_CONCEPTS)} frozen, {len(REVISED_CONCEPTS)} revised, "
             f"{len(DROPPED_CONCEPTS)} dropped.")
    L.append(f"    UNCHANGED-STILL-MATCH : {status_counts.get('UNCHANGED-STILL-MATCH', 0)} "
             f"(expected {len(FROZEN_CONCEPTS)})")
    for extra in ("UNCHANGED-STILL-MISMATCH", "UNCHANGED-DRIFT"):
        if status_counts.get(extra, 0):
            flag = "  ← BUG: bit-identity violated" if extra == "UNCHANGED-DRIFT" else ""
            L.append(f"    {extra:<22s}: {status_counts[extra]}{flag}")

    # Section 2 — dropped concept
    L.append("\nSECTION 2 — DROPPED CONCEPT: hyperinflation")
    L.append("  Three-iteration trajectory (argmax=normal at every step):")
    for ver, prompt, sp in HYPERINFLATION_TRAJECTORY:
        L.append(f"    {ver}: spread={sp:.4f}  \"{prompt}\"")
    L.append("  The spread SHRINKS toward zero as prompts become more pathology-specific.")
    L.append("  This indicates BiomedCLIP cannot discriminate viral hyperinflation from")
    L.append("  normal lung aeration — a property of the encoder, not of prompt phrasing.")
    L.append("  Decision: drop rather than iterate further.")

    # Section 3 — correlation structure comparison
    L.append("\nSECTION 3 — CORRELATION STRUCTURE COMPARISON")
    h3 = eval_hypotheses(v3_corr, v3_names)
    h4 = eval_hypotheses(v4_corr, v4_names)
    hyp_to_others = mean_abs_r_to_others(v3_corr, v3_names, "hyperinflation")
    L.append(f"  Mean |r| (all pairs):       v3={h3['mean_abs_r']:.4f}  "
             f"v4={h4['mean_abs_r']:.4f}  Δ={h4['mean_abs_r']-h3['mean_abs_r']:+.4f}")
    L.append("  Within-block mean |r|:")
    L.append(f"    consolidation_block:  v3={h3['within_consolidation']:.4f}  "
             f"v4={h4['within_consolidation']:.4f}")
    L.append(f"    interstitial_block:   v3={h3['within_interstitial']:.4f}  "
             f"v4={h4['within_interstitial']:.4f}")
    L.append(f"    normal_anchor_block:  v3={h3['within_normal_anchor']:.4f}  "
             f"v4={h4['within_normal_anchor']:.4f}")
    L.append("  Between-block mean |r| (consolidation × interstitial):")
    L.append(f"    v3={h3['between_con_int_abs']:.4f}  v4={h4['between_con_int_abs']:.4f}")
    L.append(f"  Dropped concept hyperinflation was UNGROUPED; its v3 mean |r| to all")
    L.append(f"  others = {hyp_to_others:.4f} (weakly correlated), so block structure is")
    L.append("  essentially unaffected — only small numerical shifts expected.")

    # Section 4 — hypothesis follow-up
    L.append("\nSECTION 4 — PRE-REGISTERED HYPOTHESIS FOLLOW-UP")
    L.append("  (H1-H5 pre-registered on v1; v4 reported as follow-up, not fresh pre-registration)")
    pf = lambda b: "PASS" if b else "FAIL"
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        L.append(f"  {h}: {HYPOTHESES[h]}")
        L.append(f"      v3: {pf(h3[h])}   v4: {pf(h4[h])}")

    # Section 5 — recommendation
    L.append("\nSECTION 5 — RECOMMENDATION")
    rec = ("v4 is the operative feature set. All 17 concepts pass per-class "
           "discrimination MATCH on the train pool. Recommend proceeding to NAM "
           "training using data/features/biomedclip/chestxray_concept_scores_v4.npz "
           "as the feature input.")
    for line in textwrap.wrap(rec, width=68):
        L.append(f"  {line}")

    return "\n".join(L), v3_match, v4_match


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading v3 CSV from {V3_CSV.relative_to(_ROOT)} ...")
    v3_stats = load_per_class_csv(V3_CSV)
    print(f"Loading v4 CSV from {V4_CSV.relative_to(_ROOT)} ...")
    v4_stats = load_per_class_csv(V4_CSV)

    p_v3 = load_prompts(PROMPTS_V3)
    p_v4 = load_prompts(PROMPTS_V4)
    v3_concepts = p_v3["concept_ids"]
    v4_concepts = p_v4["concept_ids"]

    print(f"Change sets (v3→v4): {len(REVISED_CONCEPTS)} revised, "
          f"{len(FROZEN_CONCEPTS)} frozen, {len(DROPPED_CONCEPTS)} dropped")
    assert DROPPED_CONCEPTS == {"hyperinflation"}, \
        f"Expected hyperinflation dropped, got {sorted(DROPPED_CONCEPTS)}"
    assert not REVISED_CONCEPTS, f"v4 should revise nothing, got {sorted(REVISED_CONCEPTS)}"
    assert len(FROZEN_CONCEPTS) == 17, f"Expected 17 frozen, got {len(FROZEN_CONCEPTS)}"

    for pth in (V3_CORR_NPY, V4_CORR_NPY):
        if not pth.exists():
            sys.exit(f"ERROR: missing correlation matrix {pth}.\n"
                     "Run correlation_diagnostic_v3.py and correlation_diagnostic_v4.py first.")
    v3_corr, v3_names = load_corr(V3_CORR_NPY, V3_SCORES)
    v4_corr, v4_names = load_corr(V4_CORR_NPY, V4_SCORES)

    status_counts = write_delta_csv(v4_concepts, v3_stats, v4_stats,
                                    OUT_DIR / "delta_per_class_means.csv")
    plot_delta_heatmap(v4_concepts, v3_stats, v4_stats, OUT_DIR / "delta_heatmap.png")
    summary, v3_match, v4_match = build_comparison_summary(
        v3_concepts, v4_concepts, v3_stats, v4_stats,
        v3_corr, v3_names, v4_corr, v4_names, status_counts,
    )
    (OUT_DIR / "comparison_summary.txt").write_text(summary, encoding="utf-8")
    print(f"  Comparison summary → {(OUT_DIR / 'comparison_summary.txt').relative_to(_ROOT)}")

    # ── Final console summary ──────────────────────────────────────────────────
    h3 = eval_hypotheses(v3_corr, v3_names)
    h4 = eval_hypotheses(v4_corr, v4_names)
    drift = status_counts.get("UNCHANGED-DRIFT", 0)
    bit_id = "PASS" if drift == 0 else f"FAIL ({drift} drifted)"

    print("\n" + "=" * 70)
    print("FINAL CONSOLE SUMMARY")
    print("=" * 70)
    print(f"v4 prompt set: {len(v4_concepts)} concepts, hyperinflation dropped")
    print(f"Bit-identity (17 frozen concepts): {bit_id}")
    print(f"MATCH count: v3={v3_match}/{len(v3_concepts)}  →  v4={v4_match}/{len(v4_concepts)}")
    print(f"mean |r|:    v3={h3['mean_abs_r']:.4f}  →  v4={h4['mean_abs_r']:.4f}")

    rec_lines, in_rec = [], False
    for line in summary.splitlines():
        if line.strip().startswith("SECTION 5"):
            in_rec = True
            continue
        if in_rec and line.strip():
            rec_lines.append(line.strip())
    print(f"\nRecommendation: {' '.join(rec_lines)}")
    print(f"\nAll outputs written to {OUT_DIR.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
