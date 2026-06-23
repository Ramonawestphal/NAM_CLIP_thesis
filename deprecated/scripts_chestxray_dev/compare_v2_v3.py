"""
v2 vs v3 chest X-ray BiomedCLIP concept score comparison.

v2 and v3 both have 18 concepts. Change sets (by byte comparison of prompt text):
    REVISED (2): bilateral_interstitial_pattern, hyperinflation
    FROZEN  (16): the remaining concepts (byte-identical, frozen from v2)
    DROPPED (0): none
(Annotation parsing is not used here: frozen v3 blocks carry their verbatim v2
annotations, so byte comparison is the reliable signal.)

Reads (all read-only):
    results/chestxray/per_class_diagnostic_v2/per_class_means.csv          (v2)
    results/chestxray/per_class_diagnostic_v3/per_class_means.csv          (v3)
    results/chestxray/correlation_diagnostic_v2/correlation_matrix.npy     (v2)
    results/chestxray/correlation_diagnostic_v3/correlation_matrix.npy     (v3)
    data/features/biomedclip/chestxray_concept_scores_{v2,v3}.npz          (orderings)

Outputs:
    results/chestxray/v2_vs_v3_comparison/delta_per_class_means.csv
    results/chestxray/v2_vs_v3_comparison/delta_heatmap.png
    results/chestxray/v2_vs_v3_comparison/comparison_summary.txt

Run from project root (after the v3 diagnostics):
    python scripts/chestxray/compare_v2_v3.py
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

# ── Paths (before = v2, after = v3) ───────────────────────────────────────────
V2_CSV       = _ROOT / "results/chestxray/per_class_diagnostic_v2/per_class_means.csv"
V3_CSV       = _ROOT / "results/chestxray/per_class_diagnostic_v3/per_class_means.csv"
V2_CORR_NPY  = _ROOT / "results/chestxray/correlation_diagnostic_v2/correlation_matrix.npy"
V3_CORR_NPY  = _ROOT / "results/chestxray/correlation_diagnostic_v3/correlation_matrix.npy"
V2_SCORES    = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v2.npz"
V3_SCORES    = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v3.npz"
PROMPTS_V2   = _ROOT / "src/features/prompts/chestxray_prompts_v2.txt"
PROMPTS_V3   = _ROOT / "src/features/prompts/chestxray_prompts_v3.txt"
OUT_DIR      = _ROOT / "results/chestxray/v2_vs_v3_comparison"

CLASSES = ["normal", "bacteria", "virus"]

# ── Blocks (identical for v2 and v3; anchor block has 4 members) ──────────────
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
    "hyperinflation":                   ["virus"],
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

VIRAL_TIER = ["bilateral_interstitial_pattern", "peribronchial_cuffing",
              "perihilar_infiltrates", "hyperinflation"]


# ── Change sets by byte comparison (v2 → v3) ──────────────────────────────────

def build_change_sets() -> Tuple[set, set, set]:
    p2 = load_prompts(PROMPTS_V2)
    p3 = load_prompts(PROMPTS_V3)
    v2m = dict(zip(p2["concept_ids"], p2["prompts"]))
    v3m = dict(zip(p3["concept_ids"], p3["prompts"]))
    v2_set, v3_set = set(p2["concept_ids"]), set(p3["concept_ids"])
    dropped = v2_set - v3_set
    revised = {c for c in v2_set & v3_set if v2m[c] != v3m[c]}
    frozen  = {c for c in v2_set & v3_set if v2m[c] == v3m[c]}
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
    """b_m = v2 (before) match bool; a_m = v3 (after) match bool."""
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


# ── Output: delta CSV ─────────────────────────────────────────────────────────

def write_delta_csv(concepts: List[str], v2_stats: Dict, v3_stats: Dict,
                    path: pathlib.Path) -> Dict[str, int]:
    rows: List[dict] = []
    status_counts: Dict[str, int] = {}
    all_concepts = list(concepts) + sorted(DROPPED_CONCEPTS)  # 18 + 0

    for cname in all_concepts:
        rs    = revision_status(cname)
        in_v2 = cname in v2_stats
        in_v3 = cname in v3_stats and rs != "DROPPED"
        v2_am = argmax_class(v2_stats[cname]) if in_v2 else None
        v3_am = argmax_class(v3_stats[cname]) if in_v3 else None
        b_m   = is_match(cname, v2_am) if v2_am is not None else None
        a_m   = is_match(cname, v3_am) if v3_am is not None else None
        dstat = discrimination_status(cname, b_m, a_m)
        status_counts[dstat] = status_counts.get(dstat, 0) + 1

        if dstat == "UNCHANGED-DRIFT":
            print(f"  WARNING: UNCHANGED-DRIFT for '{cname}' — frozen prompt produced "
                  "a different argmax; check extraction determinism.")
        if dstat == "REVISED-REGRESSED":
            print(f"  WARNING: REVISED-REGRESSED for '{cname}' — was v2 MATCH, now fails.")

        for cls in CLASSES:
            v2v = v2_stats[cname][cls]["mean"] if in_v2 else float("nan")
            v3v = v3_stats[cname][cls]["mean"] if in_v3 else float("nan")
            rows.append({
                "concept": cname, "class": cls,
                "v2_mean": f"{v2v:.6f}", "v3_mean": f"{v3v:.6f}",
                "delta_mean": f"{v3v - v2v:.6f}",
                "v2_argmax": v2_am if v2_am is not None else "NaN",
                "v3_argmax": v3_am if v3_am is not None else "NaN",
                "v2_match":  str(b_m) if b_m is not None else "NaN",
                "v3_match":  str(a_m) if a_m is not None else "NaN",
                "revision_status": rs,
                "discrimination_status": dstat,
            })

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "concept", "class", "v2_mean", "v3_mean", "delta_mean",
            "v2_argmax", "v3_argmax", "v2_match", "v3_match",
            "revision_status", "discrimination_status",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Delta CSV saved → {path.relative_to(_ROOT)}  ({len(rows)} rows)")
    return status_counts


# ── Output: delta heatmap ─────────────────────────────────────────────────────

def plot_delta_heatmap(concepts: List[str], v2_stats: Dict, v3_stats: Dict,
                       path: pathlib.Path) -> None:
    shared = [c for c in concepts if c in v2_stats]
    delta = np.array([
        [v3_stats[c][cls]["mean"] - v2_stats[c][cls]["mean"] for cls in CLASSES]
        for c in shared
    ])
    vmax = max(abs(delta.min()), abs(delta.max())) + 1e-3

    fig, (ax, ax_txt) = plt.subplots(
        1, 2, figsize=(11, max(8, len(shared) * 0.5)),
        gridspec_kw={"width_ratios": [2, 1.4]},
    )
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="v3 − v2 mean score")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_yticks(range(len(shared)))
    ax.set_yticklabels([c + (" [R]" if c in REVISED_CONCEPTS else "") for c in shared],
                       fontsize=7)
    ax.set_title("v3 − v2 mean cosine similarity\n[R] = revised prompt", fontsize=9, pad=8)
    for r in range(len(shared)):
        for c in range(len(CLASSES)):
            v = delta[r, c]
            ax.text(c, r, f"{v:+.3f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(v) > vmax * 0.6 else "black")

    ax_txt.axis("off")
    ax_txt.set_title("discrimination_status", fontsize=9, pad=8)
    for r, c in enumerate(shared):
        b_m = is_match(c, argmax_class(v2_stats[c]))
        a_m = is_match(c, argmax_class(v3_stats[c]))
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
    concepts: List[str], v2_stats: Dict, v3_stats: Dict,
    p_v2: dict, p_v3: dict,
    v2_corr: np.ndarray, v2_names: List[str],
    v3_corr: np.ndarray, v3_names: List[str],
    status_counts: Dict[str, int],
) -> Tuple[str, int, int]:
    L: List[str] = []
    v2_prompt = dict(zip(p_v2["concept_ids"], p_v2["prompts"]))
    v3_prompt = dict(zip(p_v3["concept_ids"], p_v3["prompts"]))

    v2_match = sum(is_match(c, argmax_class(v2_stats[c])) for c in concepts if c in v2_stats)
    v3_match = sum(is_match(c, argmax_class(v3_stats[c])) for c in concepts if c in v3_stats)

    L.append("=" * 70)
    L.append("V2 vs V3 COMPARISON SUMMARY")
    L.append("=" * 70)

    # Section 1 — headline counts
    fixed   = status_counts.get("REVISED-FIXED", 0)
    persist = status_counts.get("REVISED-PERSISTENT-FAIL", 0)
    regress = status_counts.get("REVISED-REGRESSED", 0)
    still   = status_counts.get("REVISED-STILL-MATCH", 0)
    unch_ok = status_counts.get("UNCHANGED-STILL-MATCH", 0)
    unch_mm = status_counts.get("UNCHANGED-STILL-MISMATCH", 0)
    unch_dr = status_counts.get("UNCHANGED-DRIFT", 0)

    L.append("\nSECTION 1 — HEADLINE COUNTS")
    L.append(f"  v2: {v2_match}/{len(concepts)} MATCH   v3: {v3_match}/{len(concepts)} MATCH")
    L.append(f"  Of {len(REVISED_CONCEPTS)} REVISED concepts:")
    L.append(f"    REVISED-FIXED           : {fixed}")
    L.append(f"    REVISED-PERSISTENT-FAIL : {persist}")
    L.append(f"    REVISED-REGRESSED       : {regress}")
    if still:
        L.append(f"    REVISED-STILL-MATCH     : {still}")
    L.append(f"  Of {len(FROZEN_CONCEPTS)} UNCHANGED (frozen) concepts:")
    L.append(f"    UNCHANGED-STILL-MATCH   : {unch_ok}  (expected {len(FROZEN_CONCEPTS)})")
    if unch_mm:
        L.append(f"    UNCHANGED-STILL-MISMATCH: {unch_mm}")
    if unch_dr:
        L.append(f"    UNCHANGED-DRIFT         : {unch_dr}  ← BUG: bit-identity violated")

    # Viral-tier status
    viral_match_v3 = sum(is_match(c, argmax_class(v3_stats[c]))
                         for c in VIRAL_TIER if c in v3_stats)
    L.append(f"  Viral tier (T2B) v3 discrimination: {viral_match_v3}/4 MATCH")

    # Section 2 — per-revised-concept table
    L.append("\nSECTION 2 — PER-REVISED-CONCEPT COMPARISON")
    for cname in sorted(REVISED_CONCEPTS):
        v2_am = argmax_class(v2_stats[cname])
        v3_am = argmax_class(v3_stats[cname])
        dstat = discrimination_status(cname, is_match(cname, v2_am), is_match(cname, v3_am))
        verdict = {"REVISED-FIXED": "FIXED",
                   "REVISED-PERSISTENT-FAIL": "PERSISTENT-FAIL",
                   "REVISED-REGRESSED": "REGRESSED",
                   "REVISED-STILL-MATCH": "STILL-MATCH"}.get(dstat, dstat)
        L.append(f"\n  [{cname}]  [{TIER_LABELS.get(cname,'?')}]  → {verdict}")
        L.append(f"    v2 prompt: {v2_prompt.get(cname,'n/a')}")
        L.append(f"    v3 prompt: {v3_prompt.get(cname,'n/a')}")
        v2_row = "  ".join(f"{c}={v2_stats[cname][c]['mean']:.4f}" for c in CLASSES)
        v3_row = "  ".join(f"{c}={v3_stats[cname][c]['mean']:.4f}" for c in CLASSES)
        L.append(f"    v2 means: {v2_row}  argmax={v2_am}  spread={spread(v2_stats[cname]):.4f}")
        L.append(f"    v3 means: {v3_row}  argmax={v3_am}  spread={spread(v3_stats[cname]):.4f}")
        L.append(f"    spread change: {spread(v3_stats[cname]) - spread(v2_stats[cname]):+.4f}")

    # Section 3 — dropped concept(s)
    L.append("\nSECTION 3 — DROPPED CONCEPTS")
    if DROPPED_CONCEPTS:
        for cname in sorted(DROPPED_CONCEPTS):
            L.append(f"  {cname}: dropped between v2 and v3")
    else:
        L.append("  None. v3 drops no concepts (no_focal_opacity was already dropped in v2).")

    # Section 4 — correlation structure comparison
    L.append("\nSECTION 4 — CORRELATION STRUCTURE COMPARISON")
    h2 = eval_hypotheses(v2_corr, v2_names)
    h3 = eval_hypotheses(v3_corr, v3_names)
    L.append(f"  Mean |r| (all pairs):       v2={h2['mean_abs_r']:.4f}  "
             f"v3={h3['mean_abs_r']:.4f}  Δ={h3['mean_abs_r']-h2['mean_abs_r']:+.4f}")
    L.append("  Within-block mean |r|:")
    L.append(f"    consolidation_block:  v2={h2['within_consolidation']:.4f}  "
             f"v3={h3['within_consolidation']:.4f}")
    L.append(f"    interstitial_block:   v2={h2['within_interstitial']:.4f}  "
             f"v3={h3['within_interstitial']:.4f}")
    L.append(f"    normal_anchor_block:  v2={h2['within_normal_anchor']:.4f}  "
             f"v3={h3['within_normal_anchor']:.4f}  (4 members both)")
    L.append("  Between-block mean |r| (consolidation × interstitial):")
    L.append(f"    v2={h2['between_con_int_abs']:.4f}  v3={h3['between_con_int_abs']:.4f}")
    L.append("  Anchor×pneumonia signed r:")
    L.append(f"    anchor×consolidation: v2={h2['anchor_x_con_signed']:+.4f}  "
             f"v3={h3['anchor_x_con_signed']:+.4f}")
    L.append(f"    anchor×interstitial:  v2={h2['anchor_x_int_signed']:+.4f}  "
             f"v3={h3['anchor_x_int_signed']:+.4f}")
    L.append("  Expected: within-block correlations persist (no structure engineering);")
    L.append("    only 2 viral-tier prompts changed, so blocks should be largely stable.")

    # Section 5 — hypothesis follow-up
    L.append("\nSECTION 5 — PRE-REGISTERED HYPOTHESIS FOLLOW-UP")
    L.append("  (H1-H5 pre-registered on v1; v3 reported as follow-up, not fresh pre-registration)")
    pf = lambda b: "PASS" if b else "FAIL"
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        L.append(f"  {h}: {HYPOTHESES[h]}")
        L.append(f"      v2: {pf(h2[h])}   v3: {pf(h3[h])}")

    # Section 6 — recommendation
    L.append("\nSECTION 6 — RECOMMENDATION")
    if fixed == 2:
        rec = ("v3 fixes the persistent viral-tier failures; recommend adopting v3 as "
               "operative feature set. The viral tier now has 4/4 discrimination MATCH.")
    elif fixed == 1:
        rec = ("v3 partially fixes the viral tier (1/2). Per the deferred decision rule, "
               "escalate to Ramona for the drop/keep call on the remaining failure.")
    else:
        rec = ("v3 does not improve on v2 for the viral-tier failures. Per the deferred "
               "decision rule, escalate to Ramona for the drop/keep call on both failures.")
    for line in textwrap.wrap(rec, width=68):
        L.append(f"  {line}")

    return "\n".join(L), v2_match, v3_match


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading v2 CSV from {V2_CSV.relative_to(_ROOT)} ...")
    v2_stats = load_per_class_csv(V2_CSV)
    print(f"Loading v3 CSV from {V3_CSV.relative_to(_ROOT)} ...")
    v3_stats = load_per_class_csv(V3_CSV)

    p_v2 = load_prompts(PROMPTS_V2)
    p_v3 = load_prompts(PROMPTS_V3)
    concepts = p_v3["concept_ids"]

    print(f"Change sets (v2→v3): {len(REVISED_CONCEPTS)} revised, "
          f"{len(FROZEN_CONCEPTS)} frozen, {len(DROPPED_CONCEPTS)} dropped")
    assert REVISED_CONCEPTS == {"bilateral_interstitial_pattern", "hyperinflation"}, \
        f"Expected 2 revised concepts, got {sorted(REVISED_CONCEPTS)}"

    for pth in (V2_CORR_NPY, V3_CORR_NPY):
        if not pth.exists():
            sys.exit(f"ERROR: missing correlation matrix {pth}.\n"
                     "Run correlation_diagnostic_v2.py and correlation_diagnostic_v3.py first.")
    v2_corr, v2_names = load_corr(V2_CORR_NPY, V2_SCORES)
    v3_corr, v3_names = load_corr(V3_CORR_NPY, V3_SCORES)

    status_counts = write_delta_csv(concepts, v2_stats, v3_stats,
                                    OUT_DIR / "delta_per_class_means.csv")
    plot_delta_heatmap(concepts, v2_stats, v3_stats, OUT_DIR / "delta_heatmap.png")
    summary, v2_match, v3_match = build_comparison_summary(
        concepts, v2_stats, v3_stats, p_v2, p_v3,
        v2_corr, v2_names, v3_corr, v3_names, status_counts,
    )
    (OUT_DIR / "comparison_summary.txt").write_text(summary, encoding="utf-8")
    print(f"  Comparison summary → {(OUT_DIR / 'comparison_summary.txt').relative_to(_ROOT)}")

    # ── Final console summary ──────────────────────────────────────────────────
    h2 = eval_hypotheses(v2_corr, v2_names)
    h3 = eval_hypotheses(v3_corr, v3_names)
    drift = status_counts.get("UNCHANGED-DRIFT", 0)
    bit_id = "PASS" if drift == 0 else f"FAIL ({drift} drifted)"
    fixed = status_counts.get("REVISED-FIXED", 0)

    print("\n" + "=" * 70)
    print("FINAL CONSOLE SUMMARY")
    print("=" * 70)
    print(f"Bit-identity (16 frozen concepts): {bit_id}")
    print(f"MATCH count: v2={v2_match}/{len(concepts)}  →  v3={v3_match}/{len(concepts)}")
    print(f"mean |r|:    v2={h2['mean_abs_r']:.4f}  →  v3={h3['mean_abs_r']:.4f}")
    print("\nStatus of the 2 revised concepts:")
    for cname in sorted(REVISED_CONCEPTS):
        v2_am = argmax_class(v2_stats[cname])
        v3_am = argmax_class(v3_stats[cname])
        dstat = discrimination_status(cname, is_match(cname, v2_am), is_match(cname, v3_am))
        print(f"  {cname:<34s}  v2={v2_am:<9s} v3={v3_am:<9s}  {dstat}")

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
