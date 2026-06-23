"""
v3 Step 3: select the winning candidate phrasing per failing concept.

For each of bilateral_interstitial_pattern and hyperinflation, compares the two
candidates (_a vs _b) on per-class means over the smoke-test subsample.

Selection criterion (target_class = virus for both):
    discrimination_score = mean_virus - 0.5 * (mean_normal + mean_bacteria)
Higher score wins.
Tie-break (|Δscore| < 0.005): prefer the candidate whose argmax == virus;
    if both share argmax, prefer _a.

Outputs:
    results/chestxray/v3_smoketest/smoketest_per_class_means.csv
    results/chestxray/v3_smoketest/winner_selection.txt
    results/chestxray/v3_smoketest/winning_prompts.txt   (concept<TAB>prompt)

Run from project root (after v3_smoketest_extract.py):
    python scripts/chestxray/v3_select_winners.py
"""

from __future__ import annotations

import csv
import pathlib
import sys
from typing import Dict, List

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # robust on cp1252 consoles
except Exception:
    pass

import numpy as np

SMOKETEST_NPZ = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v3_smoketest.npz"
SUBSAMPLE     = _ROOT / "results/chestxray/v3_smoketest/subsample_indices.npz"
OUT_DIR       = _ROOT / "results/chestxray/v3_smoketest"

CLASSES = ["normal", "bacteria", "virus"]
TARGET_CLASS = "virus"
TIE_EPS = 0.005

# concept → its two candidate names (_a first)
CONCEPTS = {
    "bilateral_interstitial_pattern": [
        "bilateral_interstitial_pattern_a",
        "bilateral_interstitial_pattern_b",
    ],
    "hyperinflation": [
        "hyperinflation_a",
        "hyperinflation_b",
    ],
}


def per_class_means(col: np.ndarray, subtype: np.ndarray) -> Dict[str, float]:
    out = {}
    for cls in CLASSES:
        vals = col[(subtype == cls) & ~np.isnan(col)]
        out[cls] = float(vals.mean()) if len(vals) else float("nan")
    return out


def discrimination_score(m: Dict[str, float]) -> float:
    return m[TARGET_CLASS] - 0.5 * (m["normal"] + m["bacteria"])


def argmax_class(m: Dict[str, float]) -> str:
    return max(CLASSES, key=lambda c: m[c])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(SMOKETEST_NPZ, allow_pickle=True)
    scores          = data["scores"]                  # (300, 4)
    candidate_names = data["candidate_names"].tolist()
    prompts         = data["prompts"].tolist()
    prompt_map      = dict(zip(candidate_names, prompts))

    sub = np.load(SUBSAMPLE, allow_pickle=True)
    subtype = sub["subsample_labels_subtype"]

    cand_idx = {c: candidate_names.index(c) for c in candidate_names}

    # CSV of all candidate per-class means
    csv_rows: List[dict] = []
    means_by_cand: Dict[str, Dict[str, float]] = {}
    for c in candidate_names:
        m = per_class_means(scores[:, cand_idx[c]], subtype)
        means_by_cand[c] = m
        for cls in CLASSES:
            csv_rows.append({"candidate": c, "class": cls, "mean": f"{m[cls]:.6f}"})

    csv_path = OUT_DIR / "smoketest_per_class_means.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate", "class", "mean"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Saved per-class means → {csv_path.relative_to(_ROOT)}")

    sel_lines: List[str] = []
    winning_prompts: List[tuple] = []
    console: List[str] = []

    for concept, cands in CONCEPTS.items():
        a, b = cands
        ma, mb = means_by_cand[a], means_by_cand[b]
        sa, sb = discrimination_score(ma), discrimination_score(mb)
        am_a, am_b = argmax_class(ma), argmax_class(mb)

        # Winner by discrimination score, with tie-break
        if abs(sa - sb) < TIE_EPS:
            a_hit = am_a == TARGET_CLASS
            b_hit = am_b == TARGET_CLASS
            if a_hit and not b_hit:
                winner, rationale = a, f"tie (|Δ|={abs(sa-sb):.4f}<{TIE_EPS}); _a argmax matches {TARGET_CLASS}"
            elif b_hit and not a_hit:
                winner, rationale = b, f"tie (|Δ|={abs(sa-sb):.4f}<{TIE_EPS}); _b argmax matches {TARGET_CLASS}"
            else:
                winner, rationale = a, f"tie (|Δ|={abs(sa-sb):.4f}<{TIE_EPS}); same argmax → prefer _a"
        elif sa > sb:
            winner, rationale = a, f"_a higher discrimination_score ({sa:.4f} > {sb:.4f})"
        else:
            winner, rationale = b, f"_b higher discrimination_score ({sb:.4f} > {sa:.4f})"

        win_suffix = "_a" if winner == a else "_b"
        win_means = means_by_cand[winner]
        win_argmax = argmax_class(win_means)
        margin = abs(sa - sb)

        # winner_selection.txt block
        sel_lines.append(concept)
        for label, c, m, s, am in [("_a", a, ma, sa, am_a), ("_b", b, mb, sb, am_b)]:
            sel_lines.append(f"  Candidate {label}:")
            sel_lines.append(f"    prompt: \"{prompt_map[c]}\"")
            sel_lines.append(f"    per-class means: normal={m['normal']:.4f}, "
                             f"bacteria={m['bacteria']:.4f}, virus={m['virus']:.4f}")
            sel_lines.append(f"    argmax: {am}")
            sel_lines.append(f"    discrimination_score: {s:.4f}")
        sel_lines.append(f"  WINNER: {win_suffix}")
        sel_lines.append(f"  Rationale: {rationale}")
        sel_lines.append("")

        winning_prompts.append((concept, prompt_map[winner]))

        # Console + failure flag
        fail_flag = "" if win_argmax == TARGET_CLASS else \
            f"  ⚠ WINNER STILL FAILS (argmax={win_argmax}, expected {TARGET_CLASS})"
        console.append(
            f"{concept}: _a score={sa:.4f} (argmax {am_a}) | "
            f"_b score={sb:.4f} (argmax {am_b}) → WINNER {win_suffix} "
            f"(margin {margin:.4f}){fail_flag}"
        )

    (OUT_DIR / "winner_selection.txt").write_text("\n".join(sel_lines), encoding="utf-8")
    print(f"Saved winner selection → {(OUT_DIR / 'winner_selection.txt').relative_to(_ROOT)}")

    with (OUT_DIR / "winning_prompts.txt").open("w", encoding="utf-8") as f:
        for concept, prompt in winning_prompts:
            f.write(f"{concept}\t{prompt}\n")
    print(f"Saved winning prompts → {(OUT_DIR / 'winning_prompts.txt').relative_to(_ROOT)}")

    print("\n" + "=" * 70)
    print("WINNER SELECTION SUMMARY")
    print("=" * 70)
    for line in console:
        print(line)


if __name__ == "__main__":
    main()
