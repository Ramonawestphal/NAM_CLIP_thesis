"""
v3 Step 4: auto-generate src/features/prompts/chestxray_prompts_v3.txt.

Concatenates:
  1. The 16 frozen v2 concepts (those that MATCH in v2) — copied verbatim
     from chestxray_prompts_v2.txt including their header annotations.
  2. The 2 winning v3 prompts (bilateral_interstitial_pattern, hyperinflation),
     formatted as new blocks with an annotation recording the v2 prompt they
     replace, the smoke-test discrimination_score, and the winning candidate.

The winners are placed in their original v2 positions, so the final file keeps
the v2 concept ordering and has exactly 18 concepts.

Run from project root (after v3_select_winners.py):
    python scripts/chestxray/v3_build_prompt_file.py
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Dict, List, Tuple

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # robust on cp1252 consoles
except Exception:
    pass

import numpy as np

from src.features.prompt_loader import load_prompts

PROMPTS_V2     = _ROOT / "src/features/prompts/chestxray_prompts_v2.txt"
PROMPTS_V3     = _ROOT / "src/features/prompts/chestxray_prompts_v3.txt"
WINNING        = _ROOT / "results/chestxray/v3_smoketest/winning_prompts.txt"
SMOKETEST_NPZ  = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v3_smoketest.npz"
SUBSAMPLE      = _ROOT / "results/chestxray/v3_smoketest/subsample_indices.npz"

REVISED_CONCEPTS = ["bilateral_interstitial_pattern", "hyperinflation"]
CLASSES = ["normal", "bacteria", "virus"]
TARGET_CLASS = "virus"


def parse_v2_blocks(raw: str) -> Tuple[List[str], Dict[str, str]]:
    """Return (ordered concept names, {concept: verbatim block text}).

    A block starts at a '[name]' line and runs to the line before the next
    '[' header. Trailing blank lines and section-divider comment lines
    (e.g. '# =====') are stripped from the end of each block.
    """
    lines = raw.splitlines()
    header_idx = [i for i, l in enumerate(lines) if re.match(r"^\[(\w+)\]", l)]
    names, blocks = [], {}
    for k, start in enumerate(header_idx):
        end = header_idx[k + 1] if k + 1 < len(header_idx) else len(lines)
        block_lines = lines[start:end]
        # strip trailing blanks and decorative divider/comment lines
        while block_lines and (
            block_lines[-1].strip() == ""
            or re.match(r"^#\s*=+", block_lines[-1])
            or re.match(r"^#\s*TIER", block_lines[-1])
        ):
            block_lines.pop()
        name = re.match(r"^\[(\w+)\]", block_lines[0]).group(1)
        names.append(name)
        blocks[name] = "\n".join(block_lines)
    return names, blocks


def winner_metadata() -> Dict[str, dict]:
    """For each revised concept, return winning prompt, suffix, discrim score."""
    # winning prompts
    win_prompt: Dict[str, str] = {}
    for line in WINNING.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            concept, prompt = line.split("\t", 1)
            win_prompt[concept.strip()] = prompt.strip()

    # smoke-test scores to derive suffix + discrimination_score
    data = np.load(SMOKETEST_NPZ, allow_pickle=True)
    scores          = data["scores"]
    candidate_names = data["candidate_names"].tolist()
    prompts         = data["prompts"].tolist()
    cand_prompt = dict(zip(candidate_names, prompts))
    sub = np.load(SUBSAMPLE, allow_pickle=True)
    subtype = sub["subsample_labels_subtype"]

    def disc(col):
        m = {c: float(col[(subtype == c) & ~np.isnan(col)].mean()) for c in CLASSES}
        return m[TARGET_CLASS] - 0.5 * (m["normal"] + m["bacteria"]), m

    meta: Dict[str, dict] = {}
    for concept in REVISED_CONCEPTS:
        wp = win_prompt[concept]
        # find which candidate name has this prompt text
        suffix, score, means = "?", float("nan"), {}
        for ci, cname in enumerate(candidate_names):
            if cand_prompt[cname] == wp and cname.startswith(concept):
                suffix = cname[len(concept):]  # '_a' or '_b'
                score, means = disc(scores[:, ci])
                break
        meta[concept] = {"prompt": wp, "suffix": suffix, "score": score, "means": means}
    return meta


def make_winner_block(concept: str, v2_block: str, meta: dict) -> str:
    """Build a new [concept] block with annotation + winning prompt."""
    # Extract the v2 prompt (t1 line) being replaced
    m = re.search(r"^t1:\s*(.*)$", v2_block, re.M)
    v2_prompt = m.group(1) if m else "(unknown)"
    means = meta["means"]
    means_str = (f"normal={means.get('normal', float('nan')):.4f}, "
                 f"bacteria={means.get('bacteria', float('nan')):.4f}, "
                 f"virus={means.get('virus', float('nan')):.4f}") if means else "n/a"
    return "\n".join([
        f"[{concept}]",
        f"# REVISED in v3 (smoke-test winner: candidate {meta['suffix']})",
        f"# Replaces v2 prompt: \"{v2_prompt}\"",
        f"# Smoke-test (300-image stratified subsample) discrimination_score "
        f"= {meta['score']:.4f}",
        f"#   (= mean_virus - 0.5*(mean_normal + mean_bacteria); "
        f"subsample per-class means: {means_str})",
        f"# Selected by the v3 smoke test BEFORE full-train-pool extraction, to",
        f"# avoid prompt tuning on the diagnostic.",
        f"tier: 2",
        f"t1: {meta['prompt']}",
    ])


HEADER = """\
# Chest X-ray Concept Prompts - VERSION 3 (smoke-test selected viral-tier revision)
#
# Dataset: Kermany et al. (2018) Chest X-ray Pneumonia; BiomedCLIP (ViT-B/16 + PubMedBERT)
#
# ──────────────────────────────────────────────────────────────────────
# Iteration history
# ──────────────────────────────────────────────────────────────────────
# v1: 19-concept set. Per-class diagnostic 14/19 MATCH; mean |r| = 0.449.
# v2: conservative targeted revision (5 revised, 1 dropped → 18 concepts).
#     Per-class diagnostic 16/18 MATCH. Two viral-tier concepts persistently
#     failed: bilateral_interstitial_pattern (argmax=normal, spread 0.066 in
#     the WRONG direction) and hyperinflation (argmax=normal, spread 0.015).
# v3 (this file): the 16 v2-MATCH concepts are FROZEN (copied verbatim from v2,
#     byte-identical). The 2 failing concepts get the winning phrasing from a
#     smoke test: two candidate phrasings per concept were scored on a 300-image
#     stratified subsample of the train pool by
#         discrimination_score = mean_virus - 0.5*(mean_normal + mean_bacteria),
#     and the higher-scoring candidate was selected BEFORE full extraction. This
#     avoids selecting prompts on their full-train-pool diagnostic performance.
#
# Pre-commit note (Ramona's deferred decision rule): if either of these two v3
# concepts persistently fails per-class discrimination on the full train pool,
# the decision on whether to drop or iterate further is held over to post-v3
# review.
#
# Total: 18 concepts (16 frozen from v2 + 2 smoke-test winners).
# Format: INI-style, one t1 prompt per concept (matches prompt_loader.py).
"""


def main() -> None:
    raw = PROMPTS_V2.read_text(encoding="utf-8")
    names, blocks = parse_v2_blocks(raw)
    assert len(names) == 18, f"Expected 18 v2 concepts, parsed {len(names)}"

    meta = winner_metadata()

    out_blocks: List[str] = [HEADER]
    for name in names:  # preserve v2 ordering
        if name in REVISED_CONCEPTS:
            out_blocks.append(make_winner_block(name, blocks[name], meta[name]))
        else:
            out_blocks.append(blocks[name])

    PROMPTS_V3.write_text("\n\n".join(out_blocks) + "\n", encoding="utf-8")
    print(f"Wrote {PROMPTS_V3.relative_to(_ROOT)}")

    # Verify
    p = load_prompts(PROMPTS_V3)
    assert len(p["concept_ids"]) == 18, \
        f"v3 must have 18 concepts, got {len(p['concept_ids'])}"
    assert set(p["prompt_template_idx"]) == {0}, "v3 should be single-template (t1)"
    pmap = dict(zip(p["concept_ids"], p["prompts"]))
    for concept in REVISED_CONCEPTS:
        assert pmap[concept] == meta[concept]["prompt"], \
            f"{concept} prompt mismatch in generated file"
    print(f"  Verified: 18 concepts, single-template, winners installed ✓")
    print("  Revised in v3:")
    for concept in REVISED_CONCEPTS:
        print(f"    [{concept}] ({meta[concept]['suffix']}, "
              f"disc={meta[concept]['score']:.4f}): {meta[concept]['prompt'][:70]}")


if __name__ == "__main__":
    main()
