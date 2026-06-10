"""
v4 Step 1: build src/features/prompts/chestxray_prompts_v4.txt = v3 minus hyperinflation.

Reads chestxray_prompts_v3.txt, drops the [hyperinflation] block (and any comment
lines that belong to it), keeps all other 17 concept blocks verbatim, and writes a
new v4 header documenting the drop. No other concept is modified.

Run from project root:
    python scripts/chestxray/build_v4_prompt_file.py
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

from src.features.prompt_loader import load_prompts

PROMPTS_V3 = _ROOT / "src/features/prompts/chestxray_prompts_v3.txt"
PROMPTS_V4 = _ROOT / "src/features/prompts/chestxray_prompts_v4.txt"

DROP_CONCEPT = "hyperinflation"


def parse_blocks(raw: str) -> Tuple[List[str], Dict[str, str]]:
    """Return (ordered concept names, {concept: verbatim block text}).

    A block starts at a '[name]' line and runs to the line before the next
    '[' header. Trailing blank lines and section-divider comment lines
    (e.g. '# =====', '# TIER ...') are stripped from the end of each block so
    a dropped block does not leave an orphaned divider attached to its
    neighbour.
    """
    lines = raw.splitlines()
    header_idx = [i for i, l in enumerate(lines) if re.match(r"^\[(\w+)\]", l)]
    names, blocks = [], {}
    for k, start in enumerate(header_idx):
        end = header_idx[k + 1] if k + 1 < len(header_idx) else len(lines)
        block_lines = lines[start:end]
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


HEADER = """\
# Chest X-ray Concept Prompts - VERSION 4 (operative feature set)
#
# Dataset: Kermany et al. (2018) Chest X-ray Pneumonia; BiomedCLIP (ViT-B/16 + PubMedBERT)
#
# ──────────────────────────────────────────────────────────────────────
# Iteration history
# ──────────────────────────────────────────────────────────────────────
# v1: 19-concept set. Per-class diagnostic 14/19 MATCH; mean |r| = 0.449.
# v2: conservative targeted revision (5 revised, 1 dropped → 18 concepts).
#     Per-class diagnostic 16/18 MATCH.
# v3: smoke-test viral-tier revision (2 revised → 18 concepts).
#     Per-class diagnostic 17/18 MATCH. Only hyperinflation still failed.
# v4 (this file): v3 MINUS the hyperinflation concept → 17 concepts.
#     No other concept was modified; all 17 prompts are byte-identical to v3.
#
# ──────────────────────────────────────────────────────────────────────
# Why hyperinflation was dropped (not iterated further)
# ──────────────────────────────────────────────────────────────────────
# hyperinflation failed per-class discrimination across all three revisions,
# always anchoring on the normal class, with spread SHRINKING toward zero as
# prompts became more pathology-specific:
#     v1 "hyperinflation of the lungs"                         spread 0.043 (argmax=normal)
#     v2 "overexpanded lungs with flattened diaphragm domes"   spread 0.015 (argmax=normal)
#     v3 "increased lung volume with depressed diaphragms and
#         widened intercostal spaces"                          spread 0.0008 (argmax=normal)
# The progressive collapse of spread despite increasingly specific phrasing
# indicates BiomedCLIP's representation does not encode the distinction between
# normal inspiratory lung expansion and pathologic hyperinflation. This is a
# property of the encoder, not of any prompt phrasing — so the concept was
# dropped rather than iterated further.
#
# ──────────────────────────────────────────────────────────────────────
# OPERATIVE FEATURE SET
# ──────────────────────────────────────────────────────────────────────
# v4 is the operative chest X-ray concept set for downstream NAM training.
# Feature input: data/features/biomedclip/chestxray_concept_scores_v4.npz
#
# Total: 17 concepts. Format: INI-style, one t1 prompt per concept.
"""


def main() -> None:
    raw = PROMPTS_V3.read_text(encoding="utf-8")
    names, blocks = parse_blocks(raw)
    assert len(names) == 18, f"Expected 18 v3 concepts, parsed {len(names)}"
    assert DROP_CONCEPT in names, f"{DROP_CONCEPT} not found in v3 prompt file"

    kept = [n for n in names if n != DROP_CONCEPT]
    out = "\n\n".join([HEADER] + [blocks[n] for n in kept]) + "\n"
    PROMPTS_V4.write_text(out, encoding="utf-8")
    print(f"Wrote {PROMPTS_V4.relative_to(_ROOT)}")

    # ── Sanity checks ──────────────────────────────────────────────────────────
    p4 = load_prompts(PROMPTS_V4)
    assert len(p4["concept_ids"]) == 17, \
        f"v4 must have 17 concepts, got {len(p4['concept_ids'])}"
    assert DROP_CONCEPT not in p4["concept_ids"], \
        f"{DROP_CONCEPT} should be absent from v4 concepts"
    assert set(p4["concept_ids"]) == set(kept), "v4 concept set != v3 minus hyperinflation"
    assert p4["concept_ids"] == kept, "v4 concept ORDER should match v3 (minus hyperinflation)"
    assert set(p4["prompt_template_idx"]) == {0}, "v4 should be single-template (t1)"

    # 'hyperinflation' may appear only in the documentation header, never in the body
    body = "\n\n".join(blocks[n] for n in kept)
    assert DROP_CONCEPT not in body, \
        f"'{DROP_CONCEPT}' leaked into the v4 body (should only be in the header)"

    # Verify the 17 kept prompts are byte-identical to v3
    p3 = load_prompts(PROMPTS_V3)
    v3_map = dict(zip(p3["concept_ids"], p3["prompts"]))
    v4_map = dict(zip(p4["concept_ids"], p4["prompts"]))
    for c in kept:
        assert v3_map[c] == v4_map[c], f"prompt text drift for {c} between v3 and v4"

    print(f"  Sanity checks PASSED ✓")
    print(f"  v4 concepts ({len(p4['concept_ids'])}): {p4['concept_ids']}")
    print(f"  Dropped: {DROP_CONCEPT}")


if __name__ == "__main__":
    main()
