"""Parses the INI-style concept prompt file into structured data."""

from __future__ import annotations

import configparser
import pathlib
from typing import Dict, List


def load_prompts(path: str | pathlib.Path) -> Dict:
    """Parse a concept prompt file and return ordered prompt metadata.

    The file uses configparser-compatible INI blocks:
        [concept_id]
        tier: <int>
        t1: <prompt string>
        t2: <prompt string>
        t3: <prompt string>

    Lines starting with '#' are treated as comments. Ordering is
    deterministic: concept 0 templates 0/1/2, then concept 1, etc.

    Returns a dict with keys:
        concept_ids        – list[str], length 24
        tiers              – list[int], length 24
        prompts            – list[str], length 72
        prompt_concept_idx – list[int], length 72, values in 0..23
        prompt_template_idx – list[int], length 72, values in {0, 1, 2}
    """
    path = pathlib.Path(path)
    raw = path.read_text(encoding="utf-8")

    # Strip comment lines so configparser doesn't choke on '#' inline
    lines = [l for l in raw.splitlines() if not l.lstrip().startswith("#")]
    cleaned = "\n".join(lines)

    parser = configparser.RawConfigParser()
    parser.read_string(cleaned)

    concept_ids: List[str] = []
    tiers: List[int] = []
    prompts: List[str] = []
    prompt_concept_idx: List[int] = []
    prompt_template_idx: List[int] = []

    for concept_idx, section in enumerate(parser.sections()):
        concept_ids.append(section)
        tiers.append(int(parser[section]["tier"]))
        for tmpl_idx, key in enumerate(("t1", "t2", "t3")):
            prompts.append(parser[section][key])
            prompt_concept_idx.append(concept_idx)
            prompt_template_idx.append(tmpl_idx)

    return {
        "concept_ids": concept_ids,
        "tiers": tiers,
        "prompts": prompts,
        "prompt_concept_idx": prompt_concept_idx,
        "prompt_template_idx": prompt_template_idx,
    }
