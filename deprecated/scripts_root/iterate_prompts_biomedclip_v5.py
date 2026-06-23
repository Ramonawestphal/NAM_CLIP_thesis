"""Re-encode 39 changed BiomedCLIP prompts for v5 and recompute the score matrix.

v4 → v5 change: 13 mel-targeted + nv-anchor concepts get PubMed-caption-style
phrasing; the other 11 concepts are unchanged and their cached embeddings are
reused without re-encoding.

Image embeddings are never re-processed; only text changes.

Outputs:
    data/features/biomedclip/ham10000_text_embeddings_v5.npy
    data/features/biomedclip/ham10000_concept_scores_v5.npz

Run from project root:
    python scripts/iterate_prompts_biomedclip_v5.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.features.clip_loader import load_biomedclip
from src.features.encode_text import encode_prompts
from src.features.prompt_loader import load_prompts

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROMPTS_V4       = _ROOT / "src/features/prompts/ham10000_prompts_v4.txt"
PROMPTS_V5       = _ROOT / "src/features/prompts/ham10000_prompts_v5_biomedclip.txt"
TEXT_EMB_V4      = _ROOT / "data/features/biomedclip/ham10000_text_embeddings_v4.npy"
IMAGE_EMBEDDINGS = _ROOT / "data/features/biomedclip/ham10000_image_embeddings.npy"
SCORES_V4_NPZ    = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v4.npz"
TEXT_EMB_V5      = _ROOT / "data/features/biomedclip/ham10000_text_embeddings_v5.npy"
SCORES_V5_NPZ    = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v5.npz"
# ---------------------------------------------------------------------------


def _v4_row_lookup(p: dict) -> dict[tuple[str, int], int]:
    """Map (concept_id, template_idx) → row index in the v4 embedding matrix."""
    lookup: dict[tuple[str, int], int] = {}
    for i in range(len(p["prompts"])):
        cid  = p["concept_ids"][p["prompt_concept_idx"][i]]
        tmpl = p["prompt_template_idx"][i]
        lookup[(cid, tmpl)] = i
    return lookup


def find_changed_indices(p4: dict, p5: dict) -> list[int]:
    """Return row indices in p5 whose prompt text differs from the matching v4 row.

    Matching is by (concept_id, template_idx).
    """
    v4_text: dict[tuple[str, int], str] = {}
    for i in range(len(p4["prompts"])):
        cid  = p4["concept_ids"][p4["prompt_concept_idx"][i]]
        tmpl = p4["prompt_template_idx"][i]
        v4_text[(cid, tmpl)] = p4["prompts"][i]

    changed = []
    for i in range(len(p5["prompts"])):
        cid  = p5["concept_ids"][p5["prompt_concept_idx"][i]]
        tmpl = p5["prompt_template_idx"][i]
        if v4_text.get((cid, tmpl)) != p5["prompts"][i]:
            changed.append(i)
    return changed


def main() -> None:
    for path in (TEXT_EMB_V4, IMAGE_EMBEDDINGS, SCORES_V4_NPZ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}\n"
                "Run scripts/extract_features_biomedclip.py first."
            )

    # ------------------------------------------------------------------
    # 1. Load prompt files; identify changed rows
    # ------------------------------------------------------------------
    p4 = load_prompts(PROMPTS_V4)
    p5 = load_prompts(PROMPTS_V5)

    changed_idx = find_changed_indices(p4, p5)
    changed_set = set(changed_idx)

    # Identify which concepts changed (for verification printout)
    changed_concepts: dict[str, list[str]] = {}
    tmpl_name = {0: "t1", 1: "t2", 2: "t3"}
    for i in changed_idx:
        cid  = p5["concept_ids"][p5["prompt_concept_idx"][i]]
        tmpl = tmpl_name[p5["prompt_template_idx"][i]]
        changed_concepts.setdefault(cid, []).append(tmpl)

    print(f"Changed prompts : {len(changed_idx)}  (expected 39)")
    print(f"Changed concepts: {len(changed_concepts)}")
    for cid, tmpls in changed_concepts.items():
        print(f"  {cid:<30}  templates: {', '.join(tmpls)}")

    # ------------------------------------------------------------------
    # 2. Load BiomedCLIP
    # ------------------------------------------------------------------
    model, _, tokenizer, device = load_biomedclip()
    print(f"\nBiomedCLIP loaded on {device}")

    # ------------------------------------------------------------------
    # 3. Load v4 text embeddings; build v5 matrix
    # ------------------------------------------------------------------
    text_emb_v4 = np.load(TEXT_EMB_V4)           # (72, 512)
    v4_rows     = _v4_row_lookup(p4)
    text_emb_v5 = np.empty_like(text_emb_v4)

    # Copy unchanged rows from v4 (matched by concept_id + template_idx)
    for i in range(len(p5["prompts"])):
        if i in changed_set:
            continue
        cid  = p5["concept_ids"][p5["prompt_concept_idx"][i]]
        tmpl = p5["prompt_template_idx"][i]
        text_emb_v5[i] = text_emb_v4[v4_rows[(cid, tmpl)]]

    # Encode only the changed prompts
    changed_texts = [p5["prompts"][i] for i in changed_idx]
    print(f"Encoding {len(changed_idx)} changed prompts…")
    new_embs = encode_prompts(model, tokenizer, changed_texts, device).numpy()
    for out_row, emb in zip(changed_idx, new_embs):
        text_emb_v5[out_row] = emb

    np.save(TEXT_EMB_V5, text_emb_v5)
    print(f"Saved v5 text embeddings → {TEXT_EMB_V5}")

    # ------------------------------------------------------------------
    # 4. Recompute similarity matrix
    # ------------------------------------------------------------------
    image_emb = np.load(IMAGE_EMBEDDINGS)                         # (N, 512)
    scores_v5 = (image_emb @ text_emb_v5.T).astype(np.float32)   # (N, 72)

    assert scores_v5.min() >= -1.0 and scores_v5.max() <= 1.0, (
        f"Scores out of [-1, 1]: min={scores_v5.min():.4f}, max={scores_v5.max():.4f}"
    )

    # ------------------------------------------------------------------
    # 5. Save NPZ (same key structure as v4)
    # ------------------------------------------------------------------
    v4_data = np.load(SCORES_V4_NPZ, allow_pickle=True)
    np.savez_compressed(
        SCORES_V5_NPZ,
        scores              = scores_v5,
        image_ids           = v4_data["image_ids"],
        labels              = v4_data["labels"],
        lesion_ids          = v4_data["lesion_ids"],
        concept_ids         = np.array(p5["concept_ids"]),
        prompts             = np.array(p5["prompts"]),
        prompt_concept_idx  = np.array(p5["prompt_concept_idx"],  dtype=np.int32),
        prompt_template_idx = np.array(p5["prompt_template_idx"], dtype=np.int32),
        tiers               = np.array(p5["tiers"],               dtype=np.int32),
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\nPrompts re-encoded : {len(changed_idx)}")
    print(f"Score min          : {scores_v5.min():.4f}")
    print(f"Score max          : {scores_v5.max():.4f}")
    print(f"Score mean         : {scores_v5.mean():.4f}")
    print(f"Saved scores       → {SCORES_V5_NPZ}")


if __name__ == "__main__":
    main()
