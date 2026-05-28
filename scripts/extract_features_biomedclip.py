"""Extract BiomedCLIP concept similarity scores for all HAM10000 images.

Uses the v4 prompt set and writes outputs to data/features/biomedclip/ so
existing ViT-B/32 files are never overwritten.

Outputs:
    data/features/biomedclip/ham10000_concept_scores_v4.npz
    data/features/biomedclip/ham10000_image_embeddings.npy
    data/features/biomedclip/ham10000_text_embeddings_v4.npy

Run from project root:
    python scripts/extract_features_biomedclip.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.features.clip_loader import load_biomedclip
from src.features.encode_images import encode_images
from src.features.encode_text import encode_prompts
from src.features.prompt_loader import load_prompts

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA_CSV    = _ROOT / "data/ham10000/HAM10000_metadata.csv"
IMAGE_DIRS      = [
    _ROOT / "data/ham10000/HAM10000_images_part_1",
    _ROOT / "data/ham10000/HAM10000_images_part_2",
]
PROMPTS_FILE    = _ROOT / "src/features/prompts/ham10000_prompts_v4.txt"
OUTPUT_DIR      = _ROOT / "data/features/biomedclip"
SCORES_PATH     = OUTPUT_DIR / "ham10000_concept_scores_v4.npz"
EMBEDDINGS_PATH = OUTPUT_DIR / "ham10000_image_embeddings.npy"
TEXT_EMB_PATH   = OUTPUT_DIR / "ham10000_text_embeddings_v4.npy"
# ---------------------------------------------------------------------------


def resolve_image_paths(
    image_ids: list[str],
) -> tuple[list[pathlib.Path], list[str]]:
    """Return (paths, ids) for every image_id found in IMAGE_DIRS; warn on missing."""
    lookup: dict[str, pathlib.Path] = {}
    for d in IMAGE_DIRS:
        for p in d.glob("*.jpg"):
            lookup[p.stem] = p

    paths, found_ids = [], []
    for iid in image_ids:
        if iid in lookup:
            paths.append(lookup[iid])
            found_ids.append(iid)
        else:
            print(f"[extract_features_biomedclip] image not found on disk: {iid}")
    return paths, found_ids


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Metadata
    meta    = pd.read_csv(METADATA_CSV)
    meta_ix = meta.set_index("image_id")
    all_ids = meta["image_id"].tolist()

    # 2. Resolve image paths
    image_paths, found_ids = resolve_image_paths(all_ids)
    print(f"Images in metadata  : {len(all_ids)}")
    print(f"Images found on disk: {len(found_ids)}")

    # 3. Load BiomedCLIP
    model, preprocess, tokenizer, device = load_biomedclip()
    print(f"BiomedCLIP device: {device}")

    # 4. Prompts
    prompt_data = load_prompts(PROMPTS_FILE)
    prompts     = prompt_data["prompts"]
    print(f"Prompt file   : {PROMPTS_FILE.relative_to(_ROOT)}")
    print(f"Prompts loaded: {len(prompts)} ({len(prompt_data['concept_ids'])} concepts × 3 templates)")

    # 5. Encode text  →  (72, 512)
    print("Encoding text prompts…")
    text_emb = encode_prompts(model, tokenizer, prompts, device)
    assert text_emb.shape == (72, 512), f"Unexpected text embedding shape: {text_emb.shape}"
    np.save(TEXT_EMB_PATH, text_emb.numpy().astype(np.float32))
    print(f"Saved text embeddings → {TEXT_EMB_PATH}")

    # 6. Encode images  →  (N, 512)
    image_emb, encoded_ids = encode_images(
        model, preprocess, image_paths, device, batch_size=32
    )
    N = image_emb.shape[0]
    print(f"Images encoded: {N}")
    assert image_emb.shape[1] == 512

    np.save(EMBEDDINGS_PATH, image_emb.numpy().astype(np.float32))
    print(f"Saved image embeddings → {EMBEDDINGS_PATH}")

    # 7. Similarity matrix  →  (N, 72)
    scores = (image_emb @ text_emb.T).numpy().astype(np.float32)
    assert scores.shape == (N, 72), f"Unexpected scores shape: {scores.shape}"
    assert scores.min() >= -1.0 and scores.max() <= 1.0, (
        f"Scores out of [-1, 1]: min={scores.min():.4f}, max={scores.max():.4f}"
    )

    # 8. Align metadata to encoded row order
    labels    = [meta_ix.loc[iid, "dx"]        for iid in encoded_ids]
    lesion_ids = [meta_ix.loc[iid, "lesion_id"] for iid in encoded_ids]

    # 9. Save NPZ  (identical key structure to ViT-B/32 version)
    np.savez_compressed(
        SCORES_PATH,
        scores              = scores,
        image_ids           = np.array(encoded_ids),
        labels              = np.array(labels),
        lesion_ids          = np.array(lesion_ids),
        concept_ids         = np.array(prompt_data["concept_ids"]),
        prompts             = np.array(prompt_data["prompts"]),
        prompt_concept_idx  = np.array(prompt_data["prompt_concept_idx"],  dtype=np.int32),
        prompt_template_idx = np.array(prompt_data["prompt_template_idx"], dtype=np.int32),
        tiers               = np.array(prompt_data["tiers"],               dtype=np.int32),
    )
    print(f"Saved scores → {SCORES_PATH}")

    # 10. Summary (for comparison with ViT-B/32 run)
    label_counts = pd.Series(labels).value_counts().sort_index()
    print("\n--- Summary ---")
    print(f"Scores shape : {scores.shape}")
    print(f"Score min    : {scores.min():.4f}")
    print(f"Score max    : {scores.max():.4f}")
    print(f"Score mean   : {scores.mean():.4f}")
    print("\nClass distribution:")
    for dx, count in label_counts.items():
        print(f"  {dx:<6}  {count:>5}")


if __name__ == "__main__":
    main()
