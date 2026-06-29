"""Smoke test: verifies BiomedCLIP loads and produces sensible embeddings."""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image

from src.features.clip_loader import load_biomedclip

HAM_DIR   = pathlib.Path("data/ham10000")
TEST_PROMPT = "a dermoscopic image of a skin lesion"


def main() -> None:
    model, preprocess, tokenizer, device = load_biomedclip()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device   : {device}")
    print(f"Model    : BiomedCLIP ViT-B/16 (PubMedBERT tokeniser)")
    print(f"Params   : {n_params:,}")

    jpg_files = sorted(HAM_DIR.glob("**/*.jpg"))
    if not jpg_files:
        print(f"\nNo .jpg images found in {HAM_DIR}. Place HAM10000 images there to test encoding.")
        return

    img_path = jpg_files[0]
    print(f"\nImage    : {img_path}")

    image  = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    tokens = tokenizer([TEST_PROMPT]).to(device)

    with torch.no_grad():
        img_emb = model.encode_image(image)
        txt_emb = model.encode_text(tokens)

    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)

    similarity = (img_emb @ txt_emb.T).item()

    assert img_emb.shape == (1, 512), f"Unexpected image embedding shape: {img_emb.shape}"
    assert txt_emb.shape == (1, 512), f"Unexpected text embedding shape: {txt_emb.shape}"

    print(f"Image embedding shape : {tuple(img_emb.shape)}  OK")
    print(f"Text  embedding shape : {tuple(txt_emb.shape)}  OK")
    print(f"Cosine similarity     : {similarity:.4f}")

    assert -1.0 <= similarity <= 1.0, f"Similarity out of range: {similarity}"
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
