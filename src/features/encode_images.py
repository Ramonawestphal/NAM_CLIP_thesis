"""Encodes a list of image paths with a frozen CLIP image encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def encode_images(
    model: torch.nn.Module,
    preprocess: Callable,
    image_paths: List[Path],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, List[str]]:
    """Encode images in batches; return L2-normalised embeddings on CPU.

    Failed images are logged and skipped; the returned tensor and ID list
    stay aligned with each other.

    Args:
        model:       Frozen CLIP model (from load_clip).
        preprocess:  torchvision transform (from load_clip).
        image_paths: List of Path objects pointing to image files.
        device:      Device the model lives on.
        batch_size:  Number of images per forward pass.

    Returns:
        embeddings: Float tensor of shape (N, 512) on CPU, where N ≤ len(image_paths).
        image_ids:  List of N image ID strings (filename stems), same row order.
    """
    all_embeddings: List[torch.Tensor] = []
    all_ids: List[str] = []

    batches = [image_paths[i : i + batch_size] for i in range(0, len(image_paths), batch_size)]

    for batch_paths in tqdm(batches, desc="Encoding images", unit="batch"):
        tensors: List[torch.Tensor] = []
        ids: List[str] = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
                ids.append(p.stem)
            except Exception as exc:
                print(f"[encode_images] skipping {p.name}: {exc}")

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            emb = model.encode_image(batch_tensor)
        emb = F.normalize(emb, dim=1)
        all_embeddings.append(emb.cpu().float())
        all_ids.extend(ids)

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings, all_ids
