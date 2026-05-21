"""Encodes a list of text prompts with a frozen CLIP text encoder."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def encode_prompts(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Tokenize and encode prompts; return L2-normalised embeddings on CPU.

    Args:
        model:     Frozen CLIP model (from load_clip).
        tokenizer: open_clip tokenizer (from load_clip).
        prompts:   List of N prompt strings.
        device:    Device the model lives on.

    Returns:
        Float tensor of shape (N, 512) on CPU.
    """
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
    embeddings = F.normalize(embeddings, dim=1)
    return embeddings.cpu().float()
