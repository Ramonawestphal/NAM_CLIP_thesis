"""Loads a frozen OpenAI CLIP ViT-B/32 model via open_clip."""

from __future__ import annotations

from typing import Tuple

import torch
import open_clip


def load_clip(device: str | None = None) -> Tuple[torch.nn.Module, object, object, torch.device]:
    """Load frozen CLIP ViT-B/32 with OpenAI weights.

    Args:
        device: 'cuda', 'cpu', or None for auto-detection.

    Returns:
        (model, preprocess, tokenizer, device)
        - model: open_clip model in eval mode with frozen parameters
        - preprocess: torchvision transform for image inputs
        - tokenizer: open_clip tokenizer for text inputs
        - device: resolved torch.device
    """
    if device is None:
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(resolved)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return model, preprocess, tokenizer, resolved
