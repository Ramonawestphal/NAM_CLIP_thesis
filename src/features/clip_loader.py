"""Loads frozen CLIP models via open_clip.

Available loaders
-----------------
load_clip()        – OpenAI CLIP ViT-B/32 (standard)
load_biomedclip()  – Microsoft BiomedCLIP ViT-B/16, PubMedBERT tokeniser
"""

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


_BIOMEDCLIP_HF_ID = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)


def load_biomedclip(
    device: str | None = None,
) -> Tuple[torch.nn.Module, object, object, torch.device]:
    """Load frozen BiomedCLIP ViT-B/16 with PubMedBERT tokeniser via open_clip.

    Weights are downloaded from HuggingFace Hub on first call and cached
    by the ``huggingface_hub`` library (already a transitive dependency of
    open_clip_torch).

    Args:
        device: 'cuda', 'cpu', or None for auto-detection.

    Returns:
        (model, preprocess, tokenizer, device)
        - model: open_clip model in eval mode with frozen parameters
        - preprocess: torchvision transform for 224×224 inputs
        - tokenizer: HFTokenizer for PubMedBERT (context length 256)
        - device: resolved torch.device
    """
    if device is None:
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)

    model, _, preprocess = open_clip.create_model_and_transforms(_BIOMEDCLIP_HF_ID)
    model = model.to(resolved)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    tokenizer = open_clip.get_tokenizer(_BIOMEDCLIP_HF_ID)

    return model, preprocess, tokenizer, resolved
