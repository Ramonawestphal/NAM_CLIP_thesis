from __future__ import annotations

from typing import Any


def clip_available() -> bool:
    try:
        import open_clip  # noqa: F401

        return True
    except ImportError:
        return False


def extract_image_features(_images: Any, _model_name: str = "ViT-B-32") -> Any:
    """Placeholder: wire open_clip here when vision datasets are added."""
    raise NotImplementedError("CLIP extraction not wired yet; install open-clip-torch when ready.")
