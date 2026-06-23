"""
Extract BiomedCLIP v6 concept score features for HAM10000.

Uses the v6 single-template prompt file (24 prompts) and cached image
embeddings. Does not re-encode images.

Outputs:
    data/features/biomedclip/ham10000_text_embeddings_v6.npy   (24, 512)
    data/features/biomedclip/ham10000_concept_scores_v6.npz    (10015, 24)

Run from project root:
    python scripts/extract_features_biomedclip_v6.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from src.features.clip_loader import load_biomedclip
from src.features.encode_text import encode_prompts
from src.features.prompt_loader import load_prompts

# ── Paths ─────────────────────────────────────────────────────────────────────
PROMPTS_V6       = _ROOT / "src/features/prompts/ham10000_prompts_v6_biomedclip.txt"
IMAGE_EMBEDDINGS = _ROOT / "data/features/biomedclip/ham10000_image_embeddings.npy"
SCORES_V5_NPZ    = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v5.npz"
TEXT_EMB_V6      = _ROOT / "data/features/biomedclip/ham10000_text_embeddings_v6.npy"
SCORES_V6_NPZ    = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v6.npz"

# ── Load v6 prompts ───────────────────────────────────────────────────────────
print("Loading v6 prompts...")
p = load_prompts(PROMPTS_V6)
assert len(p["prompts"]) == 24, f"Expected 24 prompts, got {len(p['prompts'])}"
assert set(p["prompt_template_idx"]) == {0}, "v6 should be single-template (t1 only)"
print(f"  {len(p['prompts'])} prompts across {len(p['concept_ids'])} concepts")

# ── Load BiomedCLIP ───────────────────────────────────────────────────────────
print("Loading BiomedCLIP...")
model, _, tokenizer, device = load_biomedclip()
print(f"  Device: {device}")

# ── Encode text ───────────────────────────────────────────────────────────────
print("Encoding 24 text prompts...")
text_emb = encode_prompts(model, tokenizer, p["prompts"], device)  # (24, 512)
assert text_emb.shape == (24, 512), f"Unexpected text_emb shape: {text_emb.shape}"

np.save(TEXT_EMB_V6, text_emb.numpy())
print(f"  Text embeddings saved → {TEXT_EMB_V6.relative_to(_ROOT)}")

# ── Load cached image embeddings ──────────────────────────────────────────────
print("Loading cached image embeddings...")
image_emb = np.load(IMAGE_EMBEDDINGS)   # (10015, 512) float32
assert image_emb.shape[1] == 512, f"Unexpected image_emb shape: {image_emb.shape}"
print(f"  Image embeddings: {image_emb.shape}")

# ── Compute similarity matrix ─────────────────────────────────────────────────
print("Computing cosine similarity matrix (10015, 24)...")
image_t = torch.tensor(image_emb, dtype=torch.float32)
text_t  = text_emb                         # already float32 on CPU

scores = (image_t @ text_t.T).numpy()     # (10015, 24)
assert scores.shape == (10015, 24), f"Unexpected scores shape: {scores.shape}"

s_min, s_max, s_mean = scores.min(), scores.max(), scores.mean()
assert -1.0 <= s_min and s_max <= 1.0, \
    f"Scores out of [-1, 1]: min={s_min:.4f} max={s_max:.4f}"
print(f"  Scores: min={s_min:.4f}  max={s_max:.4f}  mean={s_mean:.4f}")

# ── Copy alignment arrays from v5 ────────────────────────────────────────────
print("Loading image_ids / labels / lesion_ids from v5 npz (for alignment)...")
v5 = np.load(SCORES_V5_NPZ, allow_pickle=True)
image_ids  = v5["image_ids"]
labels     = v5["labels"]
lesion_ids = v5["lesion_ids"]
assert len(image_ids) == scores.shape[0], "Row count mismatch with v5"

# ── Save v6 npz ───────────────────────────────────────────────────────────────
print(f"Saving v6 scores → {SCORES_V6_NPZ.relative_to(_ROOT)}")
np.savez(
    SCORES_V6_NPZ,
    scores      = scores.astype(np.float32),
    image_ids   = image_ids,
    labels      = labels,
    lesion_ids  = lesion_ids,
    concept_ids = np.array(p["concept_ids"]),
    prompts     = np.array(p["prompts"]),
    tiers       = np.array(p["tiers"], dtype=np.int32),
)

# ── Class distribution ────────────────────────────────────────────────────────
unique, counts = np.unique(labels, return_counts=True)
print("\nClass distribution:")
for cls, cnt in sorted(zip(unique, counts)):
    print(f"  {cls}: {cnt} ({cnt / len(labels) * 100:.1f}%)")

print(f"\nDone. Scores shape: {scores.shape}  dtype: float32")
print(f"  → {SCORES_V6_NPZ.relative_to(_ROOT)}")
