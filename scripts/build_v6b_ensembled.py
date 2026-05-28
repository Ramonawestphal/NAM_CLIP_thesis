"""
Build BiomedCLIP v6b ensembled features for HAM10000.

Collapses v5_biomedclip's 72-prompt text embedding matrix to 24 ensembled
embeddings using the Radford et al. 2021 template-ensemble technique:
  1. Average the three template embeddings per concept element-wise.
  2. L2-normalise the averaged vector.
  3. Recompute similarity against cached image embeddings.

Does NOT re-encode any prompts or images. All computation is on cached
embeddings only.

Outputs:
    data/features/biomedclip/ham10000_text_embeddings_v6b.npy  (24, 512)
    data/features/biomedclip/ham10000_concept_scores_v6b.npz   (10015, 24)

Run from project root:
    python scripts/build_v6b_ensembled.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
TEXT_EMB_V5      = _ROOT / "data/features/biomedclip/ham10000_text_embeddings_v5.npy"
IMAGE_EMBEDDINGS = _ROOT / "data/features/biomedclip/ham10000_image_embeddings.npy"
SCORES_V5_NPZ    = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v5.npz"
TEXT_EMB_V6B     = _ROOT / "data/features/biomedclip/ham10000_text_embeddings_v6b.npy"
SCORES_V6B_NPZ   = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v6b.npz"

# ── Load v5 text embeddings + metadata ───────────────────────────────────────
print("Loading v5 text embeddings...")
text_emb_v5 = np.load(TEXT_EMB_V5)                    # (72, 512)
assert text_emb_v5.shape == (72, 512), f"Unexpected shape: {text_emb_v5.shape}"

print("Loading v5 npz metadata...")
v5 = np.load(SCORES_V5_NPZ, allow_pickle=True)
concept_ids       = list(v5["concept_ids"])            # 24 strings, concept order
tiers             = v5["tiers"]                        # (24,) int
prompt_concept_idx = v5["prompt_concept_idx"]          # (72,) concept index per row
prompt_template_idx = v5["prompt_template_idx"]        # (72,) template index per row
prompts_v5        = v5["prompts"]                      # (72,) prompt strings

assert len(concept_ids) == 24
assert len(prompt_concept_idx) == 72

# ── Verify / build concept→row mapping ───────────────────────────────────────
# Build mapping robustly from the npz arrays rather than assuming stride-3.
concept_row_map: dict[int, list[int]] = {}
for row_idx, c_idx in enumerate(prompt_concept_idx):
    concept_row_map.setdefault(int(c_idx), []).append(row_idx)

assert len(concept_row_map) == 24, f"Expected 24 concepts, got {len(concept_row_map)}"
for c_idx, rows in concept_row_map.items():
    assert len(rows) == 3, \
        f"Concept {concept_ids[c_idx]} has {len(rows)} template rows (expected 3)"

print(f"  Row ordering verified — {len(concept_ids)} concepts × 3 templates = 72 rows")
print(f"  Sample: concept_idx[:12] = {list(prompt_concept_idx[:12])}")
print(f"  Sample: template_idx[:12] = {list(prompt_template_idx[:12])}")

# ── Ensemble: average + L2-normalise per concept ──────────────────────────────
print("\nEnsembling 3 templates per concept (average + L2-normalise)...")
ensembled = np.zeros((24, 512), dtype=np.float64)

for c_idx in range(24):
    rows = concept_row_map[c_idx]           # 3 row indices in text_emb_v5
    avg  = text_emb_v5[rows, :].mean(axis=0).astype(np.float64)
    norm = np.linalg.norm(avg)
    assert norm > 0, f"Zero-norm embedding for concept {concept_ids[c_idx]}"
    ensembled[c_idx] = avg / norm

ensembled = ensembled.astype(np.float32)

# Sanity: L2 norms should all be 1.0 (within float32 tolerance)
norms = np.linalg.norm(ensembled, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5), \
    f"L2 norms not all 1.0 after normalisation: min={norms.min():.6f} max={norms.max():.6f}"
print(f"  Ensembled embeddings: {ensembled.shape}  "
      f"L2 norms in [{norms.min():.5f}, {norms.max():.5f}]")

# Save ensembled text embeddings
np.save(TEXT_EMB_V6B, ensembled)
print(f"  Saved → {TEXT_EMB_V6B.relative_to(_ROOT)}")

# ── Load image embeddings, compute similarity ─────────────────────────────────
print("\nLoading cached image embeddings...")
image_emb = np.load(IMAGE_EMBEDDINGS)                  # (10015, 512) float32
assert image_emb.shape == (10015, 512), f"Unexpected shape: {image_emb.shape}"

print("Computing similarity matrix (10015, 24)...")
scores = (image_emb @ ensembled.T).astype(np.float32)  # (10015, 24)
assert scores.shape == (10015, 24)

s_min, s_max, s_mean = float(scores.min()), float(scores.max()), float(scores.mean())
assert -1.0 <= s_min and s_max <= 1.0, \
    f"Scores outside [-1, 1]: min={s_min:.4f} max={s_max:.4f}"
print(f"  Scores: min={s_min:.4f}  max={s_max:.4f}  mean={s_mean:.4f}")

# ── Build documentation prompts (ensemble label per concept) ──────────────────
# For each concept, record the 3 v5 prompt strings concatenated as a note.
doc_prompts = []
for c_idx in range(24):
    rows   = concept_row_map[c_idx]
    joined = " | ".join(str(prompts_v5[r]) for r in rows)
    doc_prompts.append(f"[ensemble of 3 templates] {joined}")

# ── Save v6b scores npz ───────────────────────────────────────────────────────
print(f"\nSaving → {SCORES_V6B_NPZ.relative_to(_ROOT)}")
np.savez(
    SCORES_V6B_NPZ,
    scores              = scores,
    image_ids           = v5["image_ids"],
    labels              = v5["labels"],
    lesion_ids          = v5["lesion_ids"],
    concept_ids         = np.array(concept_ids),
    tiers               = tiers,
    source              = np.array("v5_template_ensemble"),
    prompts             = np.array(doc_prompts),
)

# ── Class distribution ────────────────────────────────────────────────────────
labels = v5["labels"]
unique, counts = np.unique(labels, return_counts=True)
print("\nClass distribution:")
for cls, cnt in sorted(zip(unique.tolist(), counts.tolist())):
    print(f"  {cls}: {cnt} ({cnt / len(labels) * 100:.1f}%)")

print(f"\nDone.  scores shape: {scores.shape}  dtype: float32")
print(f"  → {SCORES_V6B_NPZ.relative_to(_ROOT)}")
