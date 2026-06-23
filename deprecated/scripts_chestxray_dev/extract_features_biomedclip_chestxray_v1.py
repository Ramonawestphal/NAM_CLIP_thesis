"""
Extract BiomedCLIP v1 concept score features for Chest X-ray (Kermany dataset).

Parallel to scripts/extract_features_biomedclip_v6.py (HAM10000).
Methodological parity is maintained by reusing the same src/ imports:
    from src.features.clip_loader  import load_biomedclip
    from src.features.encode_text  import encode_prompts
    from src.features.prompt_loader import load_prompts

The image encoding loop is reproduced locally from src/features/encode_images.py
verbatim — no shared modules were modified to accommodate this dataset.

Outputs:
    data/features/biomedclip/chestxray_concept_scores_v1.npz  (N_images, 19)

Run from project root:
    python scripts/chestxray/extract_features_biomedclip_chestxray_v1.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# src/ imports reused from HAM10000 pipeline — NOT modified
from src.features.clip_loader   import load_biomedclip
from src.features.encode_text   import encode_prompts
from src.features.prompt_loader import load_prompts

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Paths ─────────────────────────────────────────────────────────────────────
PROMPTS_V1   = _ROOT / "src/features/prompts/chestxray_prompts_v1.txt"
OUTER_SPLIT  = _ROOT / "data/splits/chestxray_outer_split.npz"
DATA_ROOT    = _ROOT / "data/chest_xray"
SCORES_OUT   = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v1.npz"
FEATURES_DIR = _ROOT / "data/features/biomedclip"

BATCH_SIZE   = 32
N_CONCEPTS   = 19

# ── Version banner ────────────────────────────────────────────────────────────
_oc_version = getattr(open_clip, "__version__", None) or getattr(
    __import__("importlib.metadata", fromlist=["version"]), "version", lambda _: "unknown"
)("open_clip_torch")
print(f"open_clip version : {_oc_version}")
print(f"BiomedCLIP model  : hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

# ── Load prompts ──────────────────────────────────────────────────────────────
print(f"\nLoading prompts from {PROMPTS_V1.relative_to(_ROOT)} ...")
p = load_prompts(PROMPTS_V1)
assert len(p["prompts"]) == N_CONCEPTS, (
    f"Expected {N_CONCEPTS} prompts, got {len(p['prompts'])}. "
    "Check chestxray_prompts_v1.txt for single-template (t1 only) entries."
)
assert set(p["prompt_template_idx"]) == {0}, \
    "v1 should be single-template (t1 only); found multiple template indices"
print(f"  {len(p['prompts'])} prompts across {len(p['concept_ids'])} concepts")
for cid, prt in zip(p["concept_ids"], p["prompts"]):
    print(f"    [{cid}]  {prt[:80]}")

# ── Load BiomedCLIP ───────────────────────────────────────────────────────────
print("\nLoading BiomedCLIP (frozen)...")
model, preprocess, tokenizer, device = load_biomedclip()
print(f"  Device: {device}")

# ── Encode text prompts ───────────────────────────────────────────────────────
print(f"\nEncoding {N_CONCEPTS} text prompts...")
text_emb = encode_prompts(model, tokenizer, p["prompts"], device)  # (19, 512)
assert text_emb.shape == (N_CONCEPTS, 512), \
    f"Unexpected text_emb shape: {text_emb.shape}"
print(f"  text_emb shape: {text_emb.shape}  (L2-normalised, on CPU)")

# ── Load image paths from outer split ────────────────────────────────────────
print(f"\nLoading outer split from {OUTER_SPLIT.relative_to(_ROOT)} ...")
split = np.load(OUTER_SPLIT, allow_pickle=True)
rel_paths = split["paths"]        # relative paths, e.g. "train/NORMAL/IM-0001-0001.jpeg"
N = len(rel_paths)
print(f"  Total images in split: {N}")

abs_paths = [DATA_ROOT / rp for rp in rel_paths]

# ── Batched image encoding loop (verbatim from src/features/encode_images.py) ─

def _encode_images_batched(
    model: torch.nn.Module,
    preprocess,
    image_paths: list,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[int]]:
    """Encode images in batches; return (embeddings, successful_indices).

    Failed images are logged and their global index is excluded from the result.
    The caller aligns failures back to the full index space.
    """
    all_embeddings: list[torch.Tensor] = []
    success_indices: list[int] = []

    batches = [
        (list(range(i, min(i + batch_size, len(image_paths)))),
         image_paths[i : i + batch_size])
        for i in range(0, len(image_paths), batch_size)
    ]

    for indices, batch_paths in tqdm(batches, desc="Encoding images", unit="batch"):
        tensors: list[torch.Tensor] = []
        valid_indices: list[int] = []

        for idx, p_path in zip(indices, batch_paths):
            try:
                img = Image.open(p_path).convert("RGB")
                tensors.append(preprocess(img))
                valid_indices.append(idx)
            except Exception as exc:
                print(f"[extract] skipping {p_path.name}: {exc}")

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            emb = model.encode_image(batch_tensor)
        emb = F.normalize(emb, dim=1)
        all_embeddings.append(emb.cpu().float())
        success_indices.extend(valid_indices)

    if not all_embeddings:
        raise RuntimeError(
            "No images were successfully encoded. Check that DATA_ROOT is correct "
            f"and image paths from the split file resolve under it.\n"
            f"DATA_ROOT = {DATA_ROOT}"
        )
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return embeddings, success_indices


print("\nEncoding images...")
image_emb, success_idx = _encode_images_batched(
    model, preprocess, abs_paths, device, batch_size=BATCH_SIZE
)
print(f"  Successfully encoded: {len(success_idx)} / {N} images")
if len(success_idx) < N:
    failed = sorted(set(range(N)) - set(success_idx))
    print(f"  Failed indices: {failed}")

# ── Compute cosine similarity (N, 19) ────────────────────────────────────────
print("\nComputing cosine similarity scores...")
image_t = torch.tensor(image_emb, dtype=torch.float32)   # (N_ok, 512)
text_t  = text_emb                                         # (19, 512)  L2-normalised

scores_ok = (image_t @ text_t.T).numpy()   # (N_ok, 19)

# Re-expand to full N rows (NaN for any failed images)
scores = np.full((N, N_CONCEPTS), np.nan, dtype=np.float32)
scores[success_idx] = scores_ok

s_min  = np.nanmin(scores)
s_max  = np.nanmax(scores)
s_mean = np.nanmean(scores)
assert -1.0 <= s_min and s_max <= 1.0, \
    f"Scores out of [-1, 1]: min={s_min:.4f} max={s_max:.4f}"
print(f"  Scores shape: {scores.shape}  min={s_min:.4f}  max={s_max:.4f}  mean={s_mean:.4f}")

# ── Per-concept diagnostics ───────────────────────────────────────────────────
print("\nPer-concept mean ± std (sanity check — all stds should be > 1e-6):")
concept_ids = p["concept_ids"]
all_ok = True
for i, cid in enumerate(concept_ids):
    col = scores[:, i]
    col_valid = col[~np.isnan(col)]
    mean_val = col_valid.mean()
    std_val  = col_valid.std()
    flag = "" if std_val > 1e-6 else "  ← WARN: near-constant!"
    all_ok = all_ok and (std_val > 1e-6)
    print(f"  {cid:<40s}  mean={mean_val:.4f}  std={std_val:.4f}{flag}")

if not all_ok:
    print("\nWARNING: one or more concepts have near-constant scores — check prompt file")
else:
    print("\nAll concepts have non-constant variance ✓")

# ── Save ─────────────────────────────────────────────────────────────────────
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nSaving → {SCORES_OUT.relative_to(_ROOT)}")
np.savez(
    SCORES_OUT,
    scores        = scores.astype(np.float32),
    concept_names = np.array(concept_ids),
    image_paths   = rel_paths,
    prompts       = np.array(p["prompts"]),
)
print(f"  scores      : {scores.shape}  float32")
print(f"  concept_names: {concept_ids}")
print(f"\nDone.")
