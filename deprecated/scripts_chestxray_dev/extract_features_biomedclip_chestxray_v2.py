"""
Extract BiomedCLIP v2 concept score features for Chest X-ray (Kermany dataset).

Mirror of extract_features_biomedclip_chestxray_v1.py with only these changes:
  1. Loads prompts from chestxray_prompts_v2.txt
  2. Saves scores to chestxray_concept_scores_v2.npz
  3. Verifies bit-identity against v1 for unchanged concepts before saving

Everything else — model loading, preprocessing, image ordering, normalisation,
seeds, determinism flags — is identical to the v1 script.

src/ imports reused from HAM10000 pipeline — NOT modified:
    from src.features.clip_loader   import load_biomedclip
    from src.features.encode_text   import encode_prompts
    from src.features.prompt_loader import load_prompts

Outputs:
    data/features/biomedclip/chestxray_concept_scores_v2.npz  (N_images, 18)

Run from project root:
    python scripts/chestxray/extract_features_biomedclip_chestxray_v2.py
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

from src.features.clip_loader   import load_biomedclip
from src.features.encode_text   import encode_prompts
from src.features.prompt_loader import load_prompts

# ── Reproducibility (identical to v1) ────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Paths ─────────────────────────────────────────────────────────────────────
PROMPTS_V2   = _ROOT / "src/features/prompts/chestxray_prompts_v2.txt"
SCORES_V1    = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v1.npz"
OUTER_SPLIT  = _ROOT / "data/splits/chestxray_outer_split.npz"
DATA_ROOT    = _ROOT / "data/chest_xray"
SCORES_OUT   = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v2.npz"
FEATURES_DIR = _ROOT / "data/features/biomedclip"

BATCH_SIZE  = 32
N_CONCEPTS  = 18   # v2 = 18 concepts (13 unchanged + 5 revised); v1 had 19
BIT_ID_ATOL = 1e-6  # float32 tolerance for byte-identical-prompt score identity
PROMPTS_V1  = _ROOT / "src/features/prompts/chestxray_prompts_v1.txt"
DROPPED_EXPECTED = {"no_focal_opacity"}  # removed in v2 (redundant w/ clear_lung_fields)

# ── Version banner ────────────────────────────────────────────────────────────
_oc_version = getattr(open_clip, "__version__", None) or getattr(
    __import__("importlib.metadata", fromlist=["version"]), "version", lambda _: "unknown"
)("open_clip_torch")
print(f"open_clip version : {_oc_version}")
print(f"BiomedCLIP model  : hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
print(f"Prompt version    : v2")

# ── Load v2 prompts ───────────────────────────────────────────────────────────
print(f"\nLoading prompts from {PROMPTS_V2.relative_to(_ROOT)} ...")
p = load_prompts(PROMPTS_V2)
assert len(p["prompts"]) == N_CONCEPTS, (
    f"Expected {N_CONCEPTS} prompts, got {len(p['prompts'])}. "
    "Check chestxray_prompts_v2.txt for single-template (t1 only) entries."
)
assert set(p["prompt_template_idx"]) == {0}, \
    "v2 should be single-template (t1 only); found multiple template indices"

# Derive change sets by comparing v2 prompt text against v1 (byte-for-byte)
_p_v1   = load_prompts(PROMPTS_V1)
_v1_map = dict(zip(_p_v1["concept_ids"], _p_v1["prompts"]))
_v2_map = dict(zip(p["concept_ids"],     p["prompts"]))
_v1_set = set(_p_v1["concept_ids"])
_v2_set = set(p["concept_ids"])
NEW_IN_V2          = _v2_set - _v1_set                                   # {} expected
DROPPED_FROM_V1    = _v1_set - _v2_set                                   # {} expected
REVISED_CONCEPTS   = {c for c in _v1_set & _v2_set if _v1_map[c] != _v2_map[c]}
UNCHANGED_CONCEPTS = {c for c in _v1_set & _v2_set if _v1_map[c] == _v2_map[c]}

print(f"  {len(p['prompts'])} prompts across {len(p['concept_ids'])} concepts")
print(f"  Change audit (by prompt text vs v1): "
      f"{len(REVISED_CONCEPTS)} revised, {len(UNCHANGED_CONCEPTS)} byte-identical, "
      f"{len(NEW_IN_V2)} new, {len(DROPPED_FROM_V1)} dropped")
for cid, prt in zip(p["concept_ids"], p["prompts"]):
    if cid in NEW_IN_V2:
        tag = " [NEW]"
    elif cid in REVISED_CONCEPTS:
        tag = " [REVISED]"
    else:
        tag = " [UNCHANGED]"
    print(f"    [{cid}]{tag}  {prt[:72]}")

# ── Drop verification ─────────────────────────────────────────────────────────
assert DROPPED_FROM_V1 == DROPPED_EXPECTED, (
    f"Expected dropped concepts {DROPPED_EXPECTED}, got {DROPPED_FROM_V1}"
)
assert "no_focal_opacity" not in p["concept_ids"], \
    "no_focal_opacity should be DROPPED in v2 but is present"
assert not NEW_IN_V2, f"v2 should add no new concepts, got {NEW_IN_V2}"
print(f"  Drop verification ✓ (no_focal_opacity absent; v2 has {len(p['concept_ids'])} concepts)")

# ── Load BiomedCLIP (identical to v1) ────────────────────────────────────────
print("\nLoading BiomedCLIP (frozen)...")
model, preprocess, tokenizer, device = load_biomedclip()
print(f"  Device: {device}")

# ── Encode v2 text prompts ────────────────────────────────────────────────────
print(f"\nEncoding {N_CONCEPTS} text prompts...")
text_emb = encode_prompts(model, tokenizer, p["prompts"], device)  # (18, 512)
assert text_emb.shape == (N_CONCEPTS, 512), \
    f"Unexpected text_emb shape: {text_emb.shape}"
print(f"  text_emb shape: {text_emb.shape}  (L2-normalised, on CPU)")

# ── Load image paths (same split as v1) ──────────────────────────────────────
print(f"\nLoading outer split from {OUTER_SPLIT.relative_to(_ROOT)} ...")
split     = np.load(OUTER_SPLIT, allow_pickle=True)
rel_paths = split["paths"]
N         = len(rel_paths)
print(f"  Total images in split: {N}")
abs_paths = [DATA_ROOT / rp for rp in rel_paths]

# ── Batched image encoding (verbatim from v1 / encode_images.py) ─────────────

def _encode_images_batched(
    model: torch.nn.Module,
    preprocess,
    image_paths: list,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[int]]:
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
            "No images were successfully encoded. Check DATA_ROOT.\n"
            f"DATA_ROOT = {DATA_ROOT}"
        )
    return torch.cat(all_embeddings, dim=0).numpy(), success_indices


print("\nEncoding images...")
image_emb, success_idx = _encode_images_batched(
    model, preprocess, abs_paths, device, batch_size=BATCH_SIZE
)
print(f"  Successfully encoded: {len(success_idx)} / {N} images")
if len(success_idx) < N:
    print(f"  Failed indices: {sorted(set(range(N)) - set(success_idx))}")

# ── Compute cosine similarity (N, 18) ────────────────────────────────────────
print("\nComputing cosine similarity scores...")
image_t   = torch.tensor(image_emb, dtype=torch.float32)
scores_ok = (image_t @ text_emb.T).numpy()  # (N_ok, 18)

scores = np.full((N, N_CONCEPTS), np.nan, dtype=np.float32)
scores[success_idx] = scores_ok

s_min  = np.nanmin(scores)
s_max  = np.nanmax(scores)
s_mean = np.nanmean(scores)
assert -1.0 <= s_min and s_max <= 1.0, \
    f"Scores out of [-1, 1]: min={s_min:.4f} max={s_max:.4f}"
print(f"  Scores shape: {scores.shape}  min={s_min:.4f}  max={s_max:.4f}  mean={s_mean:.4f}")

# ── Per-concept variance check ────────────────────────────────────────────────
print("\nPer-concept mean ± std:")
concept_ids = p["concept_ids"]
all_ok = True
for i, cid in enumerate(concept_ids):
    col_valid = scores[:, i][~np.isnan(scores[:, i])]
    mean_val  = col_valid.mean()
    std_val   = col_valid.std()
    flag      = "" if std_val > 1e-6 else "  ← WARN: near-constant!"
    all_ok    = all_ok and (std_val > 1e-6)
    if cid in NEW_IN_V2:            tag = " [NEW]"
    elif cid in REVISED_CONCEPTS:   tag = " [REVISED]"
    else:                           tag = " [UNCHANGED]"
    print(f"  {cid:<40s}{tag:<12s}  mean={mean_val:.4f}  std={std_val:.4f}{flag}")

if not all_ok:
    print("\nWARNING: one or more concepts have near-constant scores — check prompt file")
else:
    print("\nAll concepts have non-constant variance ✓")

# ── Bit-identity check ────────────────────────────────────────────────────────
# v2 keeps 12 concept prompts byte-identical to v1. With frozen weights, fixed
# seeds, and identical preprocessing/image-order, those 12 columns must match
# v1's scores to float32 tolerance. A mismatch means the extraction pipeline is
# non-deterministic and the v1↔v2 comparison would be invalid — so we abort.
print(f"\nBit-identity check (atol={BIT_ID_ATOL}):")
print(f"  Concepts with byte-identical prompt text in v1 and v2: {len(UNCHANGED_CONCEPTS)}")
assert len(UNCHANGED_CONCEPTS) == 13, (
    f"Expected 13 byte-identical concepts vs v1, got {len(UNCHANGED_CONCEPTS)}: "
    f"{sorted(UNCHANGED_CONCEPTS)}. Re-check the v1/v2 prompt files."
)
if UNCHANGED_CONCEPTS:
    v1_data          = np.load(SCORES_V1, allow_pickle=True)
    v1_scores        = v1_data["scores"]
    v1_concept_names = v1_data["concept_names"].tolist()
    unchanged_fail   = False
    for cid in UNCHANGED_CONCEPTS:
        v1_ci = v1_concept_names.index(cid)
        v2_ci = concept_ids.index(cid)
        valid = ~np.isnan(v1_scores[:, v1_ci]) & ~np.isnan(scores[:, v2_ci])
        match = np.allclose(v1_scores[:, v1_ci][valid], scores[:, v2_ci][valid],
                            atol=BIT_ID_ATOL)
        max_diff = float(np.abs(v1_scores[:, v1_ci][valid] - scores[:, v2_ci][valid]).max())
        print(f"  [{cid}]  {'OK' if match else f'FAIL max_diff={max_diff:.2e}'}")
        if not match:
            unchanged_fail = True
    if unchanged_fail:
        raise RuntimeError(
            "Bit-identity FAILED for a prompt-unchanged concept — "
            "non-determinism in the extraction pipeline. Do not save v2 scores."
        )
    print(f"  Bit-identity PASSED for {len(UNCHANGED_CONCEPTS)} unchanged concept(s) ✓")
else:
    print("  All shared prompts were revised — bit-identity check not applicable ✓")
    print(f"  Revised: {len(REVISED_CONCEPTS)} shared  |  New: {len(NEW_IN_V2)}"
          f"  |  Dropped: {len(DROPPED_FROM_V1)}")

# ── Save v2 scores ────────────────────────────────────────────────────────────
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nSaving → {SCORES_OUT.relative_to(_ROOT)}")
np.savez(
    SCORES_OUT,
    scores        = scores.astype(np.float32),
    concept_names = np.array(concept_ids),
    image_paths   = rel_paths,
    prompts       = np.array(p["prompts"]),
)
print(f"  scores       : {scores.shape}  float32")
print(f"  concept_names: {concept_ids}")
print(f"\nDone.")
