"""
Extract BiomedCLIP v3 concept score features for Chest X-ray (Kermany dataset).

Mirror of extract_features_biomedclip_chestxray_v2.py with only these changes:
  1. Loads prompts from chestxray_prompts_v3.txt
  2. Saves scores to chestxray_concept_scores_v3.npz
  3. Verifies bit-identity against v2 for the 16 FROZEN concepts before saving

v3 = 18 concepts: 16 frozen from v2 (byte-identical) + 2 smoke-test winners
(bilateral_interstitial_pattern, hyperinflation). Everything else — model
loading, preprocessing, image ordering, normalisation, seeds, determinism
flags — is identical to the v1/v2 scripts.

src/ imports reused from HAM10000 pipeline — NOT modified:
    from src.features.clip_loader   import load_biomedclip
    from src.features.encode_text   import encode_prompts
    from src.features.prompt_loader import load_prompts

Outputs:
    data/features/biomedclip/chestxray_concept_scores_v3.npz  (N_images, 18)

Run from project root:
    python scripts/chestxray/extract_features_biomedclip_chestxray_v3.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # robust on cp1252 consoles
except Exception:
    pass

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
PROMPTS_V3   = _ROOT / "src/features/prompts/chestxray_prompts_v3.txt"
PROMPTS_V2   = _ROOT / "src/features/prompts/chestxray_prompts_v2.txt"   # baseline
SCORES_V2    = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v2.npz"
OUTER_SPLIT  = _ROOT / "data/splits/chestxray_outer_split.npz"
DATA_ROOT    = _ROOT / "data/chest_xray"
SCORES_OUT   = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v3.npz"
FEATURES_DIR = _ROOT / "data/features/biomedclip"

BATCH_SIZE  = 32
N_CONCEPTS  = 18   # v3 = 18 concepts (16 frozen from v2 + 2 smoke-test winners)
BIT_ID_ATOL = 1e-6  # float32 tolerance for frozen-concept score identity

# ── Version banner ────────────────────────────────────────────────────────────
_oc_version = getattr(open_clip, "__version__", None) or getattr(
    __import__("importlib.metadata", fromlist=["version"]), "version", lambda _: "unknown"
)("open_clip_torch")
print(f"open_clip version : {_oc_version}")
print(f"BiomedCLIP model  : hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
print(f"Prompt version    : v3")

# ── Load v3 prompts ───────────────────────────────────────────────────────────
print(f"\nLoading prompts from {PROMPTS_V3.relative_to(_ROOT)} ...")
p = load_prompts(PROMPTS_V3)
assert len(p["prompts"]) == N_CONCEPTS, (
    f"Expected {N_CONCEPTS} prompts, got {len(p['prompts'])}. "
    "Check chestxray_prompts_v3.txt for single-template (t1 only) entries."
)
assert set(p["prompt_template_idx"]) == {0}, \
    "v3 should be single-template (t1 only); found multiple template indices"

# Derive change sets by comparing v3 prompt text against v2 (byte-for-byte)
_p_v2   = load_prompts(PROMPTS_V2)
_v2_map = dict(zip(_p_v2["concept_ids"], _p_v2["prompts"]))
_v3_map = dict(zip(p["concept_ids"],     p["prompts"]))
_v2_set = set(_p_v2["concept_ids"])
_v3_set = set(p["concept_ids"])
NEW_IN_V3          = _v3_set - _v2_set                                   # {} expected
DROPPED_FROM_V2    = _v2_set - _v3_set                                   # {} expected
REVISED_CONCEPTS   = {c for c in _v2_set & _v3_set if _v2_map[c] != _v3_map[c]}
FROZEN_CONCEPTS    = {c for c in _v2_set & _v3_set if _v2_map[c] == _v3_map[c]}

print(f"  {len(p['prompts'])} prompts across {len(p['concept_ids'])} concepts")
print(f"  Change audit (by prompt text vs v2): "
      f"{len(REVISED_CONCEPTS)} revised, {len(FROZEN_CONCEPTS)} frozen, "
      f"{len(NEW_IN_V3)} new, {len(DROPPED_FROM_V2)} dropped")
for cid, prt in zip(p["concept_ids"], p["prompts"]):
    if cid in NEW_IN_V3:
        tag = " [NEW]"
    elif cid in REVISED_CONCEPTS:
        tag = " [REVISED]"
    else:
        tag = " [FROZEN]"
    print(f"    [{cid}]{tag}  {prt[:72]}")

# ── Change-set verification ───────────────────────────────────────────────────
assert not DROPPED_FROM_V2, f"v3 should drop no concepts, got {DROPPED_FROM_V2}"
assert not NEW_IN_V3, f"v3 should add no concepts, got {NEW_IN_V3}"
assert REVISED_CONCEPTS == {"bilateral_interstitial_pattern", "hyperinflation"}, (
    f"v3 should revise exactly the 2 failing concepts, got {REVISED_CONCEPTS}"
)
assert len(FROZEN_CONCEPTS) == 16, f"Expected 16 frozen concepts, got {len(FROZEN_CONCEPTS)}"
print(f"  Change-set verification ✓ (16 frozen + 2 revised; no adds/drops)")

# ── Load BiomedCLIP (identical to v1) ────────────────────────────────────────
print("\nLoading BiomedCLIP (frozen)...")
model, preprocess, tokenizer, device = load_biomedclip()
print(f"  Device: {device}")

# ── Encode v3 text prompts ────────────────────────────────────────────────────
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
    if cid in NEW_IN_V3:            tag = " [NEW]"
    elif cid in REVISED_CONCEPTS:   tag = " [REVISED]"
    else:                           tag = " [FROZEN]"
    print(f"  {cid:<40s}{tag:<12s}  mean={mean_val:.4f}  std={std_val:.4f}{flag}")

if not all_ok:
    print("\nWARNING: one or more concepts have near-constant scores — check prompt file")
else:
    print("\nAll concepts have non-constant variance ✓")

# ── Bit-identity check ────────────────────────────────────────────────────────
# v3 freezes 16 concept prompts byte-identical to v2. With frozen weights, fixed
# seeds, and identical preprocessing/image-order, those 16 columns must match
# v2's scores to float32 tolerance. A mismatch means the extraction pipeline is
# non-deterministic and the v2↔v3 comparison would be invalid — so we abort.
print(f"\nBit-identity check vs v2 (atol={BIT_ID_ATOL}):")
print(f"  Frozen concepts (byte-identical prompt text in v2 and v3): {len(FROZEN_CONCEPTS)}")
assert len(FROZEN_CONCEPTS) == 16, (
    f"Expected 16 frozen concepts vs v2, got {len(FROZEN_CONCEPTS)}: "
    f"{sorted(FROZEN_CONCEPTS)}. Re-check the v2/v3 prompt files."
)
v2_data          = np.load(SCORES_V2, allow_pickle=True)
v2_scores        = v2_data["scores"]
v2_concept_names = v2_data["concept_names"].tolist()
frozen_fail      = False
for cid in sorted(FROZEN_CONCEPTS):
    v2_ci = v2_concept_names.index(cid)
    v3_ci = concept_ids.index(cid)
    valid = ~np.isnan(v2_scores[:, v2_ci]) & ~np.isnan(scores[:, v3_ci])
    match = np.allclose(v2_scores[:, v2_ci][valid], scores[:, v3_ci][valid],
                        atol=BIT_ID_ATOL)
    max_diff = float(np.abs(v2_scores[:, v2_ci][valid] - scores[:, v3_ci][valid]).max())
    print(f"  [{cid}]  {'OK' if match else f'FAIL max_diff={max_diff:.2e}'}")
    if not match:
        frozen_fail = True
if frozen_fail:
    raise RuntimeError(
        "Bit-identity FAILED for a frozen concept — non-determinism in the "
        "extraction pipeline. Do not save v3 scores; the v2↔v3 comparison "
        "would be invalid."
    )
print(f"  Bit-identity PASSED for {len(FROZEN_CONCEPTS)} frozen concept(s) ✓")

# ── Save v3 scores ────────────────────────────────────────────────────────────
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
