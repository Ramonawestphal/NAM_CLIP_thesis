"""
v3 Step 2: smoke-test feature extraction for the 4 candidate prompts.

Encodes the 4 candidate phrasings (2 per failing concept) and computes cosine
similarity against the 300-image stratified subsample only. BiomedCLIP loading,
preprocessing, seeds, and determinism flags are identical to the v1/v2 extractors.

Outputs:
    data/features/biomedclip/chestxray_concept_scores_v3_smoketest.npz
        keys: scores (300x4), candidate_names, prompts, subsample_idx

Run from project root (after v3_select_smoketest_subsample.py):
    python scripts/chestxray/v3_smoketest_extract.py
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

# ── Reproducibility (identical to v1/v2) ──────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Paths ─────────────────────────────────────────────────────────────────────
CANDIDATES   = _ROOT / "src/features/prompts/chestxray_prompts_v3_smoketest_candidates.txt"
OUTER_SPLIT  = _ROOT / "data/splits/chestxray_outer_split.npz"
SUBSAMPLE    = _ROOT / "results/chestxray/v3_smoketest/subsample_indices.npz"
DATA_ROOT    = _ROOT / "data/chest_xray"
SCORES_OUT   = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v3_smoketest.npz"
FEATURES_DIR = _ROOT / "data/features/biomedclip"

N_CANDIDATES = 4

_oc_version = getattr(open_clip, "__version__", None) or getattr(
    __import__("importlib.metadata", fromlist=["version"]), "version", lambda _: "unknown"
)("open_clip_torch")
print(f"open_clip version : {_oc_version}")
print(f"BiomedCLIP model  : hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
print(f"Prompt version    : v3-smoketest")

# ── Load candidates ───────────────────────────────────────────────────────────
print(f"\nLoading candidates from {CANDIDATES.relative_to(_ROOT)} ...")
p = load_prompts(CANDIDATES)
assert len(p["prompts"]) == N_CANDIDATES, \
    f"Expected {N_CANDIDATES} candidate prompts, got {len(p['prompts'])}"
assert set(p["prompt_template_idx"]) == {0}, "candidates should be single-template (t1)"
candidate_names = p["concept_ids"]
print(f"  {len(candidate_names)} candidates:")
for cid, prt in zip(candidate_names, p["prompts"]):
    print(f"    [{cid}]  {prt[:80]}")

# ── Load BiomedCLIP ───────────────────────────────────────────────────────────
print("\nLoading BiomedCLIP (frozen)...")
model, preprocess, tokenizer, device = load_biomedclip()
print(f"  Device: {device}")

# ── Encode candidate text ─────────────────────────────────────────────────────
print(f"\nEncoding {N_CANDIDATES} candidate prompts...")
text_emb = encode_prompts(model, tokenizer, p["prompts"], device)  # (4, 512)
assert text_emb.shape == (N_CANDIDATES, 512), f"Unexpected shape: {text_emb.shape}"

# ── Load subsample image paths ────────────────────────────────────────────────
print(f"\nLoading subsample from {SUBSAMPLE.relative_to(_ROOT)} ...")
sub = np.load(SUBSAMPLE, allow_pickle=True)
subsample_idx = sub["subsample_idx"]
split = np.load(OUTER_SPLIT, allow_pickle=True)
rel_paths = split["paths"]
abs_paths = [DATA_ROOT / rel_paths[i] for i in subsample_idx]
N = len(subsample_idx)
print(f"  Subsample images: {N}")


def _encode_images_batched(model, preprocess, image_paths, device, batch_size=32):
    """Verbatim batched image encoder (matches v1/v2)."""
    all_emb, success = [], []
    batches = [
        (list(range(i, min(i + batch_size, len(image_paths)))),
         image_paths[i : i + batch_size])
        for i in range(0, len(image_paths), batch_size)
    ]
    for indices, batch_paths in tqdm(batches, desc="Encoding subsample", unit="batch"):
        tensors, valid = [], []
        for idx, pth in zip(indices, batch_paths):
            try:
                img = Image.open(pth).convert("RGB")
                tensors.append(preprocess(img))
                valid.append(idx)
            except Exception as exc:
                print(f"[smoketest] skipping {pth.name}: {exc}")
        if not tensors:
            continue
        bt = torch.stack(tensors).to(device)
        with torch.no_grad():
            emb = model.encode_image(bt)
        emb = F.normalize(emb, dim=1)
        all_emb.append(emb.cpu().float())
        success.extend(valid)
    if not all_emb:
        raise RuntimeError(f"No images encoded. Check DATA_ROOT={DATA_ROOT}")
    return torch.cat(all_emb, dim=0).numpy(), success


print("\nEncoding subsample images...")
image_emb, success_local = _encode_images_batched(model, preprocess, abs_paths, device)
print(f"  Encoded: {len(success_local)} / {N}")

# success_local holds positions into abs_paths (0..N-1); align scores by those
scores = np.full((N, N_CANDIDATES), np.nan, dtype=np.float32)
image_t = torch.tensor(image_emb, dtype=torch.float32)
scores_ok = (image_t @ text_emb.T).numpy()
scores[success_local] = scores_ok

s_min, s_max = np.nanmin(scores), np.nanmax(scores)
assert -1.0 <= s_min and s_max <= 1.0, f"Scores out of [-1,1]: {s_min},{s_max}"

# ── Per-candidate sanity ──────────────────────────────────────────────────────
print("\nPer-candidate mean ± std (subsample):")
for i, cid in enumerate(candidate_names):
    col = scores[:, i][~np.isnan(scores[:, i])]
    print(f"  {cid:<40s}  mean={col.mean():.4f}  std={col.std():.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
np.savez(
    SCORES_OUT,
    scores          = scores.astype(np.float32),
    candidate_names = np.array(candidate_names),
    prompts         = np.array(p["prompts"]),
    subsample_idx   = subsample_idx,
)
print(f"\nSaved → {SCORES_OUT.relative_to(_ROOT)}  (shape {scores.shape})")
