"""
Build BiomedCLIP v6c features: select the best single template per concept
from the v5 text embeddings based on training-partition intended-class AUC.

Selection criterion: for each concept, the template whose similarity score
achieves the highest one-vs-rest AUC for the concept's designed target class
on the TRAINING PARTITION ONLY. Test partition is never touched.

Outputs:
    reports/encoder_comparison/v6c_template_selection.csv
    data/features/biomedclip/ham10000_text_embeddings_v6c.npy   (24, 512)
    data/features/biomedclip/ham10000_concept_scores_v6c.npz    (10015, 24)

Run from project root:
    python scripts/select_best_templates_v6c.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analysis.concept_targets import CONCEPT_TARGET_CLASS

# ── Paths ─────────────────────────────────────────────────────────────────────
TEXT_EMB_V5       = _ROOT / "data/features/biomedclip/ham10000_text_embeddings_v5.npy"
IMAGE_EMBEDDINGS  = _ROOT / "data/features/biomedclip/ham10000_image_embeddings.npy"
SCORES_V5_NPZ     = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v5.npz"
SPLIT_NPZ         = _ROOT / "data/splits/train_test_lesion_split.npz"
SELECTION_CSV     = _ROOT / "reports/encoder_comparison/v6c_template_selection.csv"
TEXT_EMB_V6C      = _ROOT / "data/features/biomedclip/ham10000_text_embeddings_v6c.npy"
SCORES_V6C_NPZ    = _ROOT / "data/features/biomedclip/ham10000_concept_scores_v6c.npz"

# ── Step 1: Load v5 npz, print shapes, verify structure ──────────────────────
print("Loading v5 npz...")
v5 = np.load(SCORES_V5_NPZ, allow_pickle=True)

scores_v5          = v5["scores"]               # (10015, 72)
concept_ids        = list(v5["concept_ids"])    # 24 strings
prompt_concept_idx = v5["prompt_concept_idx"]   # (72,) int
prompt_template_idx = v5["prompt_template_idx"] # (72,) int
prompts_v5         = v5["prompts"]              # (72,) str
tiers              = v5["tiers"]                # (24,) int

print(f"  scores shape             : {scores_v5.shape}")
print(f"  concept_ids shape        : {len(concept_ids)} | unique: {len(set(concept_ids))}")
print(f"  prompt_concept_idx shape : {prompt_concept_idx.shape}")
print(f"  prompt_template_idx shape: {prompt_template_idx.shape}")
print(f"  prompts shape            : {prompts_v5.shape}")
print(f"  prompt_concept_idx[:9]   : {prompt_concept_idx[:9].tolist()}")
print(f"  prompt_template_idx[:9]  : {prompt_template_idx[:9].tolist()}")

assert scores_v5.shape == (10015, 72)
assert len(concept_ids) == 24

# Build concept_idx -> list of (col_in_scores, template_idx) using npz arrays
concept_cols: dict[int, list[tuple[int, int]]] = {}
for col, (c_idx, t_idx) in enumerate(zip(prompt_concept_idx, prompt_template_idx)):
    concept_cols.setdefault(int(c_idx), []).append((col, int(t_idx)))

for c_idx, entries in concept_cols.items():
    assert len(entries) == 3, \
        f"Concept {concept_ids[c_idx]} has {len(entries)} templates (expected 3)"
print(f"  Verified: all 24 concepts have exactly 3 templates")

# ── Step 2: Load training split ───────────────────────────────────────────────
train_idx = np.load(SPLIT_NPZ)["train_idx"]
labels_tr = v5["labels"][train_idx]
print(f"\nTraining partition: {len(train_idx):,} images")

# ── Step 3: Assert every concept_id has a target class mapping ───────────────
missing = [cid for cid in concept_ids if cid not in CONCEPT_TARGET_CLASS]
if missing:
    raise KeyError(
        f"These concept_ids are missing from CONCEPT_TARGET_CLASS: {missing}\n"
        "Update src/analysis/concept_targets.py before continuing."
    )
print(f"CONCEPT_TARGET_CLASS: all {len(concept_ids)} concept_ids present")

# ── Step 4: Template selection by training-partition intended-class AUC ───────
print("\nSelecting best template per concept (training-partition AUC)...")
selection_rows = []

v6c_text_rows   = np.zeros(24, dtype=np.int64)   # which row of text_emb_v5 to use
v6c_score_cols  = np.zeros(24, dtype=np.int64)   # which col of scores_v5 to use

for c_idx, cid in enumerate(concept_ids):
    target_cls = CONCEPT_TARGET_CLASS[cid]
    y_bin      = (labels_tr == target_cls).astype(int)
    entries    = sorted(concept_cols[c_idx], key=lambda x: x[1])  # sort by template_idx

    aucs = []
    for col, t_idx in entries:
        score_col = scores_v5[train_idx, col]
        auc       = roc_auc_score(y_bin, score_col)
        aucs.append((col, t_idx, auc))

    best_col, best_t_idx, best_auc = max(aucs, key=lambda x: x[2])
    v6c_text_rows[c_idx]  = best_col   # row in text_emb_v5 = column in scores_v5
    v6c_score_cols[c_idx] = best_col

    row = {
        "concept_id":        cid,
        "designed_target":   target_cls,
        "t0_auc":            round(aucs[0][2], 4),
        "t1_auc":            round(aucs[1][2], 4),
        "t2_auc":            round(aucs[2][2], 4),
        "selected_template": best_t_idx,
        "selected_prompt":   str(prompts_v5[best_col]),
        "selected_auc":      round(best_auc, 4),
    }
    selection_rows.append(row)

sel_df = pd.DataFrame(selection_rows).sort_values("concept_id").reset_index(drop=True)

# ── Step 5: Print + save selection table ──────────────────────────────────────
print("\nTemplate selection results (sorted by concept_id):")
print(sel_df[["concept_id", "designed_target", "t0_auc", "t1_auc",
              "t2_auc", "selected_template", "selected_auc"]].to_string(index=False))

SELECTION_CSV.parent.mkdir(parents=True, exist_ok=True)
sel_df.to_csv(SELECTION_CSV, index=False)
print(f"\nSelection saved -> {SELECTION_CSV.relative_to(_ROOT)}")

# Template win summary
from collections import Counter
wins = Counter(int(r["selected_template"]) for r in selection_rows)
print("\nTemplate wins:")
for t in sorted(wins):
    print(f"  t{t}: {wins[t]} concepts")

# ── Step 6: Build v6c text embedding matrix ───────────────────────────────────
print("\nBuilding v6c text embeddings (24, 512)...")
text_emb_v5 = np.load(TEXT_EMB_V5)    # (72, 512)
assert text_emb_v5.shape == (72, 512)

text_emb_v6c = text_emb_v5[v6c_text_rows, :]   # (24, 512) — rows are already L2-norm'd
assert text_emb_v6c.shape == (24, 512)

# Verify: selected rows are unit vectors (v5 embeddings are L2-normalised)
norms = np.linalg.norm(text_emb_v6c, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5), \
    f"L2 norms not all 1.0: min={norms.min():.6f} max={norms.max():.6f}"

np.save(TEXT_EMB_V6C, text_emb_v6c.astype(np.float32))
print(f"  Saved -> {TEXT_EMB_V6C.relative_to(_ROOT)}")

# ── Step 7: Build v6c score matrix ────────────────────────────────────────────
print("Loading cached image embeddings...")
image_emb = np.load(IMAGE_EMBEDDINGS)   # (10015, 512)
assert image_emb.shape == (10015, 512)

print("Computing similarity matrix (10015, 24)...")
scores_v6c = (image_emb @ text_emb_v6c.T).astype(np.float32)
assert scores_v6c.shape == (10015, 24)

s_min = float(scores_v6c.min())
s_max = float(scores_v6c.max())
s_mean = float(scores_v6c.mean())
assert -1.0 <= s_min and s_max <= 1.0, \
    f"Scores outside [-1, 1]: min={s_min:.4f} max={s_max:.4f}"
print(f"  Scores: min={s_min:.4f}  max={s_max:.4f}  mean={s_mean:.4f}")

# ── Step 8: Save v6c npz ──────────────────────────────────────────────────────
selected_prompts = np.array([str(prompts_v5[int(v6c_text_rows[i])])
                              for i in range(24)])
selected_t_idx   = np.array([int(sel_df.loc[sel_df["concept_id"] == cid,
                                             "selected_template"].values[0])
                              for cid in concept_ids], dtype=np.int32)

print(f"Saving -> {SCORES_V6C_NPZ.relative_to(_ROOT)}")
np.savez(
    SCORES_V6C_NPZ,
    scores                = scores_v6c,
    image_ids             = v5["image_ids"],
    labels                = v5["labels"],
    lesion_ids            = v5["lesion_ids"],
    concept_ids           = np.array(concept_ids),
    tiers                 = tiers,
    selected_template_idx = selected_t_idx,
    selected_prompts      = selected_prompts,
    source                = np.array("v5_best_template_by_intended_auc"),
)

# ── Summary ───────────────────────────────────────────────────────────────────
labels_all = v5["labels"]
unique, counts = np.unique(labels_all, return_counts=True)
print("\nClass distribution:")
for cls, cnt in sorted(zip(unique.tolist(), counts.tolist())):
    print(f"  {cls}: {cnt} ({cnt / len(labels_all) * 100:.1f}%)")

print(f"\nTemplate wins: " +
      "  ".join(f"t{t}: {wins[t]} concepts" for t in sorted(wins)))
print(f"Scores shape  : {scores_v6c.shape}  dtype: float32")
print(f"  -> {SCORES_V6C_NPZ.relative_to(_ROOT)}")
