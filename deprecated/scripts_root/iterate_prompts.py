"""Prompt iteration: compare v1 vs v4 prompts on the paired-contrast diagnostic.

Re-encodes only prompts whose text changed; image embeddings are never touched.

Outputs:
    data/features/ham10000_text_embeddings_v1.npy  (computed once, cached)
    data/features/ham10000_text_embeddings_v4.npy
    data/features/ham10000_concept_scores_v4.npz
    reports/prompt_analysis/v1_vs_v4_paired_contrasts.csv

Run from project root:
    python scripts/iterate_prompts.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.analysis.prompt_quality import analysis5_paired_contrasts
from src.features.clip_loader import load_clip
from src.features.encode_text import encode_prompts
from src.features.prompt_loader import load_prompts

# ---------------------------------------------------------------------------
# Paths (anchored to repo root)
# ---------------------------------------------------------------------------
PROMPTS_V1       = _ROOT / "src/features/prompts/ham10000_prompts.txt"
PROMPTS_V4       = _ROOT / "src/features/prompts/ham10000_prompts_v4.txt"
IMAGE_EMBEDDINGS = _ROOT / "data/features/ham10000_image_embeddings.npy"
SCORES_V1_NPZ    = _ROOT / "data/features/ham10000_concept_scores.npz"
TEXT_EMB_V1      = _ROOT / "data/features/ham10000_text_embeddings_v1.npy"
TEXT_EMB_V4      = _ROOT / "data/features/ham10000_text_embeddings_v4.npy"
SCORES_V4_NPZ    = _ROOT / "data/features/ham10000_concept_scores_v4.npz"
SPLIT_NPZ        = _ROOT / "data/splits/train_test_lesion_split.npz"
REPORT_DIR       = _ROOT / "reports/prompt_analysis"
COMPARISON_CSV   = REPORT_DIR / "v1_vs_v4_paired_contrasts.csv"
# ---------------------------------------------------------------------------


def _require_paths() -> None:
    missing = [
        p for p in (
            PROMPTS_V1,
            PROMPTS_V4,
            IMAGE_EMBEDDINGS,
            SCORES_V1_NPZ,
            SPLIT_NPZ,
        )
        if not p.is_file()
    ]
    if missing:
        lines = "\n  ".join(str(p.relative_to(_ROOT)) for p in missing)
        raise FileNotFoundError(
            "Missing required inputs:\n  " + lines + "\n\n"
            "Run scripts/extract_features.py and scripts/analyze_prompts.py first."
        )


def _build_meta(p: dict) -> pd.DataFrame:
    """Build the 72-row prompt metadata DataFrame expected by analysis functions."""
    tmpl_name = {0: "t1", 1: "t2", 2: "t3"}
    return pd.DataFrame({
        "prompt_idx":  np.arange(len(p["prompts"])),
        "concept_id":  [p["concept_ids"][i] for i in p["prompt_concept_idx"]],
        "template":    [tmpl_name[i]        for i in p["prompt_template_idx"]],
        "prompt":      p["prompts"],
        "concept_idx": list(p["prompt_concept_idx"]),
        "tier":        [p["tiers"][i]       for i in p["prompt_concept_idx"]],
    })


def _v1_row_lookup(p1: dict) -> dict[tuple[str, int], int]:
    """Map (concept_id, template_idx) -> row index in the v1 embedding matrix."""
    lookup: dict[tuple[str, int], int] = {}
    for i in range(len(p1["prompts"])):
        cid  = p1["concept_ids"][p1["prompt_concept_idx"][i]]
        tmpl = p1["prompt_template_idx"][i]
        lookup[(cid, tmpl)] = i
    return lookup


def find_changed_indices(p1: dict, p4: dict) -> list[int]:
    """Return row indices in p4 whose prompt text differs from the matching v1 row."""
    v1_text: dict[tuple[str, int], str] = {}
    for i in range(len(p1["prompts"])):
        cid  = p1["concept_ids"][p1["prompt_concept_idx"][i]]
        tmpl = p1["prompt_template_idx"][i]
        v1_text[(cid, tmpl)] = p1["prompts"][i]

    changed = []
    for i in range(len(p4["prompts"])):
        cid  = p4["concept_ids"][p4["prompt_concept_idx"][i]]
        tmpl = p4["prompt_template_idx"][i]
        if v1_text.get((cid, tmpl)) != p4["prompts"][i]:
            changed.append(i)
    return changed


def main() -> None:
    _require_paths()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load prompt files; identify changed rows
    # ------------------------------------------------------------------
    p1 = load_prompts(PROMPTS_V1)
    p4 = load_prompts(PROMPTS_V4)

    changed_idx = find_changed_indices(p1, p4)

    if not changed_idx:
        print("No prompt changes detected between v1 and v4. Nothing to do.")
        return

    changed_set = set(changed_idx)
    changed_labels = [
        f"{p4['concept_ids'][p4['prompt_concept_idx'][i]]} "
        f"t{p4['prompt_template_idx'][i] + 1}"
        for i in changed_idx
    ]
    changed_texts = [p4["prompts"][i] for i in changed_idx]

    print(f"Changed prompts ({len(changed_idx)}):")
    for label, text in zip(changed_labels, changed_texts):
        print(f"  [{label}]  \"{text}\"")

    # ------------------------------------------------------------------
    # 2. Load CLIP
    # ------------------------------------------------------------------
    model, _, tokenizer, device = load_clip()
    print(f"CLIP loaded on {device}")

    # ------------------------------------------------------------------
    # 3 + 4. Get or compute v1 text embeddings (cached after first run)
    # ------------------------------------------------------------------
    if TEXT_EMB_V1.exists():
        text_emb_v1 = np.load(TEXT_EMB_V1)
        print(f"Loaded cached v1 text embeddings from {TEXT_EMB_V1.relative_to(_ROOT)}")
    else:
        print("Computing v1 text embeddings (one-time)...")
        text_emb_v1 = encode_prompts(model, tokenizer, p1["prompts"], device).numpy()
        np.save(TEXT_EMB_V1, text_emb_v1)
        print(f"Saved -> {TEXT_EMB_V1.relative_to(_ROOT)}")

    # ------------------------------------------------------------------
    # 5. Build v4 text embedding matrix
    # ------------------------------------------------------------------
    v1_rows = _v1_row_lookup(p1)
    text_emb_v4 = np.empty_like(text_emb_v1)

    for i in range(len(p4["prompts"])):
        if i in changed_set:
            continue
        cid  = p4["concept_ids"][p4["prompt_concept_idx"][i]]
        tmpl = p4["prompt_template_idx"][i]
        text_emb_v4[i] = text_emb_v1[v1_rows[(cid, tmpl)]]

    print(f"Encoding {len(changed_idx)} changed prompt(s)...")
    new_embs = encode_prompts(model, tokenizer, changed_texts, device).numpy()
    for out_row, emb in zip(changed_idx, new_embs):
        text_emb_v4[out_row] = emb

    np.save(TEXT_EMB_V4, text_emb_v4)
    print(f"Saved v4 text embeddings -> {TEXT_EMB_V4.relative_to(_ROOT)}")

    # ------------------------------------------------------------------
    # 6. Recompute similarity matrix
    # ------------------------------------------------------------------
    image_emb = np.load(IMAGE_EMBEDDINGS)
    scores_v4 = (image_emb @ text_emb_v4.T).astype(np.float32)

    v1_data = np.load(SCORES_V1_NPZ, allow_pickle=True)
    labels = v1_data["labels"]
    lesion_ids = v1_data["lesion_ids"]
    image_ids = v1_data["image_ids"]

    np.savez_compressed(
        SCORES_V4_NPZ,
        scores=scores_v4,
        image_ids=image_ids,
        labels=labels,
        lesion_ids=lesion_ids,
        concept_ids=np.array(p4["concept_ids"]),
        prompts=np.array(p4["prompts"]),
        prompt_concept_idx=np.array(p4["prompt_concept_idx"], dtype=np.int32),
        prompt_template_idx=np.array(p4["prompt_template_idx"], dtype=np.int32),
        tiers=np.array(p4["tiers"], dtype=np.int32),
    )
    print(f"Saved v4 scores -> {SCORES_V4_NPZ.relative_to(_ROOT)}")

    # ------------------------------------------------------------------
    # 7. Paired-contrast comparison on training split
    # ------------------------------------------------------------------
    train_idx = np.load(SPLIT_NPZ)["train_idx"]

    meta_v1 = _build_meta(p1)
    meta_v4 = _build_meta(p4)

    df_v1 = analysis5_paired_contrasts(
        v1_data["scores"][train_idx], labels[train_idx], meta_v1
    )
    df_v4 = analysis5_paired_contrasts(
        scores_v4[train_idx], labels[train_idx], meta_v4
    )

    key_cols = ["positive_concept", "negative_concept", "target_class"]
    merged = df_v1[key_cols + ["diff", "status"]].merge(
        df_v4[key_cols + ["diff", "status"]],
        on=key_cols,
        suffixes=("_v1", "_v4"),
    )
    merged["delta"] = (merged["diff_v4"] - merged["diff_v1"]).round(5)

    col_w = 52
    header = (
        f"{'Contrast':<{col_w}}"
        f"{'v1 diff':>10}  {'v1':>6}  {'v4 diff':>10}  {'v4':>6}  {'d':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for _, r in merged.iterrows():
        contrast = f"{r['positive_concept']} > {r['negative_concept']} on {r['target_class']}"
        print(
            f"{contrast:<{col_w}}"
            f"{r['diff_v1']:>+10.4f}  {r['status_v1']:>6}  "
            f"{r['diff_v4']:>+10.4f}  {r['status_v4']:>6}  "
            f"{r['delta']:>+10.4f}"
        )

    merged.to_csv(COMPARISON_CSV, index=False)
    print(f"\nComparison table -> {COMPARISON_CSV.relative_to(_ROOT)}")

    n_pass_v1 = int((df_v1["status"] == "PASS").sum())
    n_pass_v4 = int((df_v4["status"] == "PASS").sum())
    n_total = len(df_v1)
    print(f"\nv1: {n_pass_v1}/{n_total} passed, v4: {n_pass_v4}/{n_total} passed.")


if __name__ == "__main__":
    main()
