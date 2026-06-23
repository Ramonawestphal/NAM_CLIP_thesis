#!/usr/bin/env python3
"""
STEP 6b — ANEC (Accuracy at Number of Effective Concepts) evaluation.

For each condition (sparsity_only, sparsity_concurvity) and each seed (42-46):
  For each K in [5, 8, 10, 15, 20]:
    1. Find the step where n_active is as close to K as possible from above
       (first step where n_active ≤ K on the warm-start path).
    2. Load the corresponding checkpoint.
    3. Evaluate on the held-out test set (1995 samples).
    4. Compute val R_perp at the checkpoint.

Outputs:
  results/HAM10000/anec_evaluation/by_seed.csv
  results/HAM10000/anec_evaluation/aggregated.csv
  results/HAM10000/anec_evaluation/summary_table.md
  results/HAM10000/anec_evaluation/rule_a_secondary.md
  results/HAM10000/anec_evaluation/methodology.md

Sanity checks:
  - Dense (K=24) checkpoint test_balacc should match STEP 2 (plain_nam ~0.530) and
    STEP 4 (concurvity_only ~0.516) within 0.015.
  - test_balacc should generally decrease as K decreases.
  - No NaN in test_balacc or test_auc_weighted.

References (ANEC framework):
  Hyperbolic CBM (2026); SCOM (2023) cites Miller (1956) and Cowan (2001).
  K anchor: 10, within Miller's 7±2 working memory range.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.v7._common import (
    FEATURES_PATH, SPLITS_PATH, N_FEATURES, N_CLASSES,
    load_raw_data, make_fixed_val_split, standardize, make_model,
)
from src.models.concurvity import multiclass_concurvity

# ── Configuration ──────────────────────────────────────────────────────────────
CONDITIONS  = ["sparsity_only", "sparsity_concurvity"]
SEEDS       = [42, 43, 44, 45, 46]
K_BUDGETS   = [5, 8, 10, 15, 20]

SWEEP_BASE   = "results/HAM10000/sparsity_sweep"
WINNER_JSON  = "results/HAM10000/architecture_search_cv/winner.json"
OUT_DIR      = "results/HAM10000/anec_evaluation"
DEVICE       = torch.device("cpu")

# Reference values from STEP 2 / STEP 4 (pre-computed, used in summary table)
REF_PLAIN_NAM = {
    "mean_test_balacc": 0.5296, "std_test_balacc": 0.0094,
    "mean_test_auc_weighted": 0.8531, "std_test_auc_weighted": 0.0026,
    "mean_val_r_perp": 0.3190, "std_val_r_perp": 0.0071,
}
REF_CONCURVITY_ONLY = {
    "mean_test_balacc": 0.5162, "std_test_balacc": 0.0097,
    "mean_test_auc_weighted": 0.8281, "std_test_auc_weighted": 0.0027,
    "mean_val_r_perp": 0.1085, "std_val_r_perp": 0.0187,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_k_row(df: pd.DataFrame, K: int):
    """Return the earliest-step row where n_active ≤ K (closest from above).

    If n_active == K exists, return the first such row.
    Otherwise return the first row where n_active < K (best achievable ≤ K).
    Returns None if the path never drops to K or below.
    """
    # Prefer exact match
    exact = df[df["n_active"] == K]
    if len(exact) > 0:
        return exact.iloc[0]
    # Otherwise first step where n_active < K
    below = df[df["n_active"] < K]
    if len(below) > 0:
        return below.iloc[0]
    return None


def load_path_checkpoint(condition: str, seed: int, lambda_val: float,
                          hidden_dims, dropout, concept_names) -> torch.nn.Module:
    """Load a sparsity-path checkpoint by lambda value."""
    lam_tag  = f"{lambda_val:.6e}"
    ckpt_pt  = os.path.join(
        SWEEP_BASE, condition, f"seed_{seed}", "checkpoints",
        f"seed{seed}_lambda{lam_tag}.pt",
    )
    if not os.path.exists(ckpt_pt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_pt}")
    model = make_model(hidden_dims, dropout, concept_names, DEVICE)
    model.load_state_dict(torch.load(ckpt_pt, map_location=DEVICE))
    model.eval()
    return model


def load_dense_checkpoint(condition: str, seed: int, concurvity_lambda: float,
                           hidden_dims, dropout, concept_names) -> torch.nn.Module:
    """Load the dense (lambda_s=0) checkpoint for K=24 sanity check."""
    conc_tag = str(concurvity_lambda).replace(".", "p")
    ckpt_pt  = os.path.join(
        SWEEP_BASE, condition, f"seed_{seed}",
        f"dense_seed{seed}_conc{conc_tag}_ep100.pt",
    )
    if not os.path.exists(ckpt_pt):
        raise FileNotFoundError(f"Dense checkpoint not found: {ckpt_pt}")
    model = make_model(hidden_dims, dropout, concept_names, DEVICE)
    model.load_state_dict(torch.load(ckpt_pt, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def evaluate_test(model, X_test_sc: np.ndarray, y_test_enc: np.ndarray,
                  class_names: list) -> dict:
    """Compute test-set metrics from a loaded model."""
    X_t    = torch.tensor(X_test_sc, dtype=torch.float32)
    logits = model(X_t)
    probs  = F.softmax(logits, dim=1).numpy()
    preds  = probs.argmax(axis=1)

    balacc   = balanced_accuracy_score(y_test_enc, preds)
    top1     = accuracy_score(y_test_enc, preds)
    macro_f1 = f1_score(y_test_enc, preds, average="macro", zero_division=0)
    try:
        auc_w = roc_auc_score(
            y_test_enc, probs, multi_class="ovr", average="weighted"
        )
    except Exception:
        auc_w = float("nan")

    per_auc, per_f1 = {}, {}
    for c_idx, c_name in enumerate(class_names):
        y_bin = (y_test_enc == c_idx).astype(int)
        try:
            per_auc[c_name] = roc_auc_score(y_bin, probs[:, c_idx])
        except Exception:
            per_auc[c_name] = float("nan")
        per_f1[c_name] = f1_score(
            y_bin, (preds == c_idx).astype(int), zero_division=0
        )

    return {
        "test_balacc":        balacc,
        "test_top1_acc":      top1,
        "test_macro_f1":      macro_f1,
        "test_auc_weighted":  auc_w,
        **{f"test_auc_{c}": v for c, v in per_auc.items()},
        **{f"test_f1_{c}":  v for c, v in per_f1.items()},
    }


@torch.no_grad()
def compute_val_r_perp(model, X_val_sc: np.ndarray) -> float:
    """Compute R_perp on validation data from the loaded model."""
    model.eval()
    X_t = torch.tensor(X_val_sc, dtype=torch.float32)
    _, shape_outs = model(X_t, return_shape_outputs=True)
    return multiclass_concurvity(shape_outs).item()


def apply_rule_a(df: pd.DataFrame, dense_val_balacc: float) -> dict | None:
    """Find the last step where val_balacc >= dense_val_balacc - 0.02."""
    threshold = dense_val_balacc - 0.02
    above = df[df["val_balacc"] >= threshold]
    if len(above) == 0:
        return None
    row = above.iloc[-1]
    return {
        "step":         int(row["step"]),
        "lambda_s":     float(row["lambda_s"]),
        "n_active":     int(row["n_active"]),
        "val_balacc":   float(row["val_balacc"]),
        "threshold":    threshold,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Architecture
    with open(WINNER_JSON) as f:
        winner = json.load(f)
    hidden_dims   = tuple(winner["hidden_dims"])  # (64, 32)
    dropout       = winner["dropout"]              # 0.1

    # Raw data
    raw           = load_raw_data()
    concept_names = raw["concept_names"]
    class_names   = raw["class_names"]

    # Test data (raw — scaled per seed below)
    test_idx      = raw["test_idx"]
    X_test_raw    = raw["scores"][test_idx].astype(np.float32)
    y_test_str    = raw["labels"][test_idx]
    class_to_idx  = {c: i for i, c in enumerate(class_names)}
    y_test_enc    = np.array([class_to_idx[c] for c in y_test_str], dtype=np.int64)

    # Val data (fixed val_random_state=42, same for all seeds in corrected run)
    train_idx         = raw["train_idx"]
    X_all_train       = raw["scores"][train_idx].astype(np.float32)
    y_all_train       = raw["labels"][train_idx]
    lesion_ids_train  = raw["lesion_ids"][train_idx]
    val_split_ref     = make_fixed_val_split(
        X_all_train, y_all_train, lesion_ids_train, class_names,
        val_random_state=42,
    )
    X_val_raw         = val_split_ref["X_val"]  # scaled per seed below

    # Concurvity lambdas per condition
    conc_lam = {"sparsity_only": 0.0, "sparsity_concurvity": 3.0}

    # ── K=24 dense sanity check ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("K=24 DENSE SANITY CHECK (vs STEP 2 / STEP 4)")
    print("="*60)
    dense_sanity = {}
    for condition in CONDITIONS:
        balaccs = []
        for seed in SEEDS:
            seed_dir    = os.path.join(SWEEP_BASE, condition, f"seed_{seed}")
            scaler_path = os.path.join(seed_dir, "scaler.pkl")
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X_test_sc = scaler.transform(X_test_raw).astype(np.float32)
            try:
                model = load_dense_checkpoint(
                    condition, seed, conc_lam[condition],
                    hidden_dims, dropout, concept_names,
                )
                m = evaluate_test(model, X_test_sc, y_test_enc, class_names)
                balaccs.append(m["test_balacc"])
                print(f"  {condition}  seed {seed}: test_balacc={m['test_balacc']:.4f}")
            except FileNotFoundError as e:
                print(f"  {condition}  seed {seed}: MISSING — {e}")
        if balaccs:
            mean_ba = np.mean(balaccs)
            dense_sanity[condition] = mean_ba
            expected = 0.530 if condition == "sparsity_only" else 0.516
            flag = "✅" if abs(mean_ba - expected) < 0.015 else "⚠️  MISMATCH"
            print(f"  {condition}  mean={mean_ba:.4f}  expected≈{expected}  {flag}")

    # ── ANEC evaluation at K∈{5,8,10,15,20} ───────────────────────────────────
    print("\n" + "="*60)
    print("ANEC EVALUATION")
    print("="*60)

    by_seed_rows  = []
    rule_a_rows   = []

    for condition in CONDITIONS:
        cond_dir = os.path.join(SWEEP_BASE, condition)
        print(f"\n{'─'*60}")
        print(f"Condition: {condition}")
        print(f"{'─'*60}")

        for seed in SEEDS:
            seed_dir    = os.path.join(cond_dir, f"seed_{seed}")
            scaler_path = os.path.join(seed_dir, "scaler.pkl")
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X_test_sc = scaler.transform(X_test_raw).astype(np.float32)
            X_val_sc  = scaler.transform(X_val_raw).astype(np.float32)

            csv_path = os.path.join(cond_dir, f"path_seed{seed}.csv")
            df = pd.read_csv(csv_path, comment="#")
            header_line = open(csv_path, encoding="utf-8").readline()
            dense_val_balacc = float(
                header_line.split("dense_val_balacc=")[1].strip()
            )

            print(f"\n  Seed {seed}  (dense_val_balacc={dense_val_balacc:.4f})")

            # Rule A (corrected run)
            ra = apply_rule_a(df, dense_val_balacc)
            if ra:
                rule_a_rows.append({
                    "run": "corrected", "condition": condition, "seed": seed,
                    **ra,
                })
                print(f"    Rule A: step={ra['step']}, lambda={ra['lambda_s']:.3f}, "
                      f"n_active={ra['n_active']}, val_balacc={ra['val_balacc']:.4f}")
            else:
                print(f"    Rule A: NO QUALIFYING STEP")

            for K in K_BUDGETS:
                row_info = find_k_row(df, K)
                if row_info is None:
                    print(f"    K={K:2d}: NOT REACHED")
                    by_seed_rows.append({
                        "condition": condition, "seed": seed, "target_K": K,
                        "achieved_n_active": None, "step": None,
                        "lambda": None, "val_balacc": None, "val_auc": None,
                        "val_r_perp": None,
                        "test_balacc": None, "test_macro_f1": None,
                        "test_auc_weighted": None, "test_top1_acc": None,
                        **{f"test_auc_{c}": None for c in class_names},
                        **{f"test_f1_{c}":  None for c in class_names},
                    })
                    continue

                step       = int(row_info["step"])
                lambda_val = float(row_info["lambda_s"])
                n_active   = int(row_info["n_active"])
                val_balacc_here = float(row_info["val_balacc"])
                val_auc_here    = float(row_info["val_auc"])
                exact_flag = "exact" if n_active == K else f"achieved={n_active}"

                try:
                    model = load_path_checkpoint(
                        condition, seed, lambda_val,
                        hidden_dims, dropout, concept_names,
                    )
                except FileNotFoundError as e:
                    print(f"    K={K:2d}: CHECKPOINT MISSING — {e}")
                    continue

                r_perp = compute_val_r_perp(model, X_val_sc)
                test_m = evaluate_test(model, X_test_sc, y_test_enc, class_names)

                print(f"    K={K:2d} ({exact_flag}): step={step}, "
                      f"lambda={lambda_val:.3f}, "
                      f"test_balacc={test_m['test_balacc']:.4f}, "
                      f"test_auc={test_m['test_auc_weighted']:.4f}, "
                      f"r_perp={r_perp:.4f}")

                by_seed_rows.append({
                    "condition":         condition,
                    "seed":              seed,
                    "target_K":          K,
                    "achieved_n_active": n_active,
                    "step":              step,
                    "lambda":            lambda_val,
                    "val_balacc":        val_balacc_here,
                    "val_auc":           val_auc_here,
                    "val_r_perp":        r_perp,
                    **test_m,
                })

    # ── Write outputs ──────────────────────────────────────────────────────────
    by_seed_df = pd.DataFrame(by_seed_rows)
    by_seed_path = os.path.join(OUT_DIR, "by_seed.csv")
    by_seed_df.to_csv(by_seed_path, index=False)
    print(f"\nWritten: {by_seed_path}")

    # Aggregate
    agg_rows = []
    for condition in CONDITIONS:
        for K in K_BUDGETS:
            sub = by_seed_df[
                (by_seed_df["condition"] == condition) &
                (by_seed_df["target_K"] == K) &
                (by_seed_df["test_balacc"].notna())
            ]
            if len(sub) == 0:
                continue
            ddof = 1 if len(sub) > 1 else 0
            agg_rows.append({
                "condition":              condition,
                "target_K":              K,
                "n_seeds_at_target_K":   len(sub),
                "mean_achieved_n_active": sub["achieved_n_active"].mean(),
                "std_achieved_n_active":  sub["achieved_n_active"].std(ddof=ddof),
                "mean_test_balacc":       sub["test_balacc"].mean(),
                "std_test_balacc":        sub["test_balacc"].std(ddof=ddof),
                "mean_test_macro_f1":     sub["test_macro_f1"].mean(),
                "std_test_macro_f1":      sub["test_macro_f1"].std(ddof=ddof),
                "mean_test_auc_weighted": sub["test_auc_weighted"].mean(),
                "std_test_auc_weighted":  sub["test_auc_weighted"].std(ddof=ddof),
                "mean_test_top1_acc":     sub["test_top1_acc"].mean(),
                "std_test_top1_acc":      sub["test_top1_acc"].std(ddof=ddof),
                "mean_val_r_perp":        sub["val_r_perp"].mean(),
                "std_val_r_perp":         sub["val_r_perp"].std(ddof=ddof),
            })

    agg_df = pd.DataFrame(agg_rows)
    agg_path = os.path.join(OUT_DIR, "aggregated.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Written: {agg_path}")

    # Summary table
    write_summary_table(agg_df, OUT_DIR, dense_sanity, class_names)
    print(f"Written: {os.path.join(OUT_DIR, 'summary_table.md')}")

    # Rule A secondary
    write_rule_a_secondary(rule_a_rows, OUT_DIR)
    print(f"Written: {os.path.join(OUT_DIR, 'rule_a_secondary.md')}")

    # Methodology
    write_methodology(OUT_DIR)
    print(f"Written: {os.path.join(OUT_DIR, 'methodology.md')}")

    # ── Verification ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    # Check all seeds present
    for condition in CONDITIONS:
        for K in K_BUDGETS:
            sub = by_seed_df[
                (by_seed_df["condition"] == condition) &
                (by_seed_df["target_K"] == K)
            ]
            n_ok = sub["test_balacc"].notna().sum()
            if n_ok < len(SEEDS):
                print(f"  ⚠️  {condition} K={K}: only {n_ok}/{len(SEEDS)} seeds evaluated")
            else:
                print(f"  ✅ {condition} K={K}: all {len(SEEDS)} seeds evaluated")

    # K=24 sanity check
    for condition in CONDITIONS:
        if condition in dense_sanity:
            mean_ba  = dense_sanity[condition]
            expected = 0.530 if condition == "sparsity_only" else 0.516
            diff     = abs(mean_ba - expected)
            flag     = "✅" if diff < 0.015 else "❌ FAIL — investigate"
            print(f"  K=24 {condition}: mean_balacc={mean_ba:.4f}, "
                  f"expected≈{expected}, diff={diff:.4f}  {flag}")

    # Monotonicity check
    for condition in CONDITIONS:
        sub = agg_df[agg_df["condition"] == condition].sort_values("target_K", ascending=False)
        prev_acc = float("inf")
        for _, r in sub.iterrows():
            K   = int(r["target_K"])
            acc = r["mean_test_balacc"]
            if acc > prev_acc + 0.02:
                print(f"  ⚠️  {condition}: K={K} acc={acc:.4f} > previous "
                      f"{prev_acc:.4f} by >{0.02:.2f} (non-monotone)")
            prev_acc = acc

    print("\nDone.")


# ── Output writers ─────────────────────────────────────────────────────────────

def write_summary_table(agg_df: pd.DataFrame, out_dir: str,
                        dense_sanity: dict, class_names: list):
    tbl = {
        (r["condition"], int(r["target_K"])): r.to_dict()
        for _, r in agg_df.iterrows()
    }

    lines = [
        "# ANEC Evaluation — Summary Table",
        "",
        "Accuracy at Number of Effective Concepts (ANEC). "
        "Test set (n=1,995 samples). Mean ± std across 5 seeds (42–46).",
        "",
        "## Primary ANEC table",
        "",
        "| K  | sparsity_only (n_active) | sparsity_only bal_acc | sparsity+conc (n_active) | sparsity+conc bal_acc | Δ bal_acc | sparsity_only AUC | sparsity+conc AUC | sparsity_only R_perp | sparsity+conc R_perp |",
        "|----|--------------------------|----------------------|--------------------------|----------------------|-----------|-------------------|-------------------|----------------------|----------------------|",
    ]

    for K in [20, 15, 10, 8, 5]:
        so = tbl.get(("sparsity_only", K))
        sc = tbl.get(("sparsity_concurvity", K))

        def fmt_nacc(r):
            if r is None: return "—"
            return f"{r['mean_achieved_n_active']:.1f}±{r['std_achieved_n_active']:.1f}"

        def fmt_acc(r):
            if r is None: return "—"
            return f"{r['mean_test_balacc']:.3f}±{r['std_test_balacc']:.3f}"

        def fmt_auc(r):
            if r is None: return "—"
            return f"{r['mean_test_auc_weighted']:.3f}±{r['std_test_auc_weighted']:.3f}"

        def fmt_rperp(r):
            if r is None: return "—"
            return f"{r['mean_val_r_perp']:.3f}±{r['std_val_r_perp']:.3f}"

        if so and sc:
            delta = sc["mean_test_balacc"] - so["mean_test_balacc"]
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "—"

        lines.append(
            f"| {K:2d} | {fmt_nacc(so):>24} | {fmt_acc(so):>20} | "
            f"{fmt_nacc(sc):>24} | {fmt_acc(sc):>20} | {delta_str:>9} | "
            f"{fmt_auc(so):>17} | {fmt_auc(sc):>17} | "
            f"{fmt_rperp(so):>20} | {fmt_rperp(sc):>20} |"
        )

    # Reference rows
    lines += [
        "",
        "## Reference rows (K=24, dense)",
        "",
        "From STEP 2 (plain_nam) and STEP 4 (concurvity_only), 5 seeds each.",
        "",
        "| Condition | n_active | bal_acc (mean±std) | AUC (mean±std) | R_perp val (mean±std) |",
        "|-----------|----------|-------------------|----------------|----------------------|",
        f"| plain_nam (STEP 2)        | 24 | "
        f"{REF_PLAIN_NAM['mean_test_balacc']:.3f}±{REF_PLAIN_NAM['std_test_balacc']:.3f} | "
        f"{REF_PLAIN_NAM['mean_test_auc_weighted']:.3f}±{REF_PLAIN_NAM['std_test_auc_weighted']:.3f} | "
        f"{REF_PLAIN_NAM['mean_val_r_perp']:.3f}±{REF_PLAIN_NAM['std_val_r_perp']:.3f} |",
        f"| concurvity_only (STEP 4)  | 24 | "
        f"{REF_CONCURVITY_ONLY['mean_test_balacc']:.3f}±{REF_CONCURVITY_ONLY['std_test_balacc']:.3f} | "
        f"{REF_CONCURVITY_ONLY['mean_test_auc_weighted']:.3f}±{REF_CONCURVITY_ONLY['std_test_auc_weighted']:.3f} | "
        f"{REF_CONCURVITY_ONLY['mean_val_r_perp']:.3f}±{REF_CONCURVITY_ONLY['std_val_r_perp']:.3f} |",
    ]

    # K=24 sanity check results
    if dense_sanity:
        lines += [
            "",
            "## K=24 sanity check (dense checkpoints from corrected sparsity sweep)",
            "",
            "| Condition | Sweep dense mean_test_balacc | Expected (STEP 2/4) | Δ | Pass? |",
            "|-----------|------------------------------|---------------------|---|-------|",
        ]
        for condition, mean_ba in dense_sanity.items():
            expected = 0.530 if condition == "sparsity_only" else 0.516
            diff     = mean_ba - expected
            ok       = abs(diff) < 0.015
            lines.append(
                f"| {condition} | {mean_ba:.4f} | {expected:.3f} | "
                f"{diff:+.4f} | {'✅' if ok else '❌'} |"
            )

    lines += [
        "",
        "---",
        "Δ = sparsity+concurvity − sparsity_only (positive means concurvity helps).",
        "R_perp: mean absolute Pearson correlation of per-feature shape function outputs on validation set.",
        "K anchor: 10 (within Miller 1956 working memory range 7±2).",
        "",
        "*Full per-seed data: `by_seed.csv`. Aggregated: `aggregated.csv`.*",
        "*Rule A secondary: `rule_a_secondary.md`.*",
    ]

    with open(os.path.join(out_dir, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_rule_a_secondary(rule_a_rows: list, out_dir: str):
    lines = [
        "# Rule A Secondary Analysis",
        "",
        "Rule A: largest lambda step where val_balacc ≥ dense_val_balacc − 0.02.",
        "This identifies the maximum sparsity achievable at no accuracy cost.",
        "",
        "## Corrected run (val_random_state=42)",
        "",
        "### sparsity_only",
        "",
        "| Seed | Dense val_balacc | Threshold | Rule A step | Lambda | n_active | val_balacc |",
        "|------|-----------------|-----------|-------------|--------|----------|------------|",
    ]

    dep_so_rows = [
        # From deprecated run (val_random_state=seed): step/lambda/n_active from run_summary.md
        (42, 0.6130, 0.5930, 58,  9.352, 22, 0.5939),
        (43, 0.5483, 0.5283, 58,  9.352, 22, 0.5328),
        (44, 0.5704, 0.5504, 54,  7.994, 23, 0.5534),
        (45, 0.5603, 0.5403, 27,  2.772, 24, 0.5444),
        (46, 0.5624, 0.5424, 57,  8.992, 21, 0.5492),
    ]
    dep_sc_rows = [
        (42, 0.6009, 0.5809, 44,  5.400, 18, 0.5812),
        (43, 0.5106, 0.4906, 60, 10.115, 12, 0.4915),
        (44, 0.5220, 0.5020, 54,  7.994, 13, 0.5041),
        (45, 0.5397, 0.5197, 34,  3.648, 16, 0.5261),
        (46, 0.5637, 0.5437, 39,  4.439, 16, 0.5458),
    ]

    corr_so = [r for r in rule_a_rows
               if r["run"] == "corrected" and r["condition"] == "sparsity_only"]
    corr_sc = [r for r in rule_a_rows
               if r["run"] == "corrected" and r["condition"] == "sparsity_concurvity"]

    for r in sorted(corr_so, key=lambda x: x["seed"]):
        lines.append(
            f"| {r['seed']} | — | {r['threshold']:.4f} | {r['step']} | "
            f"{r['lambda_s']:.3f} | {r['n_active']} | {r['val_balacc']:.4f} |"
        )

    if corr_so:
        n_acts = [r["n_active"] for r in corr_so]
        lams   = [r["lambda_s"] for r in corr_so]
        lines += [
            "",
            f"Mean n_active at Rule A: **{np.mean(n_acts):.1f}** (range {min(n_acts)}–{max(n_acts)})",
            f"Median lambda: **{np.median(lams):.3f}**",
        ]

    lines += [
        "",
        "### sparsity_concurvity",
        "",
        "| Seed | Dense val_balacc | Threshold | Rule A step | Lambda | n_active | val_balacc |",
        "|------|-----------------|-----------|-------------|--------|----------|------------|",
    ]

    for r in sorted(corr_sc, key=lambda x: x["seed"]):
        lines.append(
            f"| {r['seed']} | — | {r['threshold']:.4f} | {r['step']} | "
            f"{r['lambda_s']:.3f} | {r['n_active']} | {r['val_balacc']:.4f} |"
        )

    if corr_sc:
        n_acts = [r["n_active"] for r in corr_sc]
        lams   = [r["lambda_s"] for r in corr_sc]
        lines += [
            "",
            f"Mean n_active at Rule A: **{np.mean(n_acts):.1f}** (range {min(n_acts)}–{max(n_acts)})",
            f"Median lambda: **{np.median(lams):.3f}**",
        ]

    lines += [
        "",
        "---",
        "",
        "## Comparison: deprecated run (val_random_state=seed) vs corrected (=42)",
        "",
        "### sparsity_only",
        "",
        "| Seed | Dep. dense | Dep. n_active | Dep. lambda | Corr. n_active | Corr. lambda |",
        "|------|-----------|--------------|-------------|----------------|--------------|",
    ]
    corr_so_by_seed = {r["seed"]: r for r in corr_so}
    for seed, dense_d, thresh_d, step_d, lam_d, nacc_d, bacc_d in dep_so_rows:
        corr = corr_so_by_seed.get(seed)
        corr_nacc = str(corr["n_active"]) if corr else "—"
        corr_lam  = f"{corr['lambda_s']:.3f}" if corr else "—"
        lines.append(
            f"| {seed} | {dense_d:.4f} | {nacc_d} | {lam_d:.3f} | "
            f"{corr_nacc} | {corr_lam} |"
        )

    lines += [
        "",
        "### sparsity_concurvity",
        "",
        "| Seed | Dep. dense | Dep. n_active | Dep. lambda | Corr. n_active | Corr. lambda |",
        "|------|-----------|--------------|-------------|----------------|--------------|",
    ]
    corr_sc_by_seed = {r["seed"]: r for r in corr_sc}
    for seed, dense_d, thresh_d, step_d, lam_d, nacc_d, bacc_d in dep_sc_rows:
        corr = corr_sc_by_seed.get(seed)
        corr_nacc = str(corr["n_active"]) if corr else "—"
        corr_lam  = f"{corr['lambda_s']:.3f}" if corr else "—"
        lines.append(
            f"| {seed} | {dense_d:.4f} | {nacc_d} | {lam_d:.3f} | "
            f"{corr_nacc} | {corr_lam} |"
        )

    lines += [
        "",
        "---",
        "",
        "The Rule A candidate provides a secondary anchor: the maximum sparsity",
        "achievable at no accuracy cost. The ANEC trajectory at K∈{5,8,10,15,20}",
        "characterises the accuracy-interpretability tradeoff at fixed budgets.",
        "Operating point selection for final model training (STEP 7) follows STEP 6c.",
    ]

    with open(os.path.join(out_dir, "rule_a_secondary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_methodology(out_dir: str):
    content = """\
# Methodology: Operating Point Selection Under the Interpretability Goal

## ANEC Framework

Standard regularization-path selection (e.g., the one-standard-error rule,
Hastie et al. 2017; or the cost-bounded "largest λ within ε of dense val"
rule used in SNAM-style work) selects operating points based on predictive
performance, treating interpretability as a free byproduct. In our setting,
the goal is explicitly interpretable models with a small number of active
shape functions that a clinician can simultaneously inspect. Recent work in
the concept bottleneck literature has converged on reporting Accuracy at
Number of Effective Concepts (ANEC; Hyperbolic CBM, 2026), where models are
evaluated at fixed interpretability budgets K. The choice of K is motivated
by working-memory constraints on simultaneous concept reasoning (Miller,
1956; Cowan, 2001; cited in the CBM context by SCOM, 2023).

We adopt the ANEC framework as our primary evaluation. For each condition
(sparsity_only, sparsity_concurvity), for each seed in {42, …, 46}, we
traverse the warm-start regularization path and identify the smallest λ_s
reaching n_active = K for K ∈ {5, 8, 10, 15, 20}. When a step drops past K
exactly (e.g., 16→12 in a single proximal step), we use the first step where
n_active ≤ K (the closest achievable from above). We evaluate the resulting
models on the held-out test set (n = 1,995, 20% of the HAM10000 cohort) and
report mean ± std across seeds. The anchor budget K = 10 was selected before
evaluation as falling within the Miller (1956) working memory range.

As a secondary analysis, we apply Rule A — the largest λ_s where val
balanced accuracy stays within 0.02 of the dense (λ_s = 0) baseline — to
identify the maximum sparsity achievable at no accuracy cost. This complements
the ANEC trajectory.

## References

- Hastie T, Tibshirani R, Friedman J (2017). *The Elements of Statistical Learning*
  (2nd ed.). Springer.
- Miller G (1956). The magical number seven, plus or minus two. *Psychological
  Review*, 63(2), 81–97.
- Cowan N (2001). The magical number 4 in short-term memory. *Behavioral and
  Brain Sciences*, 24(1), 87–114.
- SCOM (2023). Sparse Concept Bottleneck Models. [cite as appropriate].
- Hyperbolic CBM (2026). Hyperbolic Concept Bottleneck Models. [cite as appropriate].
"""
    with open(os.path.join(out_dir, "methodology.md"), "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()
