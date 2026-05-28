"""
Diagnostic: compare v5/72-feature NAM vs v6/24-feature NAM training dynamics.

Determines whether v6's underperformance (test bal_acc 0.498 vs 0.555 LR)
is regularization-induced (fixable by sweep) or structural (val→test gap
exists regardless of regularization).

No outputs written to disk — console only.

Run from project root:
    python scripts/diagnose_v5_v6_gap.py
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
V5_DIR = _ROOT / "reports/nam/base"
V6_DIR = _ROOT / "reports/nam/v6_base"
SEEDS  = [42, 43, 44, 45, 46]

UNDERFITTING_TRAIN_LOSS_THRESHOLD = 0.10   # v6 train loss - v5 train loss; flagged if above
STRUCTURAL_GAP_THRESHOLD          = 0.05   # val→test gap flagged if above


def load_run(run_dir: pathlib.Path, seeds: list[int]) -> dict:
    """Load logs + aggregated metrics for one run. Returns per-seed stats."""
    agg = pd.read_csv(run_dir / "aggregated_metrics.csv")
    agg_seeds = agg[~agg["seed"].isin(["mean", "std"])].copy()
    agg_seeds["seed"] = agg_seeds["seed"].astype(int)

    records = []
    for seed in seeds:
        log = pd.read_csv(run_dir / f"seed_{seed}" / "training_log.csv")
        best_idx       = log["val_balanced_acc"].idxmax()
        best_val_acc   = float(log.loc[best_idx, "val_balanced_acc"])
        best_epoch     = int(log.loc[best_idx, "epoch"])
        train_loss_at_best = float(log.loc[best_idx, "train_loss"])
        total_epochs   = len(log)

        test_acc = float(
            agg_seeds.loc[agg_seeds["seed"] == seed, "balanced_accuracy"].values[0]
        )
        gap = best_val_acc - test_acc

        records.append({
            "seed":              seed,
            "best_val_acc":      best_val_acc,
            "best_epoch":        best_epoch,
            "total_epochs":      total_epochs,
            "train_loss_at_best": train_loss_at_best,
            "test_acc":          test_acc,
            "val_test_gap":      gap,
        })

    return {
        "records": records,
        "df":      pd.DataFrame(records),
    }


v5 = load_run(V5_DIR, SEEDS)
v6 = load_run(V6_DIR, SEEDS)

v5df = v5["df"]
v6df = v6["df"]

# ── Summary stats ─────────────────────────────────────────────────────────────
def _fmt(col: str, df: pd.DataFrame) -> str:
    return f"{df[col].mean():.4f} +/- {df[col].std():.4f}"


print("\n" + "=" * 90)
print("  v5 NAM (72 features) vs v6 NAM (24 features, regularized)  —  diagnostic")
print("=" * 90)
print(f"{'Metric':<40s}  {'v5 NAM (72)':>22}  {'v6 NAM (24, current)':>22}")
print("-" * 90)
print(f"{'Best val balanced accuracy':<40s}  {_fmt('best_val_acc', v5df):>22}  {_fmt('best_val_acc', v6df):>22}")
print(f"{'Test balanced accuracy':<40s}  {_fmt('test_acc', v5df):>22}  {_fmt('test_acc', v6df):>22}")
print(f"{'Val -> test gap (mean)':<40s}  {v5df['val_test_gap'].mean():>22.4f}  {v6df['val_test_gap'].mean():>22.4f}")
print(f"{'Train loss at best-val epoch (mean)':<40s}  {v5df['train_loss_at_best'].mean():>22.4f}  {v6df['train_loss_at_best'].mean():>22.4f}")
print(f"{'Best-val epoch (mean)':<40s}  {v5df['best_epoch'].mean():>22.1f}  {v6df['best_epoch'].mean():>22.1f}")
print(f"{'Total epochs trained (mean)':<40s}  {v5df['total_epochs'].mean():>22.1f}  {v6df['total_epochs'].mean():>22.1f}")
print("=" * 90)

print("\nPer-seed detail:")
print(f"{'':6}  {'best_val':>10}  {'test':>10}  {'gap':>10}  {'train_loss_@best':>18}  {'best_ep':>8}")
for _, r in v5df.iterrows():
    print(f"  v5 s{int(r.seed)}  {r.best_val_acc:10.4f}  {r.test_acc:10.4f}  "
          f"{r.val_test_gap:+10.4f}  {r.train_loss_at_best:18.4f}  {int(r.best_epoch):8d}")
print()
for _, r in v6df.iterrows():
    print(f"  v6 s{int(r.seed)}  {r.best_val_acc:10.4f}  {r.test_acc:10.4f}  "
          f"{r.val_test_gap:+10.4f}  {r.train_loss_at_best:18.4f}  {int(r.best_epoch):8d}")

# ── Interpretation ─────────────────────────────────────────────────────────────
v5_gap_mean = v5df["val_test_gap"].mean()
v6_gap_mean = v6df["val_test_gap"].mean()
train_loss_delta = v6df["train_loss_at_best"].mean() - v5df["train_loss_at_best"].mean()

both_structural = (v5_gap_mean > STRUCTURAL_GAP_THRESHOLD and
                   v6_gap_mean > STRUCTURAL_GAP_THRESHOLD)
only_v6_gap     = (v6_gap_mean > STRUCTURAL_GAP_THRESHOLD and
                   v5_gap_mean <= STRUCTURAL_GAP_THRESHOLD)
underfitting    = train_loss_delta > UNDERFITTING_TRAIN_LOSS_THRESHOLD

print("\n" + "=" * 90)
print("  INTERPRETATION")
print("=" * 90)

if both_structural:
    print(f"[STRUCTURAL GAP]  Val->test gap > {STRUCTURAL_GAP_THRESHOLD} in BOTH runs "
          f"(v5={v5_gap_mean:.3f}, v6={v6_gap_mean:.3f}).")
    print("  The gap is not purely a regularization artefact introduced in v6.")
    print("  A hyperparameter sweep can address the underfitting component")
    print("  but will likely not fully close the val->test gap to zero.")
    print("  Expect test performance to improve with better capacity, but a")
    print("  residual gap vs the val-set peak will remain.")
elif only_v6_gap:
    print(f"[REGULARIZATION ISSUE]  v6 gap ({v6_gap_mean:.3f}) > {STRUCTURAL_GAP_THRESHOLD} "
          f"but v5 gap ({v5_gap_mean:.3f}) is at or below threshold.")
    print("  Underfitting is the primary cause. Sweep likely to fix fully.")
else:
    print(f"[BORDERLINE]  v5 gap={v5_gap_mean:.3f}, v6 gap={v6_gap_mean:.3f}.")

if underfitting:
    print(f"\n[UNDERFITTING CONFIRMED]  v6 train loss at best-val epoch is "
          f"{train_loss_delta:+.3f} higher than v5 ({v6df['train_loss_at_best'].mean():.3f} "
          f"vs {v5df['train_loss_at_best'].mean():.3f}).")
    print("  The [32,16] + dropout=0.25 + wd=1e-4 stack is too aggressive.")
    print("  Sweeping over capacity, dropout, and weight_decay is warranted.")
    print("  Recommended sweep focus: wider subnets ([32,32] or [64,32])")
    print("  and lower dropout (0.10 or 0.20). Weight decay is secondary.")

print(f"\n[NEXT STEP]  {'Structural gap is present — discuss before sweeping if test improvement target is > 0.555 LR baseline.' if both_structural and not underfitting else 'Underfitting is the primary signal. Proceed with sweep_nam_v6.py.'}")
print("=" * 90 + "\n")
