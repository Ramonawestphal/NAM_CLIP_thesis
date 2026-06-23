# DEPRECATED — Sparsity Sweep (val_random_state=seed, per-seed val split)

**Moved:** 2026-06-07  
**Reason:** Protocol drift from STEP 2 / STEP 4

## What this run was

The first full sparsity warm-start sweep, 5 seeds × 2 conditions × 150 steps,
with schedule lambda_0=1.0, epsilon=0.04. It completed successfully and all 10
runs produced clean monotone elimination paths.

## The bug

`make_fixed_val_split` was called with `val_random_state=seed`, giving each seed
a **different validation split** (seed 42 → 17% val drawn with rs=42, seed 43 →
rs=43, etc.).

STEP 2 (plain_nam) and STEP 4 (concurvity_only) both use `val_random_state=42`
fixed across all seeds. This makes the dense baselines and val_balacc trajectories
incomparable across the three steps:

| Seed | Deprecated dense_val_balacc | Expected (rs=42) |
|------|----------------------------|------------------|
| 42   | 0.6130                     | ~0.613 (same)    |
| 43   | 0.5483                     | ~0.601 (different split) |
| 44   | 0.5704                     | ~0.601           |
| 45   | 0.5603                     | ~0.601           |
| 46   | 0.5624                     | ~0.601           |

Seed 42 happens to match (seed=42=rs=42). All other seeds have lower apparent
dense val_balacc because they saw a harder val split.

## What replaced it

Re-run with `val_random_state=42` (fixed) in:
  `results/v7/sparsity_sweep/`

The corrected run uses identical schedule/architecture/seeds.

## Retained for

- Cross-checking elimination order against the corrected run
- Auditing that `border_irregularity` is universally first-eliminated (robust to val split)
- Raw reference in case the corrected run reveals unexpected differences
- Do NOT delete — this is part of the audit trail

## Key results from this run (for reference)

- sparsity_only first elim: step 31–45, border_irregularity universal
- sparsity_concurvity first elim: step 21–26, batches of 2–4 features
- Compute: 3.4 hours total
- STEP_5_COMPLETE.flag was written (now stale — replaced by corrected run flag)
