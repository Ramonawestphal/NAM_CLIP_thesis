# Sparsity Sweep Condition Summary — sparsity_only

**Date:** 2026-06-06
**Seeds:** [42]
**Steps per seed (min):** 30

## Dense baseline

| metric | value |
|--------|-------|
| dense_val_balacc (seed=42) | 0.6130 |
| Rule A threshold (−0.02) | 0.5930 |

## Sparsity onset

**No features zeroed.** The schedule did not reach the sparsity-inducing range.

> Note: if this was a smoke test (100 steps, lambda_max ≈ 0.012), this is
> expected — the sparsity-inducing range is lambda_s >> 1. Re-run with the
> full 500-step schedule to traverse the relevant range.

## Rule A candidate operating point

Largest mean_val_balacc >= 0.5930 at:
  step=30,  lambda_s=3.1187e+00,  mean_val_balacc=0.6058,  mean_n_active=24.0

**Do NOT use this as the operating point yet** — STEP 6 applies
Rule A formally and verifies the candidate.

## Step-level summary (mean ± std across seeds, every 50 steps)

| step | lambda_s | n_active (mean±std) | val_balacc (mean±std) |
|------|----------|---------------------|-----------------------|
|    1 | 1.0000e+00 | 24.0 +/- 0.00 | 0.6131 +/- 0.0000 |

---
*Full data in path_seedN.csv.  Operating point selection: STEP 6.*