# Sparsity Sweep Condition Summary — sparsity_only

**Date:** 2026-06-06
**Seeds:** [42]
**Steps per seed (min):** 60

## Dense baseline

| metric | value |
|--------|-------|
| dense_val_balacc (seed=42) | 0.6130 |
| Rule A threshold (−0.02) | 0.5930 |

## Sparsity onset

First feature zeroed at **lambda_s ≈ 5.6165e+00**

## Rule A candidate operating point

Largest mean_val_balacc >= 0.5930 at:
  step=58,  lambda_s=9.3519e+00,  mean_val_balacc=0.5939,  mean_n_active=22.0

**Do NOT use this as the operating point yet** — STEP 6 applies
Rule A formally and verifies the candidate.

## Step-level summary (mean ± std across seeds, every 50 steps)

| step | lambda_s | n_active (mean±std) | val_balacc (mean±std) |
|------|----------|---------------------|-----------------------|
|    1 | 1.0000e+00 | 24.0 +/- 0.00 | 0.6131 +/- 0.0000 |
|   50 | 6.8333e+00 | 22.0 +/- 0.00 | 0.5654 +/- 0.0000 |

---
*Full data in path_seedN.csv.  Operating point selection: STEP 6.*