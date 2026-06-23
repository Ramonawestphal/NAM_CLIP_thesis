# Sparsity Sweep Condition Summary — sparsity_concurvity

**Date:** 2026-06-06
**Seeds:** [42, 43, 44, 45, 46]
**Steps per seed (min):** 102

## Dense baseline

| metric | value |
|--------|-------|
| dense_val_balacc (seed=42) | 0.6009 |
| Rule A threshold (−0.02) | 0.5809 |

## Sparsity onset

First feature zeroed at **lambda_s ≈ 2.1911e+00**

## Rule A candidate operating point

Mean val_balacc stayed above threshold throughout — the performance boundary was not reached. If the sweep is complete (500 steps), there may be no meaningful sparsity-accuracy trade-off at this dataset.

## Step-level summary (mean ± std across seeds, every 50 steps)

| step | lambda_s | n_active (mean±std) | val_balacc (mean±std) |
|------|----------|---------------------|-----------------------|
|    1 | 1.0000e+00 | 24.0 +/- 0.00 | 0.5247 +/- 0.0343 |
|   50 | 6.8333e+00 | 13.8 +/- 1.10 | 0.5186 +/- 0.0190 |
|  100 | 4.8562e+01 | 2.6 +/- 0.89 | 0.1937 +/- 0.0309 |

---
*Full data in path_seedN.csv.  Operating point selection: STEP 6.*