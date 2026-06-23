# Sparsity Sweep Condition Summary — sparsity_only

**Date:** 2026-06-06
**Seeds:** [42, 43, 44, 45, 46]
**Steps per seed (min):** 102

## Dense baseline

| metric | value |
|--------|-------|
| dense_val_balacc (seed=42) | 0.6130 |
| Rule A threshold (−0.02) | 0.5930 |

## Sparsity onset

First feature zeroed at **lambda_s ≈ 3.2434e+00**

## Rule A candidate operating point

Mean val_balacc stayed above threshold throughout — the performance boundary was not reached. If the sweep is complete (500 steps), there may be no meaningful sparsity-accuracy trade-off at this dataset.

## Step-level summary (mean ± std across seeds, every 50 steps)

| step | lambda_s | n_active (mean±std) | val_balacc (mean±std) |
|------|----------|---------------------|-----------------------|
|    1 | 1.0000e+00 | 24.0 +/- 0.00 | 0.5561 +/- 0.0359 |
|   50 | 6.8333e+00 | 22.4 +/- 0.55 | 0.5461 +/- 0.0164 |
|  100 | 4.8562e+01 | 6.0 +/- 1.41 | 0.1825 +/- 0.0253 |

---
*Full data in path_seedN.csv.  Operating point selection: STEP 6.*