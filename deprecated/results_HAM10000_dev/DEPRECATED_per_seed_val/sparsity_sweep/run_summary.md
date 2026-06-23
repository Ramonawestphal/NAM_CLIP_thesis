# STEP 5 — Sparsity Warm-Start Sweep: Run Summary

**Date:** 2026-06-06 → 2026-06-07  
**Status:** ✅ COMPLETE — all 10 runs finished, STEP_5_COMPLETE.flag written

---

## 1. Schedule parameters

```json
{
  "lambda_0":         1.0,
  "epsilon":          0.04,
  "max_lambda_steps": 150,
  "max_lambda":       1000.0,
  "lambda_at_t149":   345.1
}
```

Lambda preview: t=0 → 1.00, t=50 → 7.11, t=100 → 50.5, t=149 → 345.1

Concurvity lambda: 0.0 (sparsity_only) / 3.0 (sparsity_concurvity, from winner.json)  
Architecture: hidden=[64,32], dropout=0.1, weight_decay=1e-4 (Config 10)  
Device: CPU

---

## 2. Compute time

| Condition | Seed | Dense epochs | Path steps | Elapsed (min) |
|-----------|------|-------------|------------|---------------|
| sparsity_only | 42 | 62 | 103 | 17.0 |
| sparsity_only | 43 | 36 | 102 | 18.0 |
| sparsity_only | 44 | 42 | 103 | 19.7 |
| sparsity_only | 45 | 41 | 104 | 19.1 |
| sparsity_only | 46 | 31 | 103 | 18.6 |
| **sparsity_only total** | | | | **96.3 min** |
| sparsity_concurvity | 42 | 34 | 103 | 19.6 |
| sparsity_concurvity | 43 | 38 | 102 | 19.4 |
| sparsity_concurvity | 44 | 32 | 103 | 21.2 |
| sparsity_concurvity | 45 | 75 | 102 | 22.0 |
| sparsity_concurvity | 46 | 36 | 103 | 21.5 |
| **sparsity_concurvity total** | | | | **107.6 min** |
| **Grand total** | | | | **~3.4 hours** |

Note: all seeds terminated because n_active reached 0 (full elimination), not budget timeout.  
Max steps reached was 104 (seed 45, sparsity_only); lambda_max at that point ≈ 56.8.

---

## 3. Dense baselines

Each seed uses its own validation split (val_random_state=seed, Issue 7 fix). Dense
baselines therefore vary across seeds — this is expected.

| Seed | sparsity_only dense_val_balacc | sparsity_concurvity dense_val_balacc |
|------|-------------------------------|--------------------------------------|
| 42   | 0.6130                        | 0.6009 (with λ_c=3.0)                |
| 43   | 0.5483                        | 0.5106                               |
| 44   | 0.5704                        | 0.5220                               |
| 45   | 0.5603                        | 0.5397                               |
| 46   | 0.5624                        | 0.5637                               |
| Mean | **0.5709 ± 0.0244**           | **0.5474 ± 0.0347**                  |

The concurvity-regularised dense model is slightly lower on average (−0.024), consistent
with the concurvity sweep finding that λ_c=3.0 costs ~0.014 in val_balacc at the
concurvity winner.

---

## 4. Aggregate step-level summary

### 4a. sparsity_only (mean ± std across 5 seeds)

| step | lambda   | n_active (mean±std) | val_balacc (mean±std) | val_auc (mean) |
|------|----------|---------------------|-----------------------|----------------|
|   1  |   1.000  |  24.0 ± 0.0         |  0.5561 ± 0.0359      | —              |
|  10  |   1.423  |  24.0 ± 0.0         |  0.5572 ± 0.0278      | —              |
|  20  |   2.107  |  24.0 ± 0.0         |  0.5505 ± 0.0305      | —              |
|  30  |   3.119  |  24.0 ± 0.0         |  0.5590 ± 0.0304      | —              |
|  40  |   4.616  |  23.4 ± 0.5         |  0.5540 ± 0.0232      | —              |
|  50  |   6.833  |  22.4 ± 0.5         |  0.5461 ± 0.0164      | —              |
|  60  |  10.115  |  19.8 ± 0.4         |  0.5318 ± 0.0200      | —              |
|  70  |  14.973  |  16.4 ± 0.9         |  0.5196 ± 0.0218      | —              |
|  80  |  22.163  |  15.0 ± 1.2         |  0.4963 ± 0.0245      | —              |
|  90  |  32.807  |  13.2 ± 1.3         |  0.4718 ± 0.0234      | —              |
| 100  |  48.563  |   6.0 ± 1.4         |  0.1825 ± 0.0253      | —              |

First mean n_active < 24: **step 40** (mean 23.4; seed 31 already at 23, seeds 38+45 at 23)

### 4b. sparsity_concurvity (mean ± std across 5 seeds)

| step | lambda   | n_active (mean±std) | val_balacc (mean±std) | val_auc (mean) |
|------|----------|---------------------|-----------------------|----------------|
|   1  |   1.000  |  24.0 ± 0.0         |  0.5247 ± 0.0343      | —              |
|  10  |   1.423  |  24.0 ± 0.0         |  0.5262 ± 0.0366      | —              |
|  20  |   2.107  |  24.0 ± 0.0         |  0.5318 ± 0.0328      | —              |
|  30  |   3.119  |  20.0 ± 1.2         |  0.5377 ± 0.0336      | —              |
|  40  |   4.616  |  15.8 ± 1.6         |  0.5259 ± 0.0349      | —              |
|  50  |   6.833  |  13.8 ± 1.1         |  0.5186 ± 0.0190      | —              |
|  60  |  10.115  |  11.0 ± 1.2         |  0.5061 ± 0.0171      | —              |
|  70  |  14.973  |   7.8 ± 1.3         |  0.4783 ± 0.0253      | —              |
|  80  |  22.163  |   7.4 ± 1.1         |  0.4802 ± 0.0316      | —              |
|  90  |  32.807  |   5.4 ± 0.9         |  0.4452 ± 0.0375      | —              |
| 100  |  48.563  |   2.6 ± 0.9         |  0.1937 ± 0.0309      | —              |

First mean n_active < 24: **step 30** (mean 20.0 — already 4 features eliminated per seed)

---

## 5. First elimination and sparsity onset

### sparsity_only
| Seed | First-elim step | Lambda    | Feature eliminated first  |
|------|----------------|-----------|---------------------------|
| 42   | 45             | 5.617     | border_irregularity       |
| 43   | 36             | 3.946     | border_irregularity       |
| 44   | 31             | 3.243     | border_irregularity       |
| 45   | 38             | 4.268     | border_irregularity       |
| 46   | 45             | 5.617     | border_irregularity       |

**`border_irregularity` is the universally first-eliminated feature across all seeds.**  
Mean first-elimination lambda: **4.54** (range: 3.24–5.62)

### sparsity_concurvity
| Seed | First-elim step | Lambda    | Feature(s) eliminated first                                           |
|------|----------------|-----------|-----------------------------------------------------------------------|
| 42   | 22             | 2.279     | border_irregularity, irregular_streaks, irregular_pigmentation, regression_structures (4 at once) |
| 43   | 26             | 2.666     | border_irregularity, colour_variation, irregular_pigmentation, regression_structures (4 at once)  |
| 44   | 21             | 2.191     | border_irregularity                                                   |
| 45   | 23             | 2.370     | irregular_streaks                                                     |
| 46   | 23             | 2.370     | border_irregularity, regression_structures (2 at once)               |

First eliminations come **earlier** and often in **batches** (up to 4 at once).  
The concurvity regularisation in the dense phase pre-shrinks correlated features,
so they reach the proximal threshold simultaneously.

Mean first-elimination lambda: **2.37** (range: 2.19–2.67) — **2× earlier than sparsity_only**.

---

## 6. Rule A candidate operating points (per-seed; STEP 6 applies Rule A formally)

Rule A: largest lambda step where val_balacc ≥ dense_val_balacc − 0.02 (per-seed threshold)

### sparsity_only
| Seed | Dense   | Threshold | Rule A step | Lambda  | n_active | val_balacc | Note |
|------|---------|-----------|-------------|---------|----------|------------|------|
| 42   | 0.6130  | 0.5930    | 58          |  9.352  | 22       | 0.5939     |      |
| 43   | 0.5483  | 0.5283    | 58          |  9.352  | 22       | 0.5328     |      |
| 44   | 0.5704  | 0.5504    | 54          |  7.994  | 23       | 0.5534     |      |
| 45   | 0.5603  | 0.5403    | 27          |  2.772  | 24       | 0.5444     | ⚠️   |
| 46   | 0.5624  | 0.5424    | 57          |  8.992  | 21       | 0.5492     |      |

⚠️ **Seed 45 flag:** Rule A candidate at step 27 has n_active=24 (no eliminations yet).
The path val_balacc for seed 45 dips below threshold early and never recovers to a step
with sparsity. This seed's val split may be particularly harsh — its dense val_balacc
(0.5603) is reasonable, but the path fluctuations cross threshold before any feature
zeroes. STEP 6 should examine this seed's full path before including it in the
operating-point mean.

Seeds 42, 43, 44, 46 cluster at step 54–58 / lambda 8–10 / n_active 21–23 — strong agreement.

### sparsity_concurvity
| Seed | Dense   | Threshold | Rule A step | Lambda  | n_active | val_balacc | Note |
|------|---------|-----------|-------------|---------|----------|------------|------|
| 42   | 0.6009  | 0.5809    | 44          |  5.400  | 18       | 0.5812     |      |
| 43   | 0.5106  | 0.4906    | 60          | 10.115  | 12       | 0.4915     |      |
| 44   | 0.5220  | 0.5020    | 54          |  7.994  | 13       | 0.5041     |      |
| 45   | 0.5397  | 0.5197    | 34          |  3.648  | 16       | 0.5261     |      |
| 46   | 0.5637  | 0.5437    | 39          |  4.439  | 16       | 0.5458     |      |

Mean n_active at Rule A candidate: **15.0** (range: 12–18) vs **22.4** for sparsity_only.
This is the key result: concurvity pre-regularisation achieves ~7 additional eliminations
at the same relative performance threshold.

---

## 7. Elimination pattern quality

### Monotonicity
Both conditions: **n_active is strictly non-increasing across all 10 runs.** No re-activations
were observed (no "WARNING: concept re-activated" lines in stdout).

### Elimination cascade at high lambda
For both conditions, rapid batch eliminations occur at lambda ≈ 30–55:
- sparsity_only:   3–4 features eliminated simultaneously at steps 91, 100
- sparsity_concurvity:  similar cascades at steps 69 (4 at once), 81 (4 at once)

This is expected Group LASSO behaviour: once a feature's norm falls below lr×lambda, it
zeros in one proximal step; nearby-normed features often cascade together.

### Seed-to-seed agreement

**sparsity_only n_active std:**
- Steps 1–30: σ = 0.0 (identical — no eliminations yet in most seeds)
- Steps 40–70: σ = 0.5–0.9 (good agreement, minor variation in elimination order)
- Steps 80–100: σ = 1.2–1.4 (moderate spread — elimination cascade timing varies)

**sparsity_concurvity n_active std:**
- Steps 1–20: σ = 0.0
- Steps 30–60: σ = 1.1–1.6 (more spread than sparsity_only — concurvity creates more variable elimination order)
- Steps 70–90: σ = 0.9–1.3

Both conditions show low std at early steps and moderate spread at the cascade phase.
This is acceptable and expected.

---

## 8. Usability verdict

| Check | sparsity_only | sparsity_concurvity |
|-------|--------------|---------------------|
| All 5 seeds completed | ✅ | ✅ |
| No budget violations | ✅ | ✅ |
| No NaN/Inf in val_balacc | ✅ | ✅ |
| n_active drops below 24 | ✅ (by step 40) | ✅ (by step 30) |
| Monotone elimination | ✅ | ✅ |
| Clean cascade to zero | ✅ | ✅ |
| Low seed-to-seed σ on n_active | ✅ (≤1.4) | ✅ (≤1.6) |
| Seed-to-seed Rule A agreement | ✅ (4/5 seeds cluster; seed 45 flagged) | ✅ (wider spread; STEP 6 will aggregate) |

**Both conditions are usable for STEP 6 (Rule A operating point selection).**

**Actionable flags for STEP 6:**
1. Seed 45 (sparsity_only): Rule A candidate at n_active=24 — investigate full path before including in mean
2. sparsity_concurvity seeds show wider lambda spread at Rule A (3.6–10.1) vs sparsity_only (8.0–9.4 for seeds 42/43/44/46); examine whether a single shared lambda or per-seed lambda is more appropriate

---

## 9. Next step

STEP 6: Apply Rule A formally to select the operating point lambda for each condition,
then train the final 5-seed model at the selected operating point.

**Do NOT proceed to STEP 6 without explicit user direction.**

---

*Full per-seed data in: `sparsity_only/path_seed{42..46}.csv` and `sparsity_concurvity/path_seed{42..46}.csv`*  
*Config: `config.json`*  
*STEP complete flag: `results/v7/STEP_5_COMPLETE.flag`*
