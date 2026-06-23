# Sparsity Sweep Smoke Tests — Results Summary

**Date:** 2026-06-06  
**Schedule (v3):** lambda_0=1.0, epsilon=0.04, max_lambda_steps=150 (full)  
**Dense baseline (seed=42):** val_balacc=0.6130, val_auc=0.8626  

---

## Final verdict: ✅ PASS — proceed to full overnight run

Two smoke tests were run. The 30-step test (Option A) satisfied 3/4 criteria. The
60-step extension confirmed all 4 criteria with first elimination at step 45.

| # | Criterion | 30-step | 60-step |
|---|-----------|---------|---------|
| 1 | Step 1 val_balacc drop ≤ 0.05 below dense | ✅ +0.0001 | ✅ same |
| 2 | n_active drops below 24 | ❌ all 24 | ✅ step 45 → 23 |
| 3 | n_active monotonically non-increasing | ✅ trivially | ✅ 24→23→22→21→20 |
| 4 | Norm trajectory shows proximal-induced shrinkage | ✅ strong | ✅ exact zeroing |

---

## Smoke test v3b — 60 steps (Option A)

### Eliminations
| Event | Step | Lambda | Feature | Norm at step-1 | Norm zeroed |
|-------|------|--------|---------|----------------|-------------|
| 1st   |  45  | 5.617  | border_irregularity       | 6.911 | 0.000 |
| 2nd   |  50  | 6.833  | regression_structures     | 7.185 | 0.000 |
| 3rd   |  59  | 9.726  | irregular_dots_globules   | 6.983 | 0.000 |
| 4th   |  60  | 10.115 | healthy_skin              | 6.719 | 0.000 |

### n_active trajectory
```
Steps  1–44:  n_active = 24  (eroding; no zero crossings)
Step  45:     n_active = 23  ← border_irregularity zeroed
Steps 46–49:  n_active = 23
Step  50:     n_active = 22  ← regression_structures zeroed
Steps 51–58:  n_active = 22
Step  59:     n_active = 21  ← irregular_dots_globules zeroed
Step  60:     n_active = 20  ← healthy_skin zeroed
```

### val_balacc trajectory (selected steps)
| Step | Lambda | n_active | val_balacc |
|------|--------|----------|------------|
|  1   | 1.000  | 24 | 0.6131 |
| 10   | 1.423  | 24 | 0.6053 |
| 20   | 2.107  | 24 | 0.5999 |
| 30   | 3.119  | 24 | 0.6058 |
| 40   | 4.616  | 24 | 0.5883 |
| 45   | 5.617  | **23** | 0.5976 |
| 50   | 6.833  | **22** | 0.5654 |
| 55   | 8.314  | 22 | 0.5767 |
| 59   | 9.726  | **21** | 0.5696 |
| 60   | 10.115 | **20** | 0.5586 |

Rule A threshold = dense_val_balacc − 0.02 = 0.5930.  
Candidate operating point (smoke test, single seed) ≈ **step 45–49** (lambda 5.6–6.6,
n_active=23, val_balacc=0.595–0.599). Full Rule A selection runs at STEP 6 over all seeds.

### Norm decay: border_irregularity (fastest shrinking feature)
```
Step  1:  6.911
Step 10:  4.925
Step 20:  3.655
Step 30:  2.147
Step 40:  1.105
Step 43:  0.557
Step 44:  0.195
Step 45:  0.000  ← eliminated
```
Smooth monotone decay to exactly zero — textbook Group LASSO proximal behavior.

---

## Smoke test v3a — 30 steps (for reference)

Ran first. All norms declined (border_irregularity: 6.91→2.15, −69%) but no feature
reached zero. Extended to 60 steps to confirm zeroing. See `results.md` version 1 for
the full per-step table.

---

## Comparison with earlier failed schedules

| Schedule | lambda at step 30 | norm_blue_white_veil at step 30 | n_active at step 60 |
|----------|-------------------|---------------------------------|----------------------|
| v0 (broken) | 0.037 | ~8.5 (growing) | 24 |
| v2 smoke test | 0.012 | 8.48 (grew from 6.89) | 24/100 steps |
| **v3 (current)** | **3.12** | **3.58 (fell from 6.87)** | **20** |

---

## Authorisation

✅ **All criteria met. Ready for full overnight run.**

Full run command:
```bash
python scripts/v7/run_sparsity_sweep.py \
    --condition sparsity_only \
    --seeds 42 43 44 45 46 \
    --out_root results/v7/sparsity_sweep

python scripts/v7/run_sparsity_sweep.py \
    --condition sparsity_concurvity \
    --seeds 42 43 44 45 46 \
    --out_root results/v7/sparsity_sweep
```
Awaiting explicit user confirmation before launching.
