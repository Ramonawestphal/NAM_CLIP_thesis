# Concurvity Warm-up Diagnostic — Comparison Report

**Date generated:** 2026-06-05 15:14

## Experiment design

All three settings share identical hyperparameters; only `warmup_epochs` varies.

| Parameter | Value |
|-----------|-------|
| Condition | concurvity_only |
| lambda_c | 3.0 |
| Architecture | hidden=[64, 32], dropout=0.1, wd=1e-04 |
| Seeds | [42, 43, 44, 45, 46] |
| Max epochs | 100 |
| Patience | 15 |
| Fix A2 | post-warmup checkpoint reset active for warmup_epochs > 0 |

## Setting descriptions

| Setting | warmup_epochs | Description |
|---------|--------------|-------------|
| A | 0 | Concurvity active from epoch 1; no reset. Matches deprecated v6 protocol. |
| B | 5 | Current v7 default (5% of max_epochs). Matches Siems et al. 2023 App. C.1. |
| C | 2 | Short warm-up: stabilise initialisation without dominating early training. |

## 1. Main results (mean +/- std across 5 seeds)

| Metric | A (warmup=0) | B (warmup=5) | C (warmup=2) |
| --- | --- | --- | --- |
| Test bal. acc | 0.5162 +/- 0.0097 | 0.5139 +/- 0.0206 | 0.5089 +/- 0.0122 |
| Test macro F1 | 0.3834 +/- 0.0087 | 0.3901 +/- 0.0151 | 0.3869 +/- 0.0123 |
| Test AUC (OvR wtd) | 0.8281 +/- 0.0027 | 0.8375 +/- 0.0079 | 0.8331 +/- 0.0064 |
| Val bal. acc (at best ckpt) | N/A | N/A | N/A |
| Val R_perp (at best ckpt) | 0.1085 +/- 0.0186 | 0.1500 +/- 0.0512 | 0.1218 +/- 0.0144 |
| Mean best_epoch | 29.6 +/- 17.2 | 18.0 +/- 12.5 | 19.2 +/- 6.2 |
| Mean total_epochs | 44.6 +/- 17.2 | 33.0 +/- 12.5 | 34.2 +/- 6.2 |

## 2. Per-seed best epoch

| Seed | A (warmup=0) | B (warmup=5) | C (warmup=2) |
| --- | --- | --- | --- |
| 42 | 19 | 15 | 19 |
| 43 | 10 | 6 | 10 |
| 44 | 55 | 8 | 27 |
| 45 | 28 | 36 | 22 |
| 46 | 36 | 25 | 18 |

## 3. Diagnostic verdict

**Verdict: Warm-up neutral**

- No clear advantage: std_A=0.0097 vs std_B=0.0206, |R_perp_A - R_perp_B| = 0.0415 (tolerance = 0.02).
- SETTING C (warmup=2) shows improvement: mean bal_acc=0.5089 vs A=0.5162, B=0.5139; std=0.0122 vs A=0.0097, B=0.0206. Worth considering as an alternative.

Criteria applied:
- "Warm-up helping": SETTING B has lower test bal_acc std AND R_perp within 0.02 of SETTING A
- "Warm-up hurting": SETTING A has lower test bal_acc std AND R_perp within 0.02 of SETTING B
- "Warm-up neutral": neither criterion met

## 4. Recommended setting

**SETTING B (warmup_epochs=5, current v7 default) — no clear evidence to change; SETTING C (warmup_epochs=2) is a secondary candidate**

> This recommendation is based solely on val/test metrics from this diagnostic run.
> The user decides whether to adopt this recommendation for the final pipeline.
> Do NOT treat the test metrics above as final thesis numbers.
