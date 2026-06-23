# Deprecated — Smoke Test v2 (lambda_0=1e-3, epsilon=0.025, 100 steps)

**Date moved:** 2026-06-06

## Why deprecated

This smoke test used `lambda_0=1e-3` with the extended schedule (`epsilon=0.025,
max_lambda_steps=500`). The 100-step smoke test only reached `lambda_max ≈ 0.012`.

### Root cause

The proximal shrinkage factor per batch is:
```
shrinkage = 1 - lr * lambda / ||theta||
```
With `lr=0.001`, `lambda=0.012`, and feature norms `||theta|| ≈ 6-8`:
```
shrinkage ≈ 1 - 0.001 * 0.012 / 7 ≈ 1 - 1.7e-6 ≈ 0.9999983
```
Adam's parameter updates grow features at a rate that completely overwhelms this
negligible shrinkage. `n_active = 24` for all 100 steps; many feature norms
**grew** rather than shrank over the course of the smoke test.

### Evidence

From `sparsity_only/path_seed42.csv`:
- `n_active = 24` for all 100 steps
- Notable norm increases over 100 steps:
  - `norm_atypical_pigment_network`: 7.08 → 8.19 (+1.11)
  - `norm_blue_white_veil`: 6.89 → 8.48 (+1.59)
  - `norm_blue_grey_ovoid_nests`: 7.25 → 7.41 (+0.16)
- `norm_symmetric_uniform` was the only consistent decreaser: 6.51 → 5.66 (−0.85)
  but never approached zero

### Replacement

Schedule replaced with `lambda_0=1.0`, `epsilon=0.04`, `max_lambda_steps=150`.
This starts in the regime where sparsity is actually achievable:
- Step 1:   lambda = 1.0
- Step 50:  lambda ≈ 7.1
- Step 100: lambda ≈ 50.5
- Step 150: lambda ≈ 359

## Do not delete

Retained for audit trail.
