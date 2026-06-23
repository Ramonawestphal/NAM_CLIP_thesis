# Deprecated — Short Lambda Schedule (seed-42 only)

**Date moved:** 2026-06-05

## Why deprecated

This is the initial STEP 5 warm-start sparsity sweep for seed=42, run with the
following parameters:

| parameter        | value   | problem                                      |
|------------------|---------|----------------------------------------------|
| `epsilon`        | 0.02    | too fine-grained; schedule grows slowly      |
| `max_lambda_steps` | 300   | reached only lambda ≈ 0.38 at step 300       |
| `max_warm_epochs`  | 50    | excessive per step; most steps ran full 50 ep |

### Consequence

The schedule `lambda_t = 1e-3 * 1.02^t` at t=300 yields:
```
1e-3 * 1.02^300 ≈ 0.40
```

Feature sub-network norms at the dense checkpoint were in the 5–7 range.
The proximal soft-thresholding threshold for zeroing is:
```
lambda_s >= norm / lr ≈ 6.5 / 0.001 = 6500
```

The schedule never reached anywhere near the sparsity-inducing range.
`n_active = 24` (all features active) for all 300 steps. No features
were zeroed; the sweep produced no sparsity information.

Prior deprecated operating points (lambda_s = 12.0, 23.7) were known to
cause feature zeroing, but were not reached by this schedule.

### Run metadata

- `run_config.json`: epsilon=0.02, max_lambda_steps=300, max_warm_epochs=50
- Elapsed: 141.2 min for 300 steps (~0.47 min/step)
- Seeds run: 42 only (single-seed structure; not multi-seed)

## Replacement

The replacement sweep uses:
- `epsilon = 0.025` (coarser steps)
- `max_lambda_steps = 500` (reaches lambda ≈ 233 at t=500)
- `max_warm_epochs = 30` (tighter per-step budget)
- `warm_patience = 6` (tighter early stopping)
- Both conditions: `sparsity_only` (λ_c=0) and `sparsity_concurvity` (λ_c=3.0)
- All 5 seeds (42–46)

Results live at `results/v7/sparsity_sweep/`.

## Do not delete

Retained for reproducibility audit.
