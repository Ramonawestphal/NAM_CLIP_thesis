# Deprecated Pre-Audit Results

**Date moved:** 2026-06-05  
**Git commit at time of move:** `930413d`  
**Reason:** Methodology audit identified protocol issues; results must not be cited in thesis.

## Why these results are deprecated

A read-only methodology audit (`results/methodology_audit.md`) identified the following
issues in the pre-audit pipeline:

**Issue 1 (Critical):** Architecture selection in Stage 2 (`sweep_nam_v6_multiseed.py`)
used mean *test* balanced accuracy to rank configurations and pick the winner (`hidden=[64,32]`,
`dropout=0.10`, `weight_decay=1e-5`). The test set should never influence model selection.
This contamination propagates to every number in `results/thesis_tables/` and
`results/final_models/`.

**Issue 2 (Medium):** Stage 1 sweep (`sweep_nam_v6.py`) evaluated and stored test
metrics for all 12 configurations in `sweep_results.csv`, even though val was used
for ranking. The test set was therefore inspected for all configs.

**Issues 3–8:** Additional lower-severity issues (missing `random.seed`, end-of-patience
warm-start checkpoints, std denominator inconsistency, single-seed concurvity lambda
selection, scaler path inconsistency, missing CUDA determinism). See the audit for details.

See `results/methodology_audit.md` for the full issue list with file:line citations.

## Contents

```
reports_nam/         ← all directories from reports/nam/ containing trained NAM models
  base/              ← v5 base NAM (5 seeds)
  v6_base/           ← v6 Phase-1 NAM (5 seeds)
  v6_sweep/          ← Stage 1 hyperparameter sweep (12 configs, seed 42)
  v6_sweep_multiseed/← Stage 2 multi-seed validation — CONTAMINATED (selected by test)
  v6_concurvity_sweep/ ← concurvity lambda sweep (10 values, seed 42)
  v6_final/          ← 5-seed final NAM (config 9, concurvity lambda=1.0)
  v6_final_lambda0.1/← ablation run
  v6_final_lambda1/  ← ablation run
  v6_sparsity_sweep/ ← sparsity sweep from old protocol
  sparsity_sweep/
  sanity_*/          ← development diagnostic runs
  diagnostics/

thesis_tables/       ← final reported numbers (invalid — propagate Issue 1 bias)
final_models/        ← warm-start sparsity final model checkpoints
sparsity_sweep/      ← warm-start path results from old protocol
concurvity_verification/
concurvity_warmstart/
warm_start/
warm_start_smoketest/
```

## How to recover the pre-audit state

The git history at commit `930413d` contains the full code state at the time these
results were produced. The results themselves are preserved here verbatim.

## Replacement

Corrected results will be written to `results/v7/` by the new pipeline under
`scripts/v7/`. That pipeline fixes Issues 1–8 and 9–11. See
`results/v7/methodology.md` (written after training completes) for the corrected
protocol documentation.

## Do not delete this folder

These results are retained for reference and comparison. The corrected pipeline will
produce a side-by-side comparison table in `results/v7/thesis_tables/comparison_vs_deprecated.csv`.
