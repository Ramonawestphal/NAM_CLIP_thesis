# Deprecated — Warm-up = 5 Intermediate Results

**Date moved:** 2026-06-05
**Git commit at time of move:** `930413d`

## Why these results are deprecated

These are intermediate v7 results produced with `warmup_epochs = 5` (Siems et al.
2023, Appendix C.1: introduce concurvity regularization after 5% of total training).
They were replaced after a diagnostic experiment demonstrated that the 5-epoch warm-up
degraded both seed stability and R_perp regularization strength on this dataset and
protocol.

### Diagnostic evidence

A three-way comparison (`scripts/v7/train_final.py --condition concurvity_only`) was
run at λ_c = 3.0 across 5 seeds each for warmup ∈ {0, 2, 5}:

| warmup | test bal. acc (mean ± std) | val R_perp (mean ± std) |
|--------|---------------------------|-------------------------|
| 0      | **0.516 ± 0.010**         | **0.109 ± 0.019**       |
| 2      | 0.509 ± 0.012             | 0.122 ± 0.014           |
| 5      | 0.514 ± 0.021             | 0.150 ± 0.051           |

warmup = 5 produced the highest seed variance and the weakest R_perp reduction.
The root cause: with patience-based early stopping, the val balanced accuracy achieved
during the unregularized warm-up window creates a high baseline that concurvity-active
training must exceed. For ~40% of seeds this triggers early stopping on barely-
regularized checkpoints.

Full diagnostic: `results/v7/diagnostic_warmup/comparison.md`
Methodology justification: `results/v7/methodology.md` (subsection "Concurvity
warm-up: deviation from Siems et al.")

## Contents

```
concurvity_only/      ← 5-seed concurvity_only run, warmup_epochs=5, lambda_c=3.0
                         Config 10: hidden=[64,32], dropout=0.1, wd=1e-4
                         Produced by STEP 4 (pre-diagnostic)
```

## Replacement

The replacement results live at `results/v7/concurvity_only/` and were copied from
`results/v7/diagnostic_warmup/warmup_0/` (Setting A).
The replacement uses warmup_epochs = 0 and is otherwise identical.

## Do not delete this folder

Retained for reproducibility audit. These results must NOT be cited in the thesis.
