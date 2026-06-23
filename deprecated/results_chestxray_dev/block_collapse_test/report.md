# Block-Collapse Hypothesis Test — Chest X-ray ANEC Evaluation

Pre-registered test of whether within-block input correlation predicts
the sparsity selection outcome at K ∈ {5, 8, 10}.
K=8 is the primary test (within the pre-registered 6-9 concept range).
K=5 and K=10 are sensitivity analyses.

---

## Pre-Registration (verbatim, frozen)

### Block definitions and per-block survivor predictions

| Block | Concepts | Within-block mean |r| | Predicted survivors |
|---|---|---|---|
| Consolidation | consolidation, focal_opacity, lobar_consolidation, dense_segmental_opacity, air_bronchograms | 0.634 | **partial collapse, 2-3 of 5** |
| Interstitial | bilateral_interstitial_pattern, peribronchial_cuffing, perihilar_infiltrates | 0.749 | **near-full collapse, 1 of 3** |
| Normal-anchor | clear_lung_fields, sharp_costophrenic_angles, normal_cardiac_silhouette, symmetric_lung_aeration | 0.9 | **full collapse, 1 of 4** |
| Ungrouped | pleural_effusion, silhouette_sign, patchy_infiltrate, parapneumonic_effusion, round_pneumonia | varies (0.23-0.46) | **most survive, 3-4 of 5** |

**Combined prediction at operating point retaining ~6-9 concepts:**
- 1 from Normal-anchor
- 1 from Interstitial
- 2-3 from Consolidation
- 3-4 from Ungrouped
- **Total predicted: 7-9 surviving concepts**

---

## Per-Condition Per-K Block Outcome Tables

### Sparsity-only (`sparsity_only`)

#### K=5

| Block | n concepts | Predicted range | mean ± std | Modal | Seed counts | Classification |
|---|---|---|---|---|---|---|
| Consolidation | 5 | [2, 3] | 1.0 ± 0.00 | 1 | [1, 1, 1, 1, 1] | BELOW_PREDICTION |
| Interstitial | 3 | [1, 1] | 0.0 ± 0.00 | 0 | [0, 0, 0, 0, 0] | BELOW_PREDICTION |
| Normal-anchor | 4 | [1, 1] | 2.2 ± 0.45 | 2 | [2, 2, 2, 3, 2] | ABOVE_PREDICTION |
| Ungrouped | 5 | [3, 4] | 1.6 ± 0.55 | 2 | [2, 2, 2, 1, 1] | BELOW_PREDICTION |

**Per-seed totals:** {42: 5, 43: 5, 44: 5, 45: 5, 46: 4}

**Total observed:** 4.8 ± 0.45 concepts surviving (predicted 7-9, **OUTSIDE**)

**Headline: 0 of 4 blocks within prediction (using modal counts)**

#### K=8 — **PRIMARY TEST**

| Block | n concepts | Predicted range | mean ± std | Modal | Seed counts | Classification |
|---|---|---|---|---|---|---|
| Consolidation | 5 | [2, 3] | 1.8 ± 0.45 | 2 | [2, 2, 2, 2, 1] | WITHIN_PREDICTION |
| Interstitial | 3 | [1, 1] | 0.4 ± 0.55 | 0 | [1, 0, 1, 0, 0] | BELOW_PREDICTION |
| Normal-anchor | 4 | [1, 1] | 2.6 ± 0.55 | 3 | [2, 3, 2, 3, 3] | ABOVE_PREDICTION |
| Ungrouped | 5 | [3, 4] | 2.8 ± 0.45 | 3 | [3, 3, 3, 3, 2] | WITHIN_PREDICTION |

**Per-seed totals:** {42: 8, 43: 8, 44: 8, 45: 8, 46: 6}

**Total observed:** 7.6 ± 0.89 concepts surviving (predicted 7-9, within)

**Headline: 2 of 4 blocks within prediction (using modal counts)**

#### K=10

| Block | n concepts | Predicted range | mean ± std | Modal | Seed counts | Classification |
|---|---|---|---|---|---|---|
| Consolidation | 5 | [2, 3] | 2.2 ± 0.45 | 2 | [2, 3, 2, 2, 2] | WITHIN_PREDICTION |
| Interstitial | 3 | [1, 1] | 1.0 ± 0.00 | 1 | [1, 1, 1, 1, 1] | AT_PREDICTION |
| Normal-anchor | 4 | [1, 1] | 3.4 ± 0.55 | 3 | [4, 3, 4, 3, 3] | ABOVE_PREDICTION |
| Ungrouped | 5 | [3, 4] | 3.2 ± 0.45 | 3 | [3, 3, 3, 3, 4] | WITHIN_PREDICTION |

**Per-seed totals:** {42: 10, 43: 10, 44: 10, 45: 9, 46: 10}

**Total observed:** 9.8 ± 0.45 concepts surviving (predicted 7-9, **OUTSIDE**)

**Headline: 3 of 4 blocks within prediction (using modal counts)**

### Sparsity + Concurvity (`sparsity_conc`)

#### K=5

| Block | n concepts | Predicted range | mean ± std | Modal | Seed counts | Classification |
|---|---|---|---|---|---|---|
| Consolidation | 5 | [2, 3] | 1.2 ± 0.45 | 1 | [2, 1, 1, 1, 1] | BELOW_PREDICTION |
| Interstitial | 3 | [1, 1] | 0.2 ± 0.45 | 0 | [0, 0, 0, 0, 1] | BELOW_PREDICTION |
| Normal-anchor | 4 | [1, 1] | 1.8 ± 0.45 | 2 | [2, 2, 2, 2, 1] | ABOVE_PREDICTION |
| Ungrouped | 5 | [3, 4] | 1.6 ± 0.55 | 2 | [1, 2, 2, 2, 1] | BELOW_PREDICTION |

**Per-seed totals:** {42: 5, 43: 5, 44: 5, 45: 5, 46: 4}

**Total observed:** 4.8 ± 0.45 concepts surviving (predicted 7-9, **OUTSIDE**)

**Headline: 0 of 4 blocks within prediction (using modal counts)**

#### K=8 — **PRIMARY TEST**

| Block | n concepts | Predicted range | mean ± std | Modal | Seed counts | Classification |
|---|---|---|---|---|---|---|
| Consolidation | 5 | [2, 3] | 2.0 ± 0.71 | 2 | [3, 2, 2, 1, 2] | WITHIN_PREDICTION |
| Interstitial | 3 | [1, 1] | 0.4 ± 0.55 | 0 | [0, 0, 0, 1, 1] | BELOW_PREDICTION |
| Normal-anchor | 4 | [1, 1] | 2.6 ± 0.55 | 3 | [2, 2, 3, 3, 3] | ABOVE_PREDICTION |
| Ungrouped | 5 | [3, 4] | 2.8 ± 0.45 | 3 | [3, 3, 3, 3, 2] | WITHIN_PREDICTION |

**Per-seed totals:** {42: 8, 43: 7, 44: 8, 45: 8, 46: 8}

**Total observed:** 7.8 ± 0.45 concepts surviving (predicted 7-9, within)

**Headline: 2 of 4 blocks within prediction (using modal counts)**

#### K=10

| Block | n concepts | Predicted range | mean ± std | Modal | Seed counts | Classification |
|---|---|---|---|---|---|---|
| Consolidation | 5 | [2, 3] | 2.2 ± 0.84 | 2 (tie: tie [2, 3]) | [3, 2, 3, 1, 2] | WITHIN_PREDICTION |
| Interstitial | 3 | [1, 1] | 0.6 ± 0.55 | 1 | [0, 1, 0, 1, 1] | AT_PREDICTION |
| Normal-anchor | 4 | [1, 1] | 2.8 ± 0.45 | 3 | [3, 2, 3, 3, 3] | ABOVE_PREDICTION |
| Ungrouped | 5 | [3, 4] | 3.6 ± 0.55 | 4 | [3, 4, 3, 4, 4] | WITHIN_PREDICTION |

**Per-seed totals:** {42: 9, 43: 9, 44: 9, 45: 9, 46: 10}

**Total observed:** 9.2 ± 0.45 concepts surviving (predicted 7-9, **OUTSIDE**)

**Headline: 3 of 4 blocks within prediction (using modal counts)**

---

## Headline Outcome Summary

| Condition | K | Blocks within prediction | Total survivors (mean ± std) | In 7-9 range? |
|---|---|---|---|---|
| sparsity_only | K=5 | 0 of 4 | 4.8 ± 0.45 | no |
| sparsity_only | K=8 (primary) | 2 of 4 | 7.6 ± 0.89 | yes |
| sparsity_only | K=10 | 3 of 4 | 9.8 ± 0.45 | no |
| sparsity_conc | K=5 | 0 of 4 | 4.8 ± 0.45 | no |
| sparsity_conc | K=8 (primary) | 2 of 4 | 7.8 ± 0.45 | yes |
| sparsity_conc | K=10 | 3 of 4 | 9.2 ± 0.45 | no |

**Primary test (K=8):** 2 of 4 blocks within prediction for both conditions.

---

## Concept-Level Supplementary Table

Exploratory (not pre-registered). Per-concept survival rate vs. block membership.

### sparsity_only

| Block | Concept | K=5 (n/5) | K=8 (n/5) | K=10 (n/5) |
|---|---|---|---|---|
| Consolidation | consolidation | 5/5 | 5/5 | 5/5 |
| Consolidation | focal_opacity | 0/5 | 4/5 | 5/5 |
| Consolidation | lobar_consolidation | 0/5 | 0/5 | 1/5 |
| Consolidation | dense_segmental_opacity | 0/5 | 0/5 | 0/5 |
| Consolidation | air_bronchograms | 0/5 | 0/5 | 0/5 |
| Interstitial | bilateral_interstitial_pattern | 0/5 | 0/5 | 0/5 |
| Interstitial | peribronchial_cuffing | 0/5 | 2/5 | 5/5 |
| Interstitial | perihilar_infiltrates | 0/5 | 0/5 | 0/5 |
| Normal-anchor | clear_lung_fields | 5/5 | 5/5 | 5/5 |
| Normal-anchor | sharp_costophrenic_angles | 0/5 | 0/5 | 2/5 |
| Normal-anchor | normal_cardiac_silhouette | 1/5 | 3/5 | 5/5 |
| Normal-anchor | symmetric_lung_aeration | 5/5 | 5/5 | 5/5 |
| Ungrouped | pleural_effusion | 0/5 | 0/5 | 1/5 |
| Ungrouped | silhouette_sign | 5/5 | 5/5 | 5/5 |
| Ungrouped | patchy_infiltrate | 3/5 | 5/5 | 5/5 |
| Ungrouped | parapneumonic_effusion | 0/5 | 0/5 | 0/5 |
| Ungrouped | round_pneumonia | 0/5 | 4/5 | 5/5 |

### sparsity_conc

| Block | Concept | K=5 (n/5) | K=8 (n/5) | K=10 (n/5) |
|---|---|---|---|---|
| Consolidation | consolidation | 5/5 | 5/5 | 5/5 |
| Consolidation | focal_opacity | 1/5 | 3/5 | 4/5 |
| Consolidation | lobar_consolidation | 0/5 | 1/5 | 1/5 |
| Consolidation | dense_segmental_opacity | 0/5 | 1/5 | 1/5 |
| Consolidation | air_bronchograms | 0/5 | 0/5 | 0/5 |
| Interstitial | bilateral_interstitial_pattern | 0/5 | 0/5 | 1/5 |
| Interstitial | peribronchial_cuffing | 1/5 | 2/5 | 2/5 |
| Interstitial | perihilar_infiltrates | 0/5 | 0/5 | 0/5 |
| Normal-anchor | clear_lung_fields | 3/5 | 5/5 | 5/5 |
| Normal-anchor | sharp_costophrenic_angles | 1/5 | 3/5 | 4/5 |
| Normal-anchor | normal_cardiac_silhouette | 0/5 | 0/5 | 0/5 |
| Normal-anchor | symmetric_lung_aeration | 5/5 | 5/5 | 5/5 |
| Ungrouped | pleural_effusion | 0/5 | 3/5 | 5/5 |
| Ungrouped | silhouette_sign | 5/5 | 5/5 | 5/5 |
| Ungrouped | patchy_infiltrate | 1/5 | 2/5 | 2/5 |
| Ungrouped | parapneumonic_effusion | 0/5 | 0/5 | 2/5 |
| Ungrouped | round_pneumonia | 2/5 | 4/5 | 4/5 |

---

## Discussion of Deviations

### Normal-anchor block: persistent ABOVE_PREDICTION

The pre-registration predicted full collapse to 1 of 4 concepts, motivated by
the very high within-block correlation (r=0.90). Observed: 2-3 concepts survive
at all K values and both conditions. `clear_lung_fields` and `symmetric_lung_aeration`
survive near-universally; `sharp_costophrenic_angles` and `normal_cardiac_silhouette`
survive partially. The high correlation prediction held directionally (the block does
collapse relative to dense), but the floor was higher than expected. A plausible
mechanism: in a dataset where distinguishing NORMAL from PNEUMONIA is the dominant
task, normal-tissue markers collectively provide the strongest signal for the NORMAL
class. Despite high mutual correlation, the model retains multiple normal-anchor
features because their combined weight on the normal class exceeds what any single
concept provides. This is a failure mode of the block-collapse prediction that did
not account for the prediction target being a multi-class discriminative model rather
than a redundancy-pruning procedure.

### Interstitial block: BELOW_PREDICTION at K=5 and K=8

Predicted 1 of 3 to survive (near-full collapse). Observed: 0 survive at K=5/8,
1 survives at K=10 (primary survivor: `peribronchial_cuffing`). At K=5/8, the total
concept budget is too tight to accommodate any interstitial feature — the block
over-collapses relative to the prediction. This may reflect that the interstitial
signal (which primarily differentiates bacterial vs. viral pneumonia subtypes) is
partially redundant with Ungrouped features such as `patchy_infiltrate` and
`round_pneumonia` at small K. The prediction's 1-of-3 floor was plausible for larger K.

### K-level pattern across blocks

The Consolidation and Ungrouped blocks are the best-calibrated: both fall within
the predicted range at K=8 (primary) and K=10. Both fall below prediction at K=5,
which is expected — K=5 is well below the pre-registered 6-9 concept total, so
at least some blocks must over-collapse. This is a feature of the test design,
not a failure of the prediction. The K=5 results provide useful sensitivity data
about which blocks collapse first under extreme budgets.

### sparsity_only vs. sparsity_conc

The headline outcome (n_within) is identical at every K value across conditions.
The main observable difference is higher seed-to-seed variance in sparsity_conc
at some K values (e.g., Consolidation K=10: std=0.84 vs. 0.45 for sparsity_only),
attributable to the compressed regularization cascade in sparsity_conc where a
single proximal step can drop multiple features simultaneously. Despite this, the
block-level modal patterns are consistent across conditions.

---

*Generated from `results/chestxray/anec_evaluation/by_seed.csv`. 
Output directory: `results/chestxray/block_collapse_test/`. 
Pre-registration block definitions are frozen and not derived from this data.*
