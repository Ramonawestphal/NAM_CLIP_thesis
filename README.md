# clip-nam-thesis

Thesis codebase: Neural Additive Models (NAM) with planned CLIP feature integration for fairness and interpretability experiments (COMPAS, ChestX-ray, HAM10000).

## Layout

- `src/nam` — base NAM implementation (Phase 1)
- `src/extensions` — concurvity, sparsity, interactions (Phase 2)
- `src/data` — dataset loaders and preprocessing
- `src/features` — CLIP feature extraction (Phase 1, week 2)
- `src/train.py` — single training entry point
- `configs/` — experiment configs for reproducibility
- `docs/` — place `proposal.pdf`, `nam_paper.pdf`, `thesis_plan.pdf` (not tracked if large)
- `data/` — raw data per dataset (see per-folder notes)
- `results/` — metrics, figures, checkpoints (gitignored by default)
- `tests/` — includes a COMPAS regression test

## Setup

```bash
cd clip-nam-thesis
pip install -r requirements.txt
```

Place ProPublica COMPAS CSV files under `data/compas/`. Other datasets are populated when you download them.

## Train

```bash
python -m src.train --config configs/example_compas.yaml
```

## Test

```bash
pytest tests/ -q
```
