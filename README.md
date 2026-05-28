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

### Local (CPU or CUDA)

```bash
pip install -r requirements.txt
```

For GPU support, install the CUDA-enabled torch build **before** running the command above (or instead of the torch line in requirements.txt):

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

### Google Colab (T4)

Colab ships with a compatible torch. Only the extra packages are needed:

```bash
pip install open_clip_torch==2.24.0 Pillow==10.3.0 tqdm==4.66.4
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
