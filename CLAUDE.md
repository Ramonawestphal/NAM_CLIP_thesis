# Claude Code — clip-nam-thesis

## Project

Interpretable neural models (NAM) extended with CLIP features for tabular + vision settings. Phased work: base NAM (Phase 1), then concurvity, sparsity, and interactions (Phase 2).

## Conventions

- **Source of truth for runs**: `configs/*.yaml` (or JSON). Do not hardcode hyperparameters in `train.py` beyond CLI wiring.
- **Data**: Raw downloads live under `data/<dataset>/`. Loaders in `src/data/` only read from those paths (or env vars); never commit large binaries.
- **Results**: Write metrics, figures, and checkpoints under `results/` with a dated run subfolder when useful.
- **Notebooks**: Exploration only; reproducible experiments go through `python -m src.train` (or `src/train.py` as invoked from project root).

## Commands (from `clip-nam-thesis/`)

- Install: `pip install -r requirements.txt`
- Train: `python -m src.train --config configs/example_compas.yaml`
- Tests: `pytest tests/ -q`

## When editing

- Keep `src/nam` independent of dataset-specific code; dataset logic stays in `src/data`.
- CLIP-related code belongs in `src/features` and should degrade gracefully when optional deps are missing (document in README).
