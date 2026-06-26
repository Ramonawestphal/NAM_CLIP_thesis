# Beyond Tabular Data: Extending Neural Additive Models to Medical Image Recognition via CLIP-Based Concept Features

Bachelor thesis codebase. Ina Ramona Westphal (683789), Erasmus School of Economics, Econometrics and Operations Research, supervised by Markus Mueller and Professor Maria Grith.

This repository implements Neural Additive Models (NAMs) trained on BiomedCLIP concept-similarity features for two medical image classification tasks: dermoscopy (HAM10000, seven classes, 24 concepts) and paediatric chest radiography (Kermany, three classes, 17 concepts). The thesis evaluates whether the correlation structure of CLIP concept scores permits identifiable additive interpretation, and combines concurvity regularisation with group sparsity to make the trade-off between accuracy and interpretability auditable through the Accuracy at Number of Effective Concepts framework. A binary-NAM replication of Agarwal et al. (2021) on the COMPAS recidivism benchmark is included as an implementation sanity check.
Thesis codebase: Neural Additive Models (NAM) extended with BiomedCLIP concept features for interpretable skin lesion classification (HAM10000) and chest X-ray classification (Kermany). Includes a COMPAS replication of Agarwal et al. (2021).

## Repository layout

```
configs/          Experiment configs (COMPAS)
data/             Raw data and pre-computed features (see Data section below)
  compas/         ProPublica COMPAS CSV
  chest_xray/     Kermany chest X-ray images
  ham10000/       HAM10000 dermoscopy images
  features/biomedclip/  Pre-computed BiomedCLIP concept scores (.npz)
  splits/         Pre-computed train/test/fold indices (.npz, .json)
docs/             Additional documentation
results/          Metrics, figures, checkpoints (written at runtime)
scripts/
  analysis/       Plotting and analysis scripts
  HAM10000/       HAM10000 training pipeline (architecture search → training)
  chestxray/      Chest X-ray training pipeline
src/
  data/           Dataset loaders (COMPAS)
  features/       BiomedCLIP feature extraction and prompt files
  models/         NAM model (NAMMulticlass, concurvity, sparsity)
  nam/            Base binary NAM (COMPAS replication)
  utils/          Seeding, metrics, plotting utilities
tests/            Unit and regression tests
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\pip install -r requirements.txt
# macOS / Linux:
.venv/bin/pip install -r requirements.txt
```

All package versions are pinned to those used during the thesis experiments. For GPU (CUDA 12.1):

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu121
```

---

## Data

### COMPAS

Download `compas-scores-two-years.csv` from the ProPublica GitHub repository
([https://github.com/propublica/compas-analysis](https://github.com/propublica/compas-analysis))
and place it at:

```
data/compas/compas-scores-two-years.csv
```

Preprocessing (row filtering, encoding) runs automatically when the training script is invoked.

### HAM10000

Download the HAM10000 dataset from ISIC ([https://www.isic-archive.com](https://www.isic-archive.com)) or Kaggle
([https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)).
Place all images under:

```
data/ham10000/          (*.jpg)
```

The pre-computed BiomedCLIP concept scores and lesion-grouped train/test split used in the thesis are stored at:

```
data/features/biomedclip/ham10000_concept_scores_v6.npz
data/splits/train_test_lesion_split.npz
data/splits/fold_indices.json
```

To regenerate features from scratch (requires GPU recommended, ~30 min):

```bash
python scripts/HAM10000/extract_features_biomedclip_ham10000.py
```

### Chest X-ray (Kermany)

Download the Kermany chest X-ray dataset from Kaggle
([https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia))
and place it at:

```
data/chest_xray/
  train/{NORMAL,PNEUMONIA}/
  val/{NORMAL,PNEUMONIA}/
  test/{NORMAL,PNEUMONIA}/
```

The pre-computed BiomedCLIP concept scores and patient-aware train/test split are at:

```
data/features/biomedclip/chestxray_concept_scores_v4.npz
data/splits/chestxray_outer_split.npz
```

To regenerate features from scratch (~15 min):

```bash
python scripts/chestxray/extract_features_biomedclip_chestxray_v4.py
python scripts/chestxray/prepare_dataset.py
```

---

## Reproducing results

### COMPAS replication (Agarwal et al. 2021)

```bash
# Single-split (paper path)
python -m src.train --config configs/compas_replication.yaml

# 5-fold CV with 20-model ensemble
python -m src.train --config configs/compas_replication.yaml --mode cv
```

Expected: AUC-ROC ≈ 0.737 ± 0.010 (paper: 0.741 ± 0.009).

### HAM10000 pipeline

Run steps in order from the project root:

```bash
# Step 1 — Architecture search (5-fold lesion-grouped CV, ~2 h)
python scripts/HAM10000/architecture_search_cv.py

# Step 2 — Plain NAM (5 seeds)
python scripts/HAM10000/train_final.py --condition plain_nam

# Step 3 — Concurvity sweep (selects lambda_c)
python scripts/HAM10000/run_concurvity_sweep.py

# Step 4 — NAM + concurvity regularisation
python scripts/HAM10000/train_final.py --condition concurvity_only

# Step 5 — Sparsity sweep (selects lambda_s)
python scripts/HAM10000/run_sparsity_sweep.py

# Step 6/7 — NAM + sparsity / NAM + both regularisers
python scripts/HAM10000/train_final.py --condition sparsity_only   --sparsity_lambda <lambda_s>
python scripts/HAM10000/train_final.py --condition sparsity_conc   --sparsity_lambda <lambda_s>
```

Each condition writes results to `results/HAM10000/<condition>/`.

### Chest X-ray pipeline

```bash
python scripts/chestxray/select_architecture.py
python scripts/chestxray/train_final.py --condition plain_nam
python scripts/chestxray/run_concurvity_sweep.py
python scripts/chestxray/run_sparsity_sweep.py
python scripts/chestxray/train_final.py --condition sparsity_conc --sparsity_lambda <lambda_s>
```

---

## Analysis and figures

```bash
# Shape function plots (HAM10000 and chest X-ray)
python scripts/analysis/plot_shape_functions.py
python scripts/analysis/plot_shape_functions_pairwise.py

# BiomedCLIP concept score heatmaps
python scripts/analysis/plot_concept_score_heatmaps.py

# Appendix figures (shape function selections, prototype images)
python scripts/analysis/plot_shape_functions_chestxray_selection.py
python scripts/analysis/plot_shape_functions_pairwise_selection.py
```

All figures are written to `results/analysis/`.

---

## Tests

```bash
pytest tests/ -q
```

---

## Reproducibility notes

- All random seeds are set via `src/utils/seeding.py` (`seed_everything()`), which sets Python `random`, NumPy, and PyTorch seeds and enables `torch.backends.cudnn.deterministic`.
- HAM10000 training uses seeds `[42, 43, 44, 45, 46]`; architecture search uses `CV_SEED=42`.
- BiomedCLIP model: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (downloaded automatically via HuggingFace Hub on first run).
- Concept prompts are versioned in `src/features/prompts/ham10000_prompts_v6_biomedclip.txt` and `chestxray_prompts_v4.txt`.
- Per-seed scalers are saved alongside checkpoints in `results/<dataset>/<condition>/seed_<N>/scaler.pkl`.
- Minor numerical differences (< 0.1% in metrics) may occur between CPU and GPU runs due to floating-point non-associativity.
