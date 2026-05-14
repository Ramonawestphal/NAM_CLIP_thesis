# COMPAS preprocessing (NAM replication)

This document describes the cleaned COMPAS cohort used to replicate Agarwal et al. (2021) *Neural Additive Models* (NeurIPS) on the six features shown in the paper’s Figure 4. The raw file is the ProPublica **two-year recidivism** export:

- **Source:** [propublica/compas-analysis](https://github.com/propublica/compas-analysis) — `compas-scores-two-years.csv`
- **Local raw path:** `data/compas/compas-scores-two-years.csv`
- **Clean output:** `data/compas/compas_clean_v1.csv` (written by `load_compas`)

Implementation: `src/data/compas.py` — function `load_compas(raw_path, out_path)`.

## ProPublica filters

Filters follow the standard notebook **Compas-Analysis.ipynb** (same order as in that workflow). The NAM paper does not publish a preprocessing script; we adopt this public baseline and **lock the cohort size** to the replicated row count.

1. Keep rows where `days_b_screening_arrest` is between **-30 and 30** (inclusive). Values are coerced with `pd.to_numeric(..., errors="coerce")`; non-numeric rows do not satisfy the interval and are **excluded** (not imputed).
2. Drop rows where `is_recid == -1`.
3. Drop rows where `c_charge_degree == "O"`.
4. Drop rows where `score_text == "N/A"` (exact string match; raw file uses this literal).

After these steps the row count **must be exactly 6172**. If not, `load_compas` raises `AssertionError` with the actual count. This guards against silent drift if the upstream CSV changes.

**Note on “five” vs four filters:** Some write-ups count an extra notebook step separately; this codebase implements the **four filters above**, which match the usual ProPublica replication and yield **6172** rows. The post-filter assertion is the contract.

## Feature mapping (NAM → COMPAS)

| NAM (concept)     | Output column      | Rule |
|-------------------|--------------------|------|
| Age               | `age`              | Continuous integer from raw `age`. **`age_cat` is not used** (NAM uses continuous age, not ProPublica’s three buckets). |
| Charge degree     | `charge_degree`    | From `c_charge_degree`: `"F"` → `1`, `"M"` → `2`. |
| Length of stay    | `length_of_stay`   | `(c_jail_out - c_jail_in)` in days after `pd.to_datetime(..., errors="coerce")`; missing → **0**; negative days → **0**. |
| Prior counts      | `priors_count`     | Continuous integer from raw. |
| Race              | `race`             | Single string column; **six** categories in the filtered data. **No one-hot encoding** here (left to the model pipeline). |
| Gender            | `sex`              | `"Female"` / `"Male"` strings from raw. **No one-hot encoding** here. |

## Target

- **`two_year_recid`:** binary `0` / `1` (integer).

## Explicit non-decisions (out of scope for this function)

- **No z-score or other normalisation** inside `load_compas` — fit scalers on the training split only in the model pipeline to avoid leakage.
- **No extra row drops** beyond the four filters above.
- **No additional features** beyond the six inputs plus the target.

## Output schema

Columns are written **in this order** to CSV (`index=False`):

`age`, `charge_degree`, `length_of_stay`, `priors_count`, `race`, `sex`, `two_year_recid`

Intended dtypes after load: integer columns for `age`, `charge_degree`, `length_of_stay`, `priors_count`, `two_year_recid`; strings for `race` and `sex`.

## Regeneration

From the repository root (with dependencies installed):

```bash
python -c "from pathlib import Path; from src.data.compas import load_compas; p = Path('data/compas'); load_compas(p / 'compas-scores-two-years.csv', p / 'compas_clean_v1.csv')"
```

## Git and large files

`.gitignore` includes `data/compas/*.csv`, so raw and cleaned CSVs are **not committed** by default. Keep them locally or supply them in CI when running integration tests.

## Tests

`tests/test_compas_preprocessing.py` encodes the eight sanity checks from the replication spec (row count, value ranges, categorical levels, binary target, felony/misdemeanour encoding). Those tests **skip** if the raw ProPublica file is missing.
