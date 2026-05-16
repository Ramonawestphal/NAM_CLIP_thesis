from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

COMPAS_COLUMNS = (
    "age",
    "priors_count",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_compas_csv(data_dir: str | Path | None = None) -> Path | None:
    """Return first CSV found under data/compas (ProPublica naming varies)."""
    root = Path(data_dir) if data_dir is not None else _project_root() / "data" / "compas"
    if not root.is_dir():
        return None
    csvs = sorted(root.glob("*.csv"))
    return csvs[0] if csvs else None


def load_compas_frame(csv_path: str | os.PathLike[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


_COMPAS_PREPROCESS_USECOLS = (
    "days_b_screening_arrest",
    "is_recid",
    "c_charge_degree",
    "score_text",
    "age",
    "c_jail_in",
    "c_jail_out",
    "priors_count",
    "race",
    "sex",
    "two_year_recid",
)


def load_compas(raw_path: str | os.PathLike[str], out_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load ProPublica COMPAS CSV, apply standard filters, derive NAM features, save cleaned CSV.

    Filters match ProPublica ``Compas-Analysis.ipynb`` (row count must be 6172). See
    ``docs/preprocessing_compas.md``.
    """
    raw_path = Path(raw_path)
    out_path = Path(out_path)

    df = pd.read_csv(raw_path, usecols=list(_COMPAS_PREPROCESS_USECOLS))
    n_raw = len(df)

    days = pd.to_numeric(df["days_b_screening_arrest"], errors="coerce")
    df = df.loc[days.between(-30, 30, inclusive="both")].copy()
    n_after_days = len(df)

    df = df.loc[df["is_recid"] != -1]
    n_after_recid = len(df)

    df = df.loc[df["c_charge_degree"] != "O"]
    n_after_charge = len(df)

    df = df.loc[df["score_text"] != "N/A"]
    n_after_score = len(df)

    print(
        f"COMPAS filter log: raw={n_raw}"
        f" -> screening_window={n_after_days} (dropped {n_raw - n_after_days})"
        f" -> is_recid!=-1={n_after_recid} (dropped {n_after_days - n_after_recid})"
        f" -> charge!=O={n_after_charge} (dropped {n_after_recid - n_after_charge})"
        f" -> score!=N/A={n_after_score} (dropped {n_after_charge - n_after_score})"
    )

    n = n_after_score
    if not (6170 <= n <= 6175):
        raise AssertionError(
            f"Expected ~6172 rows after ProPublica filters (acceptable range 6170–6175), "
            f"got {n}. Check the filter log above for which step lost unexpected rows. "
            f"ProPublica source: https://github.com/propublica/compas-analysis"
        )

    jail_in = pd.to_datetime(df["c_jail_in"], errors="coerce")
    jail_out = pd.to_datetime(df["c_jail_out"], errors="coerce")
    los = (jail_out - jail_in).dt.days
    los = los.fillna(0).clip(lower=0)

    charge_map = {"F": 1, "M": 2}
    charge_degree = df["c_charge_degree"].map(charge_map).astype("int64")

    out = pd.DataFrame(
        {
            "age": pd.to_numeric(df["age"], errors="coerce").astype("int64"),
            "charge_degree": charge_degree,
            "length_of_stay": los.astype("int64"),
            "priors_count": pd.to_numeric(df["priors_count"], errors="coerce").astype("int64"),
            "race": df["race"].astype(str),
            "sex": df["sex"].astype(str),
            "two_year_recid": pd.to_numeric(df["two_year_recid"], errors="coerce").astype("int64"),
        }
    )

    cols = [
        "age",
        "charge_degree",
        "length_of_stay",
        "priors_count",
        "race",
        "sex",
        "two_year_recid",
    ]
    out = out[cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    return out


def compas_numeric_feature_matrix(
    df: pd.DataFrame, columns: tuple[str, ...] = COMPAS_COLUMNS
) -> tuple[pd.DataFrame, list[str]]:
    """Select numeric columns present in the frame; return matrix and used names."""
    use = [c for c in columns if c in df.columns]
    if not use:
        raise ValueError(f"None of {columns} found in COMPAS columns: {list(df.columns)}")
    X = df[use].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, use
