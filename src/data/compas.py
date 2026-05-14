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


def compas_numeric_feature_matrix(
    df: pd.DataFrame, columns: tuple[str, ...] = COMPAS_COLUMNS
) -> tuple[pd.DataFrame, list[str]]:
    """Select numeric columns present in the frame; return matrix and used names."""
    use = [c for c in columns if c in df.columns]
    if not use:
        raise ValueError(f"None of {columns} found in COMPAS columns: {list(df.columns)}")
    X = df[use].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, use
