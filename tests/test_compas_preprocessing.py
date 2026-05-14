"""Sanity checks for ProPublica-filtered COMPAS preprocessing (NAM replication).

Full checks require ``data/compas/compas-scores-two-years.csv`` from
https://github.com/propublica/compas-analysis — tests skip if the raw file is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.compas import load_compas

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "compas" / "compas-scores-two-years.csv"

pytestmark = pytest.mark.skipif(not RAW.is_file(), reason=f"Raw COMPAS CSV not found at {RAW}")


def test_load_compas_preprocessing_sanity(tmp_path: Path) -> None:
    out = tmp_path / "compas_clean_test.csv"
    df = load_compas(RAW, out)

    assert len(df) == 6172
    assert df["age"].between(18, 96).all()
    assert df["priors_count"].between(0, 40).all()
    assert df["length_of_stay"].between(0, 900).all()
    assert set(df["race"].unique()) == {
        "African-American",
        "Asian",
        "Caucasian",
        "Hispanic",
        "Native American",
        "Other",
    }
    assert set(df["sex"].unique()) == {"Female", "Male"}
    assert set(df["two_year_recid"].unique()) == {0, 1}
    assert set(df["charge_degree"].unique()) == {1, 2}

    assert out.is_file()


def test_load_compas_writes_versioned_csv_under_data_compas() -> None:
    """Writes ``data/compas/compas_clean_v1.csv`` when raw data is present (local verification)."""
    versioned = ROOT / "data" / "compas" / "compas_clean_v1.csv"
    df = load_compas(RAW, versioned)
    assert versioned.is_file()
    assert len(df) == 6172
