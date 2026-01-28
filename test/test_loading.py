from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import pytest

from df_analyze._constants import (
    COMPLEX_XLSX,
    COMPLEX_XLSX2,
    SIMPLE_CSV,
    SIMPLE_CSV2,
    SIMPLE_XLSX,
)
from df_analyze.loading import load_csv, load_excel


def generate_spreadsheet(tempdir: Path) -> Path: ...


@pytest.mark.fast
def test_xlsx_loading() -> None:
    df, meta = load_excel(SIMPLE_XLSX)
    assert meta == "--target y --"
    assert list(df.columns) == ["x1", "x2", "x3", "y"]
    dtypes = list(df.dtypes)
    assert dtypes == ["float64", "float64", "float64", "int64"]


@pytest.mark.fast
def test_complex_xlsx_loading() -> None:
    df, meta = load_excel(COMPLEX_XLSX)
    assert (
        meta
        == "--outdir ./results --target y --mode classify --categoricals s,x0 --nan median --norm robust --classifiers knn lgbm dummy --feat-select filter embed"
    )
    assert list(df.columns) == ["s", "x0", "x1", "x2", "x3", "y"]
    dtypes = list(df.dtypes)
    assert dtypes == ["object", "float64", "float64", "float64", "float64", "int64"]

    df, meta = load_excel(COMPLEX_XLSX2)
    assert (
        meta
        == "--outdir ./results --target y --mode classify --categoricals s,x0 --nan mean --norm minmax --classifiers knn lgbm dummy --feat-select filter wrap"
    )
    assert list(df.columns) == ["s", "x0", "x1", "x2", "x3", "y"]
    dtypes = list(df.dtypes)
    assert dtypes == ["object", "float64", "float64", "float64", "float64", "int64"]


@pytest.mark.fast
def test_csv_loading() -> None:
    df, meta = load_csv(SIMPLE_CSV)
    assert meta == "--target y --"
    assert list(df.columns) == ["x1", "x2", "x3", "y"]
    dtypes = list(df.dtypes)
    assert dtypes == ["float64", "float64", "float64", "int64"]


@pytest.mark.fast
def test_csv_loading2() -> None:
    df, meta = load_csv(SIMPLE_CSV2)
    assert meta == "--target y"
    assert list(df.columns) == ["x1", "x2", "x3", "y"]
    dtypes = list(df.dtypes)
    assert dtypes == ["float64", "float64", "float64", "int64"]
