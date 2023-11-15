from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from src._constants import SIMPLE_CSV, SIMPLE_ODS, SIMPLE_XLSX
from src.loading import load_excel


def test_xlsx_loading() -> None:
    df, meta = load_excel(SIMPLE_XLSX)
    assert meta[0] == "--target y"
    assert meta[1] == "--"
    assert list(df.columns) == ["x1", "x2", "x3", "y"]
    dtypes = list(df.dtypes)
    assert dtypes == ["float64", "float64", "float64", "int64"]
