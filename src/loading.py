from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import Any

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet
from pandas import DataFrame

from src._constants import SIMPLE_CSV, SIMPLE_ODS, SIMPLE_XLSX


def load_excel(path: Path) -> Any:
    wb = load_workbook(path, data_only=True)
    sheetnames = wb.sheetnames
    if len(sheetnames) != 1:
        raise RuntimeError(
            "Found multiple sheets in Excel workbook. Make sure the Excel file "
            "has only a single sheet formatted correctly for df-analyze."
        )
    ws: Worksheet = wb[sheetnames[0]]
    row: tuple[Any]
    meta = {}
    data = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        first = row[0]
        if first is not None and str(first).strip().startswith("--"):
            line = " ".join([str(value) for value in row if value is not None])
            meta[i] = line
            continue
        if all(val is None for val in row):  # ignore blank rows
            continue
        data.append(row)

    df = DataFrame(data=data[1:], columns=data[0])

    return df, meta


if __name__ == "__main__":
    df, meta = load_excel(SIMPLE_XLSX)
    print(type(list(df.dtypes)[0]))
