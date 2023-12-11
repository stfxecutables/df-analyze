from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from contextlib import redirect_stderr
from io import StringIO
from pprint import pprint
from shutil import get_terminal_size
from sys import stderr
from time import perf_counter

import numpy as np
import pytest
from pandas import DataFrame

from src.preprocessing.inspection.containers import InspectionInfo, InspectionResults
from src.preprocessing.inspection.inspection import inflation, inspect_data
from src.testing.datasets import (
    FAST_INSPECTION,
    TestDataset,
    all_ds,
    fast_ds,
    med_ds,
    slow_ds,
)

if __name__ == "__main__":
    dsname: str
    ds: TestDataset
    for dsname, ds in FAST_INSPECTION.items():
        df = ds.load()
        with redirect_stderr(StringIO()):
            info = inspect_data(df=df, target="target", categoricals=ds.categoricals, _warn=False)
            info.print_basic_infos()
            print(info.basic_df())
            print()

        # b = basics = user_info.basic_df()
        # b = b[~b.reason.str.contains("String|Binary|numeric|(?:Single value)")]
        # print("=" * 81)
        # print(dsname)
        # if len(b) > 0:
        #     df_sus = df[b.feature_name.to_list()]
        #     print(df_sus)
        #     desc = df_sus.describe().T
        #     desc["perc_inflated"] = df_sus.apply(inflation)
        #     print(desc)
        #     print(b.to_markdown(tablefmt="simple", index=False))
        # else:
        #     print("No issues")

        # input()
