from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from contextlib import redirect_stderr
from io import StringIO

from src.preprocessing.inspection.inspection import inspect_data
from src.testing.datasets import (
    FAST_INSPECTION,
    TestDataset,
)

REPORTS = ROOT / "results/reports"
REPORTS.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    dsname: str
    ds: TestDataset
    for dsname, ds in FAST_INSPECTION.items():
        df = ds.load()
        with redirect_stderr(StringIO()):
            info = inspect_data(df=df, target="target", categoricals=ds.categoricals, _warn=False)
            report = info.short_report(pad=81)
            out = REPORTS / f"{dsname}.txt"
            out.write_text(report)
            print(f"Saved short report to {out}")

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
