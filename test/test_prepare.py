from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pprint import pprint

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from src.preprocessing.cleaning import prepare_data
from src.preprocessing.inspection.inspection import (
    inspect_data,
)
from src.testing.datasets import (
    FAST_INSPECTION,
    MEDIUM_INSPECTION,
    SLOW_INSPECTION,
    TestDataset,
    fast_ds,
    med_ds,
    slow_ds,
)


def do_prepare(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    cats = ds.categoricals

    try:
        results = inspect_data(df, "target", cats)
        prepare_data(
            df=df,
            target="target",
            results=results,
            is_classification=ds.is_classification,
            _warn=False,
        )
    except ValueError as e:
        if dsname == "credit-approval_reduced":
            message = str(e)
            assert "Target" in message
            assert "is constant" in message
    except Exception as e:
        raise ValueError(f"Could not prepare data: {dsname}") from e


@fast_ds
def test_prepare_fast(dataset: tuple[str, TestDataset]) -> None:
    do_prepare(dataset)


@med_ds
def test_prepare_med(dataset: tuple[str, TestDataset]) -> None:
    do_prepare(dataset)


@slow_ds
def test_prepare_slow(dataset: tuple[str, TestDataset]) -> None:
    do_prepare(dataset)


if __name__ == "__main__":
    tinfos = []
    for dsname, ds in tqdm(SLOW_INSPECTION):
        df = ds.load()
        cats = ds.categoricals

        try:
            sink = StringIO()
            with redirect_stderr(sink):
                with redirect_stdout(sink):
                    results = inspect_data(df, "target", cats)
            X, y, X_cat, info = prepare_data(
                df=df,
                target="target",
                results=results,
                is_classification=ds.is_classification,
                _warn=False,
            )
            funcs, times = [*zip(*info["runtimes"].items())]
            tinfo = DataFrame(data=[times], columns=funcs, index=[dsname])
            tinfos.append(tinfo)

        except ValueError as e:
            if dsname in ["credit-approval_reduced", "credit-approval_reproduced"]:
                message = str(e)
                assert "is constant" in message
            else:
                raise ValueError(f"Error for {dsname}") from e
    df = pd.concat(tinfos, axis=0, ignore_index=False)
    out = ROOT / "perf_times.parquet"
    df.to_parquet(out)
    print(f"Saved timings to {out}")
    print(df.max().sort_values(ascending=False))

"""
>>> df.max().sort_values(ascending=False)  # FAST
encode_categoricals        0.427563
unify_nans                 0.162584
deflate_categoricals       0.041391
inspect_target             0.026867
convert_categoricals       0.022642
drop_target_nans           0.018817
encode_target              0.007125
clean_regression_target    0.006605
drop_unusable              0.003869

>>> df.max().sort_values(ascending=False)  # MEDIUM
encode_categoricals        1.712539
unify_nans                 1.679253
inspect_target             0.102524
convert_categoricals       0.057981
encode_target              0.050567
deflate_categoricals       0.045729
drop_target_nans           0.034897
drop_unusable              0.007253
clean_regression_target    0.006596

>>> df.max().sort_values(ascending=False)  # SLOW
encode_categoricals        36.807302
unify_nans                 18.126168
encode_target               1.710700
deflate_categoricals        0.603854
convert_categoricals        0.364583
drop_target_nans            0.242952
clean_regression_target     0.110536
inspect_target              0.055626
drop_unusable               0.043274
"""
