from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from sklearn.utils.validation import check_X_y
from tqdm import tqdm

from df_analyze.testing.datasets import (
    FAST_INSPECTION,
    TestDataset,
    fast_ds,
    med_ds,
    slow_ds,
)


def do_prepare(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset

    try:
        prepared = ds.prepared(load_cached=False, force=True)
        X = prepared.X
        y = prepared.y
        check_X_y(X, y, y_numeric=True)

        assert prepared.X_cont is not None
        assert prepared.X_cat is not None

        if not prepared.X_cont.empty:
            check_X_y(prepared.X_cont, y, y_numeric=True)
        assert prepared.X_cont.shape[0] == prepared.X_cat.shape[0] == len(y)
        lens = np.array([len(X), len(y), len(prepared.X_cat), len(prepared.X_cont)])  # type: ignore
        assert np.all(lens == lens[0]), "Lengths of returned cardinality splits differ"

        assert prepared.inspection is not None, "Missing inspection data on PreparedData"
        cats = prepared.inspection.cats
        conts = prepared.inspection.conts
        if len(cats.cols.intersection(conts.cols)) > 0:
            raise RuntimeError("Inspection categorical and continuous overlap")

        X_cont, X_cat = prepared.X_cont, prepared.X_cat
        cats = set(X_cat.columns.to_list())
        conts = set(X_cont.columns.to_list())
        if len(cats.intersection(conts)) > 0:
            raise RuntimeError("Returned X_cat and X_cont overlap")

    except ValueError as e:
        if dsname in ["credit-approval_reproduced"]:
            message = str(e)
            assert "is constant" in message
        else:
            raise e
    except Exception as e:
        raise ValueError(f"Could not prepare data: {dsname}") from e


def do_prep_cached(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset

    try:
        ds.inspect(load_cached=True)
        ds.prepared(load_cached=True)
    except ValueError as e:
        if dsname == "credit-approval_reproduced":
            message = str(e)
            assert "is constant" in message
        else:
            raise e
    except Exception as e:
        raise ValueError(f"Could not prepare data: {dsname}") from e


@fast_ds
@pytest.mark.regen
def test_prepare_fast(dataset: tuple[str, TestDataset]) -> None:
    do_prepare(dataset)


@med_ds
@pytest.mark.regen
def test_prepare_med(dataset: tuple[str, TestDataset]) -> None:
    do_prepare(dataset)


@slow_ds
@pytest.mark.regen
def test_prepare_slow(dataset: tuple[str, TestDataset]) -> None:
    do_prepare(dataset)


@fast_ds
@pytest.mark.cached
def test_prep_cached_fast(dataset: tuple[str, TestDataset]) -> None:
    do_prep_cached(dataset)


@med_ds
@pytest.mark.cached
def test_prep_cached_med(dataset: tuple[str, TestDataset]) -> None:
    do_prep_cached(dataset)


@slow_ds
@pytest.mark.cached
def test_prep_cached_slow(dataset: tuple[str, TestDataset]) -> None:
    do_prep_cached(dataset)


if __name__ == "__main__":
    tinfos = []
    for dsname, ds in tqdm(FAST_INSPECTION):
        if dsname != "abalone":
            continue
        df = ds.load()
        cats = ds.categoricals

        try:
            prepared = ds.prepared(load_cached=False)
            if prepared.inspection is None:
                raise ValueError("Missing inspection data on PreparedData")

            cats = prepared.inspection.cats
            conts = prepared.inspection.conts
            if len(cats.cols.intersection(conts.cols)) > 0:
                raise RuntimeError("Inspection categorical and continuous overlap")

            X_cont, X_cat = prepared.X_cont, prepared.X_cat
            if X_cont is None:
                raise ValueError("Missing X_cont!")
            if X_cat is None:
                raise ValueError("Missing X_cat!")

            cats = set(X_cat.columns.to_list())
            conts = set(X_cont.columns.to_list())
            if len(cats.intersection(conts)) > 0:
                raise RuntimeError("Returned X_cat and X_cont overlap")

            funcs, times = [*zip(*prepared.info["runtimes"].items())]
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
