from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from shutil import get_terminal_size
from sys import stderr

import numpy as np
import pytest
from _pytest.capture import CaptureFixture
from pandas import DataFrame

from src.enumerables import NanHandling
from src.preprocessing.cleaning import (
    encode_categoricals,
    handle_continuous_nans,
)
from src.preprocessing.inspection.inspection import (
    get_unq_counts,
)
from src.testing.datasets import (
    TEST_DATASETS,
    TestDataset,
    all_ds,
    fast_ds,
    med_ds,
    slow_ds,
)


def no_cats(df: DataFrame, target: str) -> bool:
    return df.drop(columns=target).select_dtypes(include=["object", "string[python]"]).shape[1] == 0


@fast_ds
@pytest.mark.cached
def test_na_handling(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    results = ds.inspect(load_cached=True)
    cats = [*results.cats.infos.keys(), *results.binaries.infos.keys()]
    X_cats = df.loc[:, cats]
    # remember NaNs in target cause dropping in handle_cont_nans below
    # X_cats = X_cats[~X_cats["target"].isna()].drop(columns="target")
    cat_nan_idx = X_cats.isna().to_numpy()

    for nans in [NanHandling.Mean, NanHandling.Median]:
        dfc = handle_continuous_nans(df, target="target", results=results, nans=nans)[0]
        clean = dfc.drop(columns=["target", *cats])
        assert clean.isna().sum().sum() == 0, f"NaNs remaning in data {dsname}"

        # Check that categorical and target NaNs unaffected
        X_cat_clean = dfc.loc[:, cats]
        cat_nan_idx_clean = X_cat_clean.isna().to_numpy()
        np.testing.assert_equal(cat_nan_idx, cat_nan_idx_clean)

    for nans in [NanHandling.Drop]:
        try:
            dfc = handle_continuous_nans(df, target="target", results=results, nans=nans)[0]
            clean = dfc.drop(columns=["target", *cats])
            assert clean.isna().sum().sum() == 0, f"NaNs remaining in data {dsname}"
        except RuntimeError as e:
            if dsname not in [
                "dermatology",
                "colic",
                "colleges",
                "Traffic_violations",
                "hypothyroid",
                "Midwest_Survey_nominal",
            ]:
                raise e


@all_ds
@pytest.mark.cached
@pytest.mark.fast
def test_multivariate_interpolate(dataset: tuple[str, TestDataset], capsys: CaptureFixture) -> None:
    dsname, ds = dataset
    if dsname in ["community_crime", "news_popularity"]:
        return  # extremely slow
    if dsname in [
        "analcatdata_marketing",
        "analcatdata_reviewer",
        "internet_usage",
        "ipums_la_97-small",
        "kdd_internet_usage",
        "Mercedes_Benz_Greener_Manufacturing",
        "Midwest_Survey_nominal",
        "Midwest_Survey",
        "Midwest_survey2",
        "ozone_level",
        "primary-tumor",
        "soybean",
        "vote",
    ]:
        return  # nothing to impute, no continuous

    df = ds.load()
    results = ds.inspect(load_cached=True)
    cats = [*results.cats.infos.keys(), *results.binaries.infos.keys()]
    X_cats = df.loc[:, cats]
    if not X_cats.empty:
        n_cont = df.shape[1] - X_cats.shape[1]
    else:
        n_cont = df.shape[1] - 1
    cat_nan_idx = X_cats.isna().to_numpy()
    if n_cont > 20:
        return

    results = ds.inspect(load_cached=True)
    with pytest.warns(UserWarning, match="Using experimental multivariate"):
        dfc = handle_continuous_nans(
            df,
            target="target",
            results=results,
            nans=NanHandling.Impute,
        )[0]

    clean = dfc.drop(columns=["target", *cats])
    assert clean.isna().sum().sum() == 0, f"NaNs remaning in data {dsname}"

    # Check that categorical and target NaNs unaffected
    X_cat_clean = dfc.loc[:, cats]
    cat_nan_idx_clean = X_cat_clean.isna().to_numpy()
    np.testing.assert_equal(cat_nan_idx, cat_nan_idx_clean)


def do_encode(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    results = ds.inspect(load_cached=True)
    try:
        enc = encode_categoricals(df, target="target", results=results)[0]
    except TypeError as e:
        if dsname == "community_crime" and (
            "Cannot automatically determine the cardinality" in str(e)
        ):
            return
        raise e
    except Exception as e:
        raise ValueError(f"Could not encode categoricals for data: {dsname}") from e
    assert no_cats(enc, target="target"), f"Found categoricals remaining for {dsname}"


@fast_ds
@pytest.mark.cached
def test_encoding_fast(dataset: tuple[str, TestDataset]) -> None:
    do_encode(dataset)


@med_ds
@pytest.mark.cached
def test_encoding_med(dataset: tuple[str, TestDataset]) -> None:
    do_encode(dataset)


@slow_ds
@pytest.mark.cached
def test_encoding_slow(dataset: tuple[str, TestDataset]) -> None:
    do_encode(dataset)


if __name__ == "__main__":
    for dsname, ds in TEST_DATASETS.items():
        # if dsname != "forest_fires":
        #     continue
        df = ds.load()
        X = df.drop(columns="target")
        dtypes = ["object", "string[python]"]
        unqs = get_unq_counts(df, "target")
        str_cols = X.select_dtypes(include=dtypes).columns.tolist()

        # print(f"Inspecting string/object columns of {dsname}")
        # inspect_str_columns(df, str_cols=str_cols, categoricals=ds.categoricals)
        w = get_terminal_size((81, 24))[0]
        print("#" * w, file=stderr)
        print(f"Checking {dsname}", file=stderr)
        results = ds.inspect(load_cached=True)
        encode_categoricals(df, "target", results)
        print("#" * w, file=stderr)
        # input("Continue?")
