from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from math import ceil

import numpy as np
import pytest
from _pytest.capture import CaptureFixture
from pandas import DataFrame

from src._constants import ELDER_DATA, ELDER_TYPES, MUSHROOM_DATA, MUSHROOM_TYPES
from src.cli.cli import get_options
from src.enumerables import NanHandling
from src.preprocessing.cleaning import (
    encode_categoricals,
    get_unq_counts,
    handle_continuous_nans,
    inspect_str_columns,
    load_as_df,
    remove_timestamps,
)
from src.preprocessing.inspection import inspect_str_columns
from src.testing.datasets import TEST_DATASETS, fake_data


def no_cats(df: DataFrame, target: str) -> bool:
    return df.drop(columns=target).select_dtypes(include=["object", "string[python]"]).shape[1] == 0


def test_na_handling() -> None:
    for dsname, ds in TEST_DATASETS.items():
        df = ds.load()
        cats = ds.categoricals
        X_cats = df.loc[:, ["target", *cats]]
        cat_nan_idx = X_cats.isna().to_numpy()

        for nans in [NanHandling.Mean, NanHandling.Median]:
            dfc = handle_continuous_nans(df, target="target", cat_cols=cats, nans=nans)
            clean = dfc.drop(columns=["target", *cats])
            assert clean.isna().sum().sum() == 0, f"NaNs remaning in data {dsname}"

            # Check that categorical and target NaNs unaffected
            X_cat_clean = dfc.loc[:, ["target", *cats]]
            cat_nan_idx_clean = X_cat_clean.isna().to_numpy()
            np.testing.assert_equal(cat_nan_idx, cat_nan_idx_clean)

        for nans in [NanHandling.Drop]:
            dfc = handle_continuous_nans(df, target="target", cat_cols=cats, nans=nans)
            clean = dfc.drop(columns=["target", *cats])
            assert clean.isna().sum().sum() == 0, f"NaNs remaning in data {dsname}"


def test_multivariate_interpolate(capsys: CaptureFixture) -> None:
    for dsname, ds in TEST_DATASETS.items():
        if dsname in ["community_crime", "news_popularity"]:
            continue  # extremely slow
        df = ds.load()
        cats = ds.categoricals
        X_cats = df.loc[:, ["target", *cats]]
        cat_nan_idx = X_cats.isna().to_numpy()

        nans = NanHandling.Impute
        with capsys.disabled():
            print(f"Performing multivariate imputation for data: {dsname}")
        dfc = handle_continuous_nans(df, target="target", cat_cols=cats, nans=nans)
        clean = dfc.drop(columns=["target", *cats])
        assert clean.isna().sum().sum() == 0, f"NaNs remaning in data {dsname}"

        # Check that categorical and target NaNs unaffected
        X_cat_clean = dfc.loc[:, ["target", *cats]]
        cat_nan_idx_clean = X_cat_clean.isna().to_numpy()
        np.testing.assert_equal(cat_nan_idx, cat_nan_idx_clean)


def test_encode() -> None:
    for dsname, ds in TEST_DATASETS.items():
        df = ds.load()
        cats = ds.categoricals
        enc = encode_categoricals(df, target="target", categoricals=cats, warn_sus=False)[0]
        assert no_cats(enc, target="target"), f"Found categoricals remaining for {dsname}"


def test_encode_warn() -> None:
    for dsname, ds in TEST_DATASETS.items():
        if dsname not in ["community_crime"]:
            continue
        df = ds.load()
        cats = ds.categoricals
        with pytest.warns(UserWarning, match="Found string-valued"):
            enc = encode_categoricals(df, target="target", categoricals=cats, warn_sus=True)[0]
        assert no_cats(enc, target="target"), f"Found categoricals remaining for {dsname}"


def test_timestamp_detection() -> None:
    for dsname, ds in TEST_DATASETS.items():
        if dsname not in ["elder"]:
            continue
        df = ds.load()
        with pytest.warns(UserWarning, match="A significant proportion"):
            enc, dropped = remove_timestamps(df, target="target")


def test_str_continuous_warn() -> None:
    for dsname, ds in TEST_DATASETS.items():
        if dsname not in ["community_crime"]:
            continue
        df = ds.load()
        X = df.drop(columns="target")
        dtypes = ["object", "string[python]"]
        cols = X.select_dtypes(include=dtypes).columns.tolist()

        # with pytest.warns(UserWarning, match=".*converted into floating.*"):
        inspect_str_columns(df, str_cols=cols)


def test_detect_ordinal() -> None:
    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(50)], size=500, replace=True),
        columns=["ints"],
    ).astype(int)
    str_cols = ["ints"]
    with pytest.warns(UserWarning, match=".*Columns that may be ordinal.*"):
        ords = inspect_str_columns(df, str_cols=str_cols)[1]
    assert "ints" in ords


def test_detect_probably_ordinal() -> None:
    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(30000)], size=200, replace=False),
        columns=["ints"],
    ).astype(int)
    with pytest.warns(UserWarning):
        ords, ids = inspect_str_columns(df, str_cols=str_cols)[1:]
    assert "ints" in ords
    assert "ints" in ids
    assert "All unique values in large range" in ords["ints"]
    assert "All values including possible NaNs" in ids["ints"]


def test_detect_heuristically_ordinal() -> None:
    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(7)], size=10, replace=True),
        columns=["ints"],
    ).astype(int)
    with pytest.warns(UserWarning):
        ords = inspect_str_columns(df, str_cols=["ints"])[1]
    assert "ints" in ords
    assert "common 0-indexed Likert" in ords["ints"], ords["ints"]

    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(8)], size=10, replace=True),
        columns=["ints"],
    ).astype(int)
    with pytest.warns(UserWarning):
        ords = inspect_str_columns(df, str_cols=["ints"])[1]
    assert "ints" in ords
    assert "common Likert" in ords["ints"], ords["ints"]

    rng = np.random.default_rng(68)
    df = DataFrame(
        data=rng.choice([*range(101)], size=100, replace=True),
        columns=["ints"],
    ).astype(int)
    with pytest.warns(UserWarning):
        ords = inspect_str_columns(df, str_cols=["ints"])[1]
    assert "ints" in ords
    assert "common scale max" in ords["ints"], ords["ints"]

    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(101)], size=100, replace=True),
        columns=["ints"],
    ).astype(int)
    with pytest.warns(UserWarning):
        ords = inspect_str_columns(df, str_cols=["ints"])[1]
    assert "ints" in ords
    assert "common 0-indexed scale max" in ords["ints"], ords["ints"]


def test_detect_ids() -> None:
    df = DataFrame(
        data=np.random.choice([*range(1000)], size=100, replace=False),
        columns=["ints"],
    )
    str_cols = ["ints"]
    with pytest.warns(UserWarning, match=".*Columns that likely are identifiers.*"):
        ids = inspect_str_columns(df, str_cols=str_cols)[2]
    assert "ints" in ids


if __name__ == "__main__":
    for dsname, ds in TEST_DATASETS.items():
        if dsname != "community_crime":
            continue
        df = ds.load()
        X = df.drop(columns="target")
        dtypes = ["object", "string[python]"]
        unqs = get_unq_counts(df, "target")
        str_cols = X.select_dtypes(include=dtypes).columns.tolist()

        print(f"Inspecting string/object columns of {dsname}")
        inspect_str_columns(df, str_cols=str_cols)
        # input("Continue?")
