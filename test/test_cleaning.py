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
from src.preprocessing.inspection import (
    get_str_cols,
    get_unq_counts,
    inspect_data,
    inspect_str_columns,
)
from src.testing.datasets import TEST_DATASETS, TestDataset


def no_cats(df: DataFrame, target: str) -> bool:
    return df.drop(columns=target).select_dtypes(include=["object", "string[python]"]).shape[1] == 0


@pytest.mark.parametrize("dataset", [*TEST_DATASETS.items()], ids=lambda pair: str(pair[0]))
def test_na_handling(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    cats = ds.categoricals
    X_cats = df.loc[:, ["target", *cats]]
    cat_nan_idx = X_cats.isna().to_numpy()

    for nans in [NanHandling.Mean, NanHandling.Median]:
        dfc = handle_continuous_nans(df, target="target", categoricals=cats, nans=nans)
        clean = dfc.drop(columns=["target", *cats])
        assert clean.isna().sum().sum() == 0, f"NaNs remaning in data {dsname}"

        # Check that categorical and target NaNs unaffected
        X_cat_clean = dfc.loc[:, ["target", *cats]]
        cat_nan_idx_clean = X_cat_clean.isna().to_numpy()
        np.testing.assert_equal(cat_nan_idx, cat_nan_idx_clean)

    for nans in [NanHandling.Drop]:
        dfc = handle_continuous_nans(df, target="target", categoricals=cats, nans=nans)
        clean = dfc.drop(columns=["target", *cats])
        assert clean.isna().sum().sum() == 0, f"NaNs remaning in data {dsname}"


@pytest.mark.parametrize("dataset", [*TEST_DATASETS.items()], ids=lambda pair: str(pair[0]))
def test_multivariate_interpolate(dataset: tuple[str, TestDataset], capsys: CaptureFixture) -> None:
    dsname, ds = dataset
    if dsname in ["community_crime", "news_popularity"]:
        return  # extremely slow

    df = ds.load()
    cats = ds.categoricals
    X_cats = df.loc[:, ["target", *cats]]
    cat_nan_idx = X_cats.isna().to_numpy()

    nans = NanHandling.Impute
    # with capsys.disabled():
    #     print(f"Performing multivariate imputation for data: {dsname}")
    with pytest.warns(UserWarning, match="Using experimental multivariate"):
        dfc = handle_continuous_nans(
            df,
            target="target",
            categoricals=cats,
            nans=nans,
        )

    clean = dfc.drop(columns=["target", *cats])
    assert clean.isna().sum().sum() == 0, f"NaNs remaning in data {dsname}"

    # Check that categorical and target NaNs unaffected
    X_cat_clean = dfc.loc[:, ["target", *cats]]
    cat_nan_idx_clean = X_cat_clean.isna().to_numpy()
    np.testing.assert_equal(cat_nan_idx, cat_nan_idx_clean)


@pytest.mark.parametrize("dataset", [*TEST_DATASETS.items()], ids=lambda pair: str(pair[0]))
def test_encode(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    cats = ds.categoricals
    try:
        results = inspect_data(df, "target", cats)
        enc = encode_categoricals(
            df, target="target", results=results, categoricals=cats, ordinals=[]
        )[0]
    except Exception as e:
        raise ValueError(f"Could not encode categoricals for data: {dsname}") from e
    assert no_cats(enc, target="target"), f"Found categoricals remaining for {dsname}"


def test_timestamp_detection() -> None:
    for dsname, ds in TEST_DATASETS.items():
        if dsname not in ["elder"]:
            continue
        df = ds.load()
        str_cols = get_str_cols(df, "target")
        times = inspect_str_columns(df, str_cols, ds.categoricals, ordinals=[], _warn=False)[3]
        assert len(times) == 1


def test_str_continuous_warn() -> None:
    for dsname, ds in TEST_DATASETS.items():
        if dsname not in ["community_crime"]:
            continue
        df = ds.load()
        X = df.drop(columns="target")
        dtypes = ["object", "string[python]"]
        cols = X.select_dtypes(include=dtypes).columns.tolist()

        # with pytest.warns(UserWarning, match=".*converted into floating.*"):
        inspect_str_columns(
            df, str_cols=cols, categoricals=ds.categoricals, ordinals=[], _warn=False
        )


def test_detect_ordinal() -> None:
    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(50)], size=500, replace=True),
        columns=["ints"],
    ).astype(int)
    str_cols = ["ints"]
    other: dict = dict(categoricals=[], ordinals=[])
    ords = inspect_str_columns(df, str_cols=str_cols, **other)[1]
    assert "ints" in ords


def test_detect_probably_ordinal() -> None:
    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(30000)], size=200, replace=False),
        columns=["ints"],
    ).astype(int)
    other: dict = dict(categoricals=[], ordinals=[])
    ords, ids = inspect_str_columns(df, str_cols=["ints"], **other)[1:3]
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
    other: dict = dict(categoricals=[], ordinals=[])
    ords = inspect_str_columns(df, str_cols=["ints"], **other)[1]
    assert "ints" in ords
    assert "common 0-indexed Likert" in ords["ints"], ords["ints"]

    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(8)], size=10, replace=True),
        columns=["ints"],
    ).astype(int)
    other: dict = dict(categoricals=[], ordinals=[])
    ords = inspect_str_columns(df, str_cols=["ints"], **other)[1]
    assert "ints" in ords
    assert "common Likert" in ords["ints"], ords["ints"]

    rng = np.random.default_rng(68)
    df = DataFrame(
        data=rng.choice([*range(101)], size=100, replace=True),
        columns=["ints"],
    ).astype(int)
    other: dict = dict(categoricals=[], ordinals=[])
    ords = inspect_str_columns(df, str_cols=["ints"], **other)[1]
    assert "ints" in ords
    assert "common scale max" in ords["ints"], ords["ints"]

    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(101)], size=100, replace=True),
        columns=["ints"],
    ).astype(int)
    other: dict = dict(categoricals=[], ordinals=[])
    ords = inspect_str_columns(df, str_cols=["ints"], **other)[1]
    assert "ints" in ords
    assert "common 0-indexed scale max" in ords["ints"], ords["ints"]


def test_detect_ids() -> None:
    df = DataFrame(
        data=np.random.choice([*range(1000)], size=100, replace=False),
        columns=["ints"],
    )
    str_cols = ["ints"]
    other: dict = dict(categoricals=[], ordinals=[])
    ids = inspect_str_columns(df, str_cols=str_cols, **other)[2]
    assert "ints" in ids

    for dsname, ds in TEST_DATASETS.items():
        if dsname not in ["community_crime"]:
            continue
        df = ds.load()
        cats = ds.categoricals
        try:
            results = inspect_data(df, "target", cats)
            assert "communityname" in results.ids
        except Exception as e:
            raise ValueError("Identifier 'communityname' was not detected") from e


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
        results = inspect_data(df, "target", ds.categoricals)
        encode_categoricals(df, "target", results, ds.categoricals, [])
        print("#" * w, file=stderr)
        # input("Continue?")
