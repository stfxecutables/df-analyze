from math import ceil

import numpy as np
import pytest
from _pytest.capture import CaptureFixture
from pandas import DataFrame

from src._constants import ELDER_DATA, ELDER_TYPES, MUSHROOM_DATA, MUSHROOM_TYPES, ROOT
from src.cli.cli import get_options
from src.preprocessing.cleaning import (
    detect_timestamps,
    encode_categoricals,
    get_clean_data,
    handle_nans,
    load_as_df,
    normalize,
)


def no_cats(df: DataFrame, target: str) -> bool:
    return df.drop(columns=target).select_dtypes(include=["object", "string[python]"]).shape[1] == 0


def test_drop() -> None:
    for data, target in [(MUSHROOM_DATA, "target"), (ELDER_DATA, "temperature")]:
        options = get_options(f"--df {data} --target {target} --nan drop")
        df = load_as_df(data, spreadsheet=False)
        clean = handle_nans(df, target=options.target, nans=options.nan_handling)
        assert clean.isna().sum().sum() == 0


@pytest.mark.parametrize("method", ["mean", "median"])
def test_interpolate(method: str) -> None:
    datas = (MUSHROOM_DATA, ELDER_DATA)
    types = (MUSHROOM_TYPES, ELDER_TYPES)
    targets = ("target", "temperature")
    for data, typ, target in zip(datas, types, targets):
        df = load_as_df(data, spreadsheet=False)
        dft = load_as_df(typ, spreadsheet=False)
        cats = dft[dft["type"] == "categorical"]["feature_name"].to_list()
        cat_arg = " ".join(cats)
        options = get_options(
            f"--df {data} --target {target} --nan {method} --categoricals {cat_arg}"
        )

        df = encode_categoricals(df, target=options.target, categoricals=cats)[0]
        clean = handle_nans(df, target=options.target, nans=options.nan_handling)
        assert clean.isna().sum().sum() == 0


def test_multivariate_interpolate(capsys: CaptureFixture) -> None:
    datas = (MUSHROOM_DATA, ELDER_DATA)
    types = (MUSHROOM_TYPES, ELDER_TYPES)
    targets = ("target", "temperature")
    for data, typ, target in zip(datas, types, targets):
        df = load_as_df(data, spreadsheet=False)
        n_max = min(50, len(df))
        idx = np.random.permutation(len(df))[:n_max]
        df = df.loc[idx]

        dft = load_as_df(typ, spreadsheet=False)
        cats = dft[dft["type"] == "categorical"]["feature_name"].to_list()
        cat_arg = " ".join(cats)
        options = get_options(
            f"--df {data} --target {target} --nan impute --categoricals {cat_arg}"
        )
        with capsys.disabled():
            df = encode_categoricals(df, target=options.target, categoricals=cats)[0]
        clean = handle_nans(df, target=options.target, nans=options.nan_handling)
        assert clean.isna().sum().sum() == 0


def test_encode() -> None:
    datas = (MUSHROOM_DATA, ELDER_DATA)
    types = (MUSHROOM_TYPES, ELDER_TYPES)
    targets = ("target", "temperature")
    for data, typ, target in zip(datas, types, targets):
        df = load_as_df(data, spreadsheet=False)
        dft = load_as_df(typ, spreadsheet=False)
        cats = dft[dft["type"] == "categorical"]
        cat_arg = " ".join(cats["feature_name"].to_list())
        options = get_options(f"--df {data} --target {target} --nan drop --categoricals {cat_arg}")
        df = encode_categoricals(df, target=options.target, categoricals=options.categoricals)[0]
        assert no_cats(df, target=options.target)


def test_encode_warn() -> None:
    data = MUSHROOM_DATA
    typ = MUSHROOM_TYPES
    target = "target"

    df = load_as_df(data, spreadsheet=False)
    dft = load_as_df(typ, spreadsheet=False)
    cats = dft[dft["type"] == "categorical"]
    n = ceil(len(cats) / 2)
    some_cats = cats[:n]
    cat_arg = " ".join(some_cats["feature_name"].to_list())
    options = get_options(f"--df {data} --target {target} --nan drop --categoricals {cat_arg}")
    with pytest.warns(UserWarning):
        df = encode_categoricals(df, target=options.target, categoricals=options.categoricals)[0]
        assert no_cats(df, target=options.target)


def test_encode_auto() -> None:
    datas = (MUSHROOM_DATA, ELDER_DATA)
    targets = ("target", "temperature")

    for data, target in zip(datas, targets):
        df = load_as_df(data, spreadsheet=False)
        options = get_options(f"--df {data} --target {target} --nan drop --categoricals 3")
        df = encode_categoricals(df, target=options.target, categoricals=options.categoricals)[0]
        assert no_cats(df, target=options.target)


def test_timestamp_detection() -> None:
    df = load_as_df(MUSHROOM_DATA, spreadsheet=False)
    detect_timestamps(df, "target")

    df = load_as_df(ELDER_DATA, spreadsheet=False)
    with pytest.raises(ValueError):
        detect_timestamps(df, "temperature")
