import traceback
from os import PathLike
from pathlib import Path
from typing import Union
from warnings import warn

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.io import loadmat

from src._constants import CLEAN_JSON, DATA_JSON, DATAFILE
from src.cli.cli import CleaningOptions


def load_as_df(path: Path) -> Union[DataFrame, ndarray]:
    FILETYPES = [".json", ".csv", ".npy"]
    if path.suffix not in FILETYPES:
        raise ValueError("Invalid data file. Currently must be one of ")
    if path.suffix == ".json":
        df = pd.read_json(str(path))
    elif path.suffix == ".csv":
        df = pd.read_csv(str(path))
    elif path.suffix == ".npy":
        arr: ndarray = np.load(str(path), allow_pickle=False)
        if arr.ndim != 2:
            raise RuntimeError(
                f"Invalid NumPy data in {path}. NumPy array must be two-dimensional."
            )
        cols = [f"c{i}" for i in range(arr.shape[1])]
        df = DataFrame(data=arr, columns=cols)
    else:
        raise RuntimeError("Unreachable!")
    # pd.get_dummies is idempotent-ish so below is safe-ish
    dummified = pd.get_dummies(df, drop_first=False)
    return dummified


def remove_nan_features(df: DataFrame, target: str) -> DataFrame:
    """Remove columns (features) that are ALL NaN"""
    try:
        y = df[target].to_numpy()
    except Exception as e:
        trace = traceback.format_exc()
        raise ValueError(f"Could not convert target column to NumPy:\n{trace}") from e
    df = df.drop(columns=target)
    df = df.dropna(axis=1, how="any")
    df[target] = y
    return df


def remove_nan_samples(df: DataFrame) -> DataFrame:
    """Remove rows (samples) that have ANY NaN"""
    return df.dropna(axis=0, how="any").dropna(axis=0, how="any")


# this is usually really fast, no real need to memoize probably
# @MEMOIZER.cache
def get_clean_data(options: CleaningOptions) -> DataFrame:
    """Perform minimal cleaning, like removing NaN features"""
    df = load_as_df(options.datapath)
    print("Shape before dropping:", df.shape)
    if options.drop_nan in ["all", "rows"]:
        df = remove_nan_samples(df)
    if options.drop_nan in ["all", "cols"]:
        df = remove_nan_features(df, target=options.target)
    print("Shape after dropping:", df.shape)
    if df[options.target].isnull().any():
        warn(
            f"DataFrame has NaN values in target variable: {options.target}. "
            "Currently, most/all classifiers and regressors do not support this. ",
            category=UserWarning,
        )
    return df


# https://www.statsmodels.org/stable/imputation.html
