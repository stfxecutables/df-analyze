import traceback
from math import ceil
from os import PathLike
from pathlib import Path
from typing import Union
from warnings import warn

import numpy as np
import pandas as pd
from dateutil.parser import parse
from numpy import ndarray
from pandas import DataFrame
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from src._constants import CLEAN_JSON, DATA_JSON, DATAFILE
from src.cli.cli import CleaningOptions, ProgramOptions
from src.enumerables import NanHandling
from src.loading import load_spreadsheet


def normalize(df: DataFrame, target: str) -> DataFrame:
    """
    MUST be min-max normalization (e.g. for stat tests with no negatives)
    and on the categorical-encoded df.
    """
    X = df.drop(columns=target)
    X_norm = DataFrame(data=MinMaxScaler().fit_transform(X), columns=X.columns)
    X_norm[target] = df[target]
    return X_norm


def handle_nans(df: DataFrame, target: str, nans: NanHandling) -> DataFrame:
    """MUST be on the categorical-encoded df"""
    # drop rows where target is NaN: meaningless
    idx = ~df[target].isna()
    df = df.loc[idx]
    X = df.drop(columns=target)
    y = df[target]

    if nans is NanHandling.Drop:
        X_clean = X.dropna(axis="columns").dropna(axis="index")
        if 0 in X_clean.shape:
            others = [na.value for na in NanHandling if na is not NanHandling.Drop]
            raise RuntimeError(
                "Dropping NaNs resulted in either no remaining samples or features. "
                "Consider changing the option `--nan` to another valid option. Other "
                f"valid options: {others}"
            )

    elif nans in [NanHandling.Mean, NanHandling.Median]:
        strategy = "mean" if nans is NanHandling.Mean else "median"
        X_clean = DataFrame(
            data=SimpleImputer(strategy=strategy).fit_transform(X), columns=X.columns
        )
    elif nans is NanHandling.Impute:
        imputer = IterativeImputer()
        X_clean = DataFrame(data=imputer.fit_transform(X), columns=X.columns)
    else:
        raise NotImplementedError(f"Unhandled enum case: {nans}")

    X_clean[target] = y
    return X_clean


def encode_target():
    ...


def encode_categoricals(
    df: DataFrame, target: str, categoricals: Union[list[str], int]
) -> DataFrame:
    """Treat all features with <= options.cat_threshold as categorical"""
    X = df.drop(columns=target).infer_objects().convert_dtypes()
    unique_counts = {}
    for colname in X.columns:
        try:
            unique_counts[colname] = len(np.unique(df[colname]))
        except TypeError:  # happens when can't sort for unique
            unique_counts[colname] = len(np.unique(df[colname].astype(str)))
    str_cols = X.select_dtypes(include=["object", "string[python]"]).columns.tolist()
    int_cols = X.select_dtypes(include="int").columns.to_list()
    sus_ints = [col for col in int_cols if unique_counts[col] < 5]

    cats = categoricals
    auto = isinstance(cats, int)
    # if no threshold specified, only convert what has to be converted
    threshold = cats if auto else -1
    if auto:
        to_convert = [col for col, cnt in unique_counts.items() if cnt <= threshold]
        to_convert += str_cols
        new = pd.get_dummies(X, columns=to_convert, dummy_na=True, dtype=float)
        new = new.astype(float)
        new[target] = df[target]
        return new

    # warn the user if some int columns look sus, and check for unspecified
    # categoricals
    assert isinstance(cats, list)
    cats = set(cats)
    unspecified = sorted(set(str_cols).difference(cats))
    if len(unspecified) > 0:
        warn(
            "Found string-valued features not specified with `--categoricals` "
            "argument. These will be one-hot encoded to allow use in subsequent "
            "analyses. To silence this warning, specify the categoricals "
            "manually either via the CLI or in the spreadsheet file header."
            f"Unspecified string-valued features: {unspecified}"
        )
    sus_cols = sorted(set(sus_ints).difference(cats))
    if len(sus_cols) > 0:
        warn(
            "Found integer-valued features not specified with `--categoricals` "
            "argument, and which have less than 5 levels (classes) each. "
            "These features may in fact be categoricals, and if so, should be "
            "specified as such manually either via the CLI or in the "
            "spreadsheet file header. Otherwise, they will be left as is, and "
            "treated as ordinal / continuous. Categorical-like integer-valued "
            f"features: {sus_cols}"
        )

    to_convert = list(set(str_cols).union(cats))
    new = pd.get_dummies(X, columns=to_convert, dummy_na=True, dtype=float)
    new = new.astype(float)
    new[target] = df[target]
    return new


def is_timelike(s: str) -> bool:
    # https://stackoverflow.com/a/25341965 for this...
    try:
        parse(s, fuzzy=False)
        return True
    except ValueError:
        return False


def detect_timestamps(df: DataFrame, target: str) -> None:
    n_subsamp = max(ceil(0.5 * len(df)), 500)
    n_subsamp = min(n_subsamp, len(df))
    # time checks can be very slow, so just check a few for each feature first
    X = df.drop(columns=target).infer_objects().select_dtypes(include="object").dropna(axis="index")
    for col in X.columns:
        idx = np.random.permutation(len(X))[:n_subsamp]
        percent = X[col].loc[idx].apply(is_timelike).sum() / n_subsamp
        if percent > 1.0 / 3.0:
            p = X[col].loc[idx].apply(is_timelike).mean()
            if p > 0.3:
                raise ValueError(
                    f"A significant proportion ({p*100:02f}%) of the data for feature "
                    f"`{col}` appears to be parseable as datetime data. Datetime data "
                    "cannot currently be handled by `df-analyze` (or most AutoML or "
                    "or most automated predictive approaches) due to special requirements "
                    "in data preprocessing (e.g. Fourier features), splitting (e.g. time-"
                    "based cross-validation, forecasting, hindcasting) and in the models "
                    "used (e.g. ARIMA, VAR, etc.).\n\n"
                    f"To remove this error, either DELETE the `{col}` column from your data, "
                    "or manually edit the column values so they are clearly interpretable "
                    "as a categorical variable."
                )


def load_as_df(path: Path, spreadsheet: bool) -> DataFrame:
    FILETYPES = [".json", ".csv", ".npy", "xlsx", ".parquet"]
    if path.suffix not in FILETYPES:
        raise ValueError(f"Invalid data file. Currently must be one of: {FILETYPES}")
    if path.suffix == ".json":
        df = pd.read_json(str(path))
    elif path.suffix == ".csv":
        if spreadsheet:
            df = load_spreadsheet(path)[0]
        else:
            df = pd.read_csv(str(path))
    elif path.suffix == ".parquet":
        df = pd.read_parquet(str(path))
    elif path.suffix == ".xlsx":
        if spreadsheet:
            df = load_spreadsheet(path)[0]
        else:
            df = pd.read_excel(str(path))
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
    return df


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
def get_clean_data(options: ProgramOptions) -> DataFrame:
    """Perform minimal cleaning, like removing NaN features"""
    df = load_as_df(options.datapath, options.is_spreadsheet)
    print("Shape before dropping:", df.shape)
    if options.nan_handling in ["all", "rows"]:
        df = remove_nan_samples(df)
    if options.nan_handling in ["all", "cols"]:
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
