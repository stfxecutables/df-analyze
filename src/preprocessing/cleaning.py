import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from shutil import get_terminal_size
from typing import Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from dateutil.parser import parse
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

from src.enumerables import NanHandling
from src.loading import load_spreadsheet
from src.preprocessing.inspection import (
    MessyDataWarning,
    inspect_int_columns,
    inspect_str_columns,
    looks_timelike,
    messy_inform,
)


class DataCleaningWarning(UserWarning):
    """For when df-analyze significantly and automatically alters input data"""

    def __init__(self, message: str) -> None:
        cols = get_terminal_size((81, 24))[0]
        sep = "=" * cols
        underline = "." * (len(self.__class__.__name__) + 1)
        self.message = f"\n{sep}\n{self.__class__.__name__}\n{underline}\n{message}\n{sep}"

    def __str__(self) -> str:
        return str(self.message)


@dataclass
class InspectionResults:
    floats: dict[str, str]
    ords: dict[str, str]
    ids: dict[str, str]
    times: dict[str, str]
    cats: dict[str, str]
    int_ords: dict[str, str]
    int_ids: dict[str, str]
    big_cats: dict[str, str]


def cleaning_inform(message: str) -> None:
    cols = get_terminal_size((81, 24))[0]
    sep = "=" * cols
    title = "Removing Data Feature"
    underline = "." * (len(title) + 1)
    message = f"\n{sep}\n{title}\n{underline}\n{message}\n{sep}"
    print(message, file=sys.stderr)


def normalize(df: DataFrame, target: str) -> DataFrame:
    """
    MUST be min-max normalization (e.g. for stat tests with no negatives)
    and on the categorical-encoded df.
    """
    X = df.drop(columns=target)
    X_norm = DataFrame(data=MinMaxScaler().fit_transform(X), columns=X.columns)
    X_norm[target] = df[target]
    return X_norm


def handle_continuous_nans(
    df: DataFrame, target: str, cat_cols: list[str], nans: NanHandling
) -> DataFrame:
    """Impute or drop nans based on values not in `cat_cols`"""
    # drop rows where target is NaN: meaningless
    idx = ~df[target].isna()
    df = df.loc[idx]
    # NaNs in categoricals are handled as another dummy indicator
    drops = list(set(cat_cols).union([target]))
    X = df.drop(columns=drops, errors="ignore")
    X_cat = df[cat_cols]
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
        warn(
            "Using experimental multivariate imputation. This could take a very "
            "long time for even tiny (<500 samples, <50 features) datasets."
        )
        imputer = IterativeImputer(verbose=2)
        X_clean = DataFrame(data=imputer.fit_transform(X), columns=X.columns)
    else:
        raise NotImplementedError(f"Unhandled enum case: {nans}")

    X_clean[target] = y

    return pd.concat([X_cat, X_clean], axis=1)


def encode_target(df: DataFrame, target: Series) -> tuple[DataFrame, Series]:
    unqs, cnts = np.unique(target, return_counts=True)
    idx = cnts <= 20
    n_cls = len(unqs)
    if np.sum(idx) > 0:
        cleaning_inform(
            f"The target variable has a number of class labels ({unqs[idx]}) with "
            "less than 20 members. This will cause problems with splitting in "
            "various nested k-fold procedures used in `df-analyze`. In addition, "
            "any estimates or metrics produced for such a class will not be "
            "statistically meaningful (i.e. the uncertainty on those metrics or "
            "estimates will be exceedingly large). We thus remove all samples "
            "that belong to these labels, bringing the total number of classes "
            f"down to {n_cls - np.sum(idx)}"
        )
        idx_drop = ~target.isin(unqs[idx])
        df = df.copy().loc[idx_drop]
        target = target[idx_drop]

    # drop NaNs: Makes no sense to count correct NaN predictions toward
    # classification performance
    idx_drop = ~target.isna()
    df = df.copy().loc[idx_drop]
    target = target[idx_drop]

    encoded = np.array(LabelEncoder().fit_transform(target))
    return df, Series(encoded, name=target.name)


def clean_regression_target(df: DataFrame, target: Series) -> tuple[DataFrame, Series]:
    """NaN targets cannot be predicted. Remove them, and then robustly
    normalize target to facilitate convergence and interpretation
    of metrics
    """
    idx_drop = ~target.isna()
    df = df.loc[idx_drop]
    target = target[idx_drop]

    y = (
        RobustScaler(quantile_range=(2.5, 97.5))
        .fit_transform(target.to_numpy().reshape(-1, 1))
        .ravel()
    )
    target = Series(y, name=target.name)

    return df, target


def drop_cols(
    df: DataFrame,
    kind: str,
    categoricals: list[str],
    ordinals: list[str],
    *col_dicts: dict[str, str],
) -> tuple[DataFrame, list[str], list[str]]:
    cols = set()
    cols_descs = []
    for d in col_dicts:
        for col, desc in d.items():
            cols.add(col)
            cols_descs.append((col, desc))
    cols_descs = sorted(cols_descs, key=lambda pair: pair[0])

    if len(cols) <= 0:  # nothing to drop
        return df, categoricals, ordinals

    w = max(len(col) for col in cols) + 2
    info = "\n".join([f"{col:<{w}} {desc}" for col, desc in cols_descs])
    cleaning_inform(
        f"Dropping features that appear to be {kind}. Additional information "
        "should be available above.\n\n"
        f"Dropped features:\n{info}"
    )
    drops = list(cols)
    categoricals = list(set(categoricals).difference(drops))
    ordinals = list(set(ordinals).difference(drops))

    df = df.drop(columns=drops, errors="ignore")
    return df, categoricals, ordinals


def detect_big_cats(
    unique_counts: dict[str, int], str_cols: list[str], _warn: bool = True
) -> dict[str, str]:
    big_cats = [col for col in str_cols if (col in unique_counts) and (unique_counts[col] >= 20)]
    if len(big_cats) > 0 and _warn:
        messy_inform(
            "Found string-valued features with more than 50 unique levels. "
            "Unless you have a large number of samples, or if these features "
            "have a highly imbalanced / skewed distribution, then they will "
            "cause sparseness after one-hot encoding. This is generally not "
            "beneficial to most algorithms. You should inspect these features "
            "and think if it makes sense if they would be predictively useful "
            "for the given target. If they are unlikely to be useful, consider "
            "removing them from the data. This will also likely considerably "
            "improve `df-analyze` predictive performance and reduce compute "
            "times. However, we do NOT remove these features automatically\n\n"
            f"String-valued features with over 50 levels: {big_cats}"
        )
    return {col: f"{unique_counts[col]} levels" for col in big_cats}


def get_unq_counts(df: DataFrame, target: str) -> dict[str, int]:
    X = df.drop(columns=target, errors="ignore").infer_objects().convert_dtypes()
    unique_counts = {}
    for colname in X.columns:
        try:
            unique_counts[colname] = len(np.unique(df[colname]))
        except TypeError:  # happens when can't sort for unique
            unique_counts[colname] = len(np.unique(df[colname].astype(str)))
    return unique_counts


def floatify(df: DataFrame) -> DataFrame:
    df = df.copy()
    cols = df.select_dtypes(include=["object", "string[python]"]).columns.tolist()
    for col in cols:
        try:
            df[col] = df[col].astype(float)
        except Exception:
            pass
    return df


def get_str_cols(df: DataFrame, target: str) -> list[str]:
    X = df.drop(columns=target, errors="ignore")
    return X.select_dtypes(include=["object", "string[python]"]).columns.tolist()


def get_int_cols(df: DataFrame, target: str) -> list[str]:
    X = df.drop(columns=target, errors="ignore")
    return X.select_dtypes(include="int").columns.tolist()


def inspect_data(
    df: DataFrame,
    target: str,
    categoricals: Optional[list[str]] = None,
    ordinals: Optional[list[str]] = None,
    _warn: bool = True,
) -> InspectionResults:
    categoricals = categoricals or []
    ordinals = ordinals or []

    str_cols = get_str_cols(df, target)
    int_cols = get_int_cols(df, target)

    df = df.drop(columns=target)

    floats, ords, ids, times, cats = inspect_str_columns(
        df, str_cols, categoricals, ordinals=ordinals, _warn=_warn
    )
    int_ords, int_ids = inspect_int_columns(
        df, int_cols, categoricals, ordinals=ordinals, _warn=_warn
    )

    all_cats = [*categoricals, *cats.keys()]
    unique_counts = get_unq_counts(df=df, target=target)
    bigs = detect_big_cats(unique_counts, all_cats, _warn=_warn)

    all_ordinals = set(ords.keys()).union(int_ords.keys())
    ambiguous = all_ordinals.intersection(cats)
    ambiguous.difference_update(categoricals)
    ambiguous.difference_update(ordinals)
    ambiguous = sorted(ambiguous)
    if len(ambiguous) > 0:
        raise TypeError(
            "Cannot automatically determine the cardinality of features: "
            f"{ambiguous}. Specify each of these as either ordinal or "
            "categorical using the `--ordinals` and `--categoricals` options "
            "to df-analyze."
        )

    return InspectionResults(
        floats=floats,
        ords=ords,
        ids=ids,
        times=times,
        cats=cats,
        int_ords=int_ords,
        int_ids=int_ids,
        big_cats=bigs,
    )


def drop_unusable(
    df: DataFrame,
    results: InspectionResults,
    categoricals: list[str],
    ordinals: list[str],
) -> tuple[DataFrame, list[str], list[str]]:
    cats = categoricals
    ords = ordinals
    ids, int_ids, times = results.ids, results.int_ids, results.times
    df, cats, ords = drop_cols(df, "identifiers", cats, ords, ids, int_ids)
    df, cats, ords = drop_cols(df, "datetime data", cats, ords, times)
    return df, cats, ords


def encode_categoricals(
    df: DataFrame,
    target: str,
    results: InspectionResults,
    categoricals: list[str],
    ordinals: list[str],
) -> tuple[DataFrame, DataFrame]:
    """Treat all features with <= options.cat_threshold as categorical

    Returns
    -------
    encoded: DataFrame
        Pandas DataFrame with categorical variables one-hot encoded

    unencoded: DataFrame
        Pandas DataFrame with original categorical variables
    """

    y = df[target]
    df = df.drop(columns=target)
    df, cats, ords = drop_unusable(df, results, categoricals, ordinals)
    to_convert = [*cats, *results.cats.keys()]

    # below will FAIL if we didn't remove timestamps or etc.
    try:
        new = pd.get_dummies(df, columns=to_convert, dummy_na=True, dtype=float)
        new = new.astype(float)
    except TypeError:
        inspect_data(df, target, categoricals, ordinals, _warn=True)
        raise RuntimeError(
            "Could not convert data to floating point after cleaning. Some "
            "messy data must be removed or cleaned before `df-analyze` can "
            "continue. See information above."
        )
    new = pd.concat([new, y], axis=1)
    return new, df.loc[:, to_convert]


def prepare_data(
    df: DataFrame,
    target: str,
    categoricals: list[str],
    ordinals: list[str],
    is_classification: bool,
    _warn: bool = True,
) -> tuple[DataFrame, Series, list[str]]:
    y = df[target]
    results = inspect_data(df, target, categoricals, ordinals, _warn=_warn)
    df, cats, ords = drop_unusable(df, results, categoricals, ordinals)
    raise NotImplementedError()

    cats = []
    df = handle_continuous_nans(df=df, target=target, cat_cols=cat_cols, nans=NanHandling.Mean)

    df = df.drop(columns=target)
    if is_classification:
        df, y = encode_target(df, y)
    else:
        df, y = clean_regression_target(df, y)
    return df, y, cat_cols


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
