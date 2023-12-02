from math import ceil
from pathlib import Path
from shutil import get_terminal_size
from textwrap import dedent
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


class MessyDataWarning(UserWarning):
    """For when df-analyze detects (but does not resolve) data problems"""

    def __init__(self, message: str) -> None:
        cols = get_terminal_size((81, 24))[0]
        sep = "=" * cols
        self.message = f"\n{sep}\n{self.__class__.__name__}:\n{message}\n{sep}"

    def __str__(self) -> str:
        return str(self.message)


def is_timelike(s: str) -> bool:
    # https://stackoverflow.com/a/25341965 for this...
    try:
        int(s)
        return False
    except Exception:
        ...
    try:
        parse(s, fuzzy=False)
        return True
    except ValueError:
        return False


def looks_timelike(series: Series) -> tuple[bool, str]:
    series = series.apply(str)
    N = len(series)
    n_subsamp = max(ceil(0.5 * N), 500)
    n_subsamp = min(n_subsamp, N)
    idx = np.random.permutation(N)[:n_subsamp]

    percent = series.iloc[idx].apply(is_timelike).sum() / n_subsamp
    if percent >= 1.0:
        return True, "100% of data parses as datetime"
    if percent > (1.0 / 3.0):
        p = series.loc[idx].apply(is_timelike).mean()
        if p > 0.5:
            return True, f"{p*100:02f}% of data appears parseable as datetime data"
    return False, ""


def looks_id_like(series: Series) -> tuple[bool, str]:
    """
    Returns
    -------
    id_like: bool
        If the variable has a good chance of being an identifier

    desc: str
        A string describing why the variable looks like an identifier
    """
    cnts = np.unique(series.apply(str), return_counts=True)[1]
    if np.all(cnts == 1):  # obvious case
        return True, "All values including possible NaNs are unique"

    dropped = series.dropna()
    if len(dropped) < 0.5 * len(series):
        # seems unlikely only half of data would have identifier info?
        return False, "More than half of data is NaN"

    cnts = np.unique(dropped, return_counts=True)[1]
    if np.all(cnts == 1):  # also obvious case
        return True, "All non-NaN values are unique"

    return False, ""


def prob_unq_ordinals(n_samples: int, ordinal_max: int) -> float:
    """
    Return the probability of all `n_samples` samples having a unique value
    when drawn uniformly from the values [0, ordinal_max].
    """
    if n_samples > ordinal_max:
        return 0.0
    p = 1.0
    for i in range(1, n_samples - 1):
        # case i you have M - i choices out of M total
        prob = (ordinal_max - i) / ordinal_max
        p *= prob
    return p


def maybe_large_ordinal(series: Series) -> bool:
    prob = prob_unq_ordinals(n_samples=len(series), ordinal_max=int(np.max(series)))
    return prob >= 0.5


def converts_to_int(series: Series) -> bool:
    converted = series.convert_dtypes(
        infer_objects=True,
        convert_string=True,
        convert_integer=True,
        convert_boolean=True,
        convert_floating=False,  # type: ignore
    )
    return converted.dtype.kind in ["i", "b"]


def converts_to_float(series: Series) -> bool:
    converted = series.convert_dtypes(
        infer_objects=True,
        convert_string=True,
        convert_integer=False,
        convert_boolean=False,
        convert_floating=True,  # type: ignore
    )
    return converted.dtype.kind == "f"


def looks_ordinal(series: Series) -> tuple[bool, str]:
    """
    Returns
    -------
    ordinal: bool
        If the variable has a good chance of being ordinal

    desc: str
        A string describing the apparent ordinality when `ordinal` is True
    """
    forced = pd.to_numeric(series, errors="coerce")
    idx = forced.isna()

    if np.all(idx):  # columns definitely all not numerical
        return False, ""

    dropped = forced.dropna()
    if not converts_to_int(dropped):
        return False, ""

    try:
        ints = dropped.astype(int)
    except Exception:
        # this `dropped` contains things that look like floats
        return False, ""

    unq_ints, cnts = np.unique(ints, return_counts=True)

    # typical case might be ordinal or ids with some NaNs
    # Two obvious cases here:
    #
    # (1) every non-NaN value is unique because it is an identifier
    # (2) we have a very large ordinal relative to the number of samples
    if np.all(cnts == 1):
        if maybe_large_ordinal(dropped):
            vmin, vmax = ints.min(), ints.max()
            return (
                True,
                f"All unique values in large range [{vmin}, {vmax}] relative to number of samples",
            )
        return False, "All unique values"

    # Now we have something int-like with few unique values. If diffs
    # on sorted unique values are all 1, we have something extremely
    # likely to be ordinal again.
    diffs = np.diff(np.sort(unq_ints))
    unq_diffs = np.unique(diffs)
    if len(unq_diffs) == 1:
        vmin, vmax = ints.min(), ints.max()
        if vmin == 0 and vmax == 1:
            return True, f"Binary {{{vmin}, {vmax}}} indicator"
        return True, f"Increasing integers in [{vmin}, {vmax}]"

    # Small chance remains that we are missing some level(s) of an ordinal, so
    # that we have all values in [0, ..., N] except for a couple, making some
    # diffs on the sorted unique values greater than 1. Here, we just
    # heuristically probably want to warn the user for some likely common cases
    # In most cases, this would be some rating
    imax = np.max(unq_ints)
    if imax in [4, 6]:
        return True, f"Largest int is a common 0-indexed Likert-type scale value: {imax}"
    if imax in [5, 7]:
        return True, f"Largest int is a common Likert-type scale value: {imax}"

    # The 9, 10 cases below are extremely unlikely
    if imax in [10, 100]:
        return True, f"Largest int is a common scale max: {imax}"

    if imax in [9, 99]:
        return True, f"Largest int is a common 0-indexed scale max: {imax}"

    return False, ""


def looks_floatlike(series: Series) -> tuple[bool, str]:
    """
    Returns
    -------
    floaty: bool
        If the variable has a good chance of being continuous

    desc: str
        A string describing the apparent continuousness

    """
    # for some reasons Python None converts inconsistently to NaN...
    if converts_to_int(series):
        return False, "Converts to int"
    try:
        series.astype(float)
        return True, "All values are not integers and convert to float"
    except Exception:
        pass

    forced = pd.to_numeric(series, errors="coerce")
    idx = forced.isna()
    if np.all(idx):  # columns definitely all not float
        return False, "No values parse as valid floats"

    if np.mean(idx) > 0.2:
        odd_vals = np.unique(series.apply(str)[idx]).tolist()
        if len(odd_vals) > 5:
            desc = f"{str(odd_vals[:5])[:-1]} ...]"
        else:
            desc = str(odd_vals)
        return True, rf"More than 20% of values parse as floats, but couldn't parse values: {desc}"
    return False, r"Less than 20% of values parse as floats"


def inspect_str_columns(
    df: DataFrame,
    str_cols: list[str],
    _warn: bool = True,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """
    Returns
    -------
    float_cols: dict[str, str]
        Columns that may be continuous
    ord_cols: dict[str, str]
        Columns that may be ordinal
    id_cols: dict[str, str]
        Columns that may be identifiers
    time_cols: dict[str, str]
        Columns that may be timestamps or timedeltas
    """
    float_cols: dict[str, str] = {}
    ord_cols: dict[str, str] = {}
    id_cols: dict[str, str] = {}
    time_cols: dict[str, str] = {}
    for col in str_cols:
        maybe_time, desc = looks_timelike(df[col])
        if maybe_time:
            time_cols[col] = desc
            continue  # time is bad enough we don't need other checks

        maybe_ord, desc = looks_ordinal(df[col])
        if maybe_ord:
            ord_cols[col] = desc
        maybe_id, desc = looks_id_like(df[col])
        if maybe_id:
            id_cols[col] = desc

        if maybe_ord or maybe_id:
            continue

        maybe_float, desc = looks_floatlike(df[col])
        if maybe_float:
            float_cols[col] = desc
    if not _warn:
        return float_cols, ord_cols, id_cols, time_cols

    all_cols = {**float_cols, **ord_cols, **id_cols, **time_cols}
    if len(all_cols) > 0:
        w = max(len(col) for col in all_cols) + 2
    else:
        w = 0
    if len(float_cols) > 0:
        info = "\n".join([f"{col:<{w}} {desc}" for col, desc in float_cols.items()])
        warn(
            "Found string-valued features that seem to mostly contain "
            "continuous values. This likely means these columns are formatted "
            "oddly (e.g. all values might be quoted) or that there is a typo "
            "or strange value in the column. df-analyze will automatically "
            "treat these columns as continuous to prevent wasted compute "
            "resources that would be incurred with encoding them as "
            "categorical, however, this might be an error. "
            f"Columns that may be continuous:\n\n{info}",
            category=MessyDataWarning,
        )

    if len(ord_cols) > 0:
        info = "\n".join([f"{col:<{w}} {desc}" for col, desc in ord_cols.items()])
        warn(
            "Found string-valued features that could contain ordinal values "
            "(i.e. non-categorical integer values). Make sure you have NOT "
            "identified these values as `--categorical` when configuring "
            "df-analyze. "
            f"Columns that may be ordinal:\n\n{info}",
            category=MessyDataWarning,
        )

    if len(id_cols) > 0:
        info = "\n".join([f"{col:<{w}} {desc}" for col, desc in id_cols.items()])
        warn(
            "Found string-valued features likely containing identifiers (i.e. "
            "unique string or integer values that are assigned arbitrarily). "
            "These have no predictive value and must be removed from your data "
            "in order not to waste compute resources. df-analyze will do this "
            "automatically, but you should inspect these features yourself and "
            "ensure that they are in fact identifiers, and, if so, remove them "
            "from your data to silence this warning."
            f"Columns that likely are identifiers:\n\n{info} ",
            category=MessyDataWarning,
        )

    if len(time_cols) > 0:
        info = "\n".join([f"{col:<{w}} {desc}" for col, desc in time_cols.items()])
        warn(
            "Found string-valued features that appear to be datetime data or "
            "time differences. Datetime data cannot currently be handled by "
            "`df-analyze` (or most AutoML or or most automated predictive "
            "approaches) due to special data preprocessing needs (e.g. "
            "Fourier features), splitting (e.g. time-based cross-validation, "
            "forecasting, hindcasting) and in the models used (ARIMA, VAR, "
            "etc.).\n\n "
            f"Columns that are likely timestamps:\n\n{info}\n\n"
            ""
            "To remove this warning, DELETE these columns from your data, "
            "or manually edit or convert the column values so they are "
            "interpretable as a categorical or continuous variable reflecting "
            "some kind of cyclicality that may be relevant to the predictive "
            "task. E.g. a variable that stores the time a sample was recorded "
            "might be converted to categorical variable like:\n\n"
            ""
            "  - morning, afternoon, evening, night (for daily data)\n"
            "  - day of the week (for monthly data)\n"
            "  - month (for yearly data)\n"
            "  - season (for multi-year data)\n\n"
            ""
            "Or to continuous / ordinal cyclical versions of these, like:\n\n"
            ""
            "  - values from 0 to 23 for ordinal representation of day hour\n"
            "  - values from 0.0 to 23.99 for continuous version of above\n"
            "  - values from 0 to 7 for day of the week, 0 to 365 for year\n\n"
            ""
            "It is possible to convert a single time feature into all of the "
            "above, i.e. to expand the feature into multiple cyclical features. "
            "This would be a variant of Fourier features (see e.g.\n\n"
            ""
            "\tTian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong "
            'Jin:\n\t"FEDformer: Frequency Enhanced Decomposed Transformer\n\tfor '
            'Long-term Series Forecasting", 2022;\n\t'
            "[http://arxiv.org/abs/2201.12740].\n\n "
            ""
            "`df-analyze` may in the future attempt to automate this via "
            "`sktime` (https://www.sktime.net). ",
            category=MessyDataWarning,
        )

    return float_cols, ord_cols, id_cols, time_cols
