import sys
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
        underline = "." * (len(self.__class__.__name__) + 1)
        self.message = f"\n{sep}\n{self.__class__.__name__}\n{underline}\n{message}\n{sep}"

    def __str__(self) -> str:
        return str(self.message)


def messy_inform(message: str) -> None:
    cols = get_terminal_size((81, 24))[0]
    sep = "=" * cols
    title = "Found Messy Data"
    underline = "." * (len(title) + 1)
    message = f"\n{sep}\n{title}\n{underline}\n{message}\n{sep}"
    print(message, file=sys.stderr)


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
    if looks_floatlike(series)[0]:
        return False, "Appears float-like"

    cnts = np.unique(series.apply(str), return_counts=True)[1]
    if np.all(cnts == 1):  # obvious case
        return True, "All values including possible NaNs are unique"

    dropped = series.dropna()
    if len(dropped) < 0.5 * len(series):
        # seems unlikely only half of data would have identifier info?
        return False, "More than half of data is NaN"

    unqs, cnts = np.unique(dropped, return_counts=True)
    if np.all(cnts == 1):  # also obvious case
        return True, "All non-NaN values are unique"

    if len(unqs) >= (len(dropped) / 2):
        return True, "More unique values than one half of number of non-NaN samples"

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
    if looks_ordinal(series)[0]:
        return False, "Looks ordinal"

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


def looks_categorical(series: Series) -> tuple[bool, str]:
    if looks_floatlike(series)[0]:
        return False, "Converts to float"
    if looks_timelike(series)[0]:
        return False, "Looks timelike"
    if looks_id_like(series)[0]:
        return False, "Looks identifier-like"

    if not converts_to_int(series):  # already checked not float
        return True, "String data"

    # Now we have something that converts to int, and doesn't look like an
    # identifier. It could be a small ordinal or a categorical, but generally
    # there is no way to be sure. All we can say is if it does NOT look ordinal
    # then we definitely want to label it as categorical. Otherwise, we should
    # still flag it.
    if not looks_ordinal(series)[0]:
        return True, "Integer data but not ordinal-like"

    return True, "Either categorical or ordinal"


def inspect_str_columns(
    df: DataFrame,
    str_cols: list[str],
    categoricals: list[str],
    ordinals: list[str],
    _warn: bool = True,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
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
    cat_cols: dict[str, str]
        Columns that may be categorical and not specified in `--categoricals`
    """
    float_cols: dict[str, str] = {}
    ord_cols: dict[str, str] = {}
    id_cols: dict[str, str] = {}
    time_cols: dict[str, str] = {}
    cat_cols: dict[str, str] = {}
    cats = set(categoricals)
    ords = set(ordinals)
    for col in str_cols:
        maybe_time, desc = looks_timelike(df[col])
        if maybe_time:
            time_cols[col] = desc
            continue  # time is bad enough we don't need other checks

        maybe_ord, desc = looks_ordinal(df[col])
        if maybe_ord and (col not in ords) and (col not in cats):
            ord_cols[col] = desc

        maybe_id, desc = looks_id_like(df[col])
        if maybe_id:
            id_cols[col] = desc

        maybe_float, desc = looks_floatlike(df[col])
        if maybe_float:
            float_cols[col] = desc

        if maybe_float or maybe_id or maybe_time:
            continue

        maybe_cat, desc = looks_categorical(df[col])
        if maybe_cat and (col not in ords) and (col not in cats):
            cat_cols[col] = desc

    if not _warn:
        return float_cols, ord_cols, id_cols, time_cols, cat_cols

    all_cols = {**float_cols, **ord_cols, **id_cols, **time_cols, **cat_cols}
    if len(all_cols) > 0:
        w = max(len(col) for col in all_cols) + 2
    else:
        w = 0
    if len(float_cols) > 0:
        info = "\n".join([f"{col:<{w}} {desc}" for col, desc in float_cols.items()])
        messy_inform(
            "Found string-valued features that seem to mostly contain "
            "continuous values. This likely means these columns are formatted "
            "oddly (e.g. all values might be quoted) or that there is a typo "
            "or strange value in the column. df-analyze will automatically "
            "treat these columns as continuous to prevent wasted compute "
            "resources that would be incurred with encoding them as "
            "categorical, however, this might be an error.\n\n"
            f"Columns that may be continuous:\n{info}"
        )

    if len(ord_cols) > 0:
        info = "\n".join([f"{col:<{w}} {desc}" for col, desc in ord_cols.items()])
        messy_inform(
            "Found string-valued features that could contain ordinal values "
            "(i.e. non-categorical integer values) NOT specified with the "
            "`--ordinals` option. Check if these features are ordinal or "
            "categorical, and then explicitly pass them to either the "
            "`--categoricals` or `--ordinals` options when configuring "
            "df-analyze.\n\n"
            f"Columns that may be ordinal:\n{info}"
        )

    if len(id_cols) > 0:
        info = "\n".join([f"{col:<{w}} {desc}" for col, desc in id_cols.items()])
        messy_inform(
            "Found string-valued features likely containing identifiers (i.e. "
            "unique string or integer values that are assigned arbitrarily), "
            "or which have more levels (unique values) than one half of the "
            "number of (non-NaN) samples in the data. This is most likely an "
            "identifier or junk feature which has no predictive value, and thus "
            "should be removed from the data. Even if the feature is not an "
            "identifer, with such a large number of levels, then a test set "
            "(either in k-fold, or holdout) will, on average, mostly contain "
            "values that were never seen during training. Thus, these features "
            "are essentially undersampled, and too sparse to be useful given "
            "the amount of data. Encoding this many values also massively "
            "increases compute costs for little gain. We thus REMOVE these "
            "features. However, but you should inspect these features "
            "yourself and ensure these features are not better described as "
            "either ordinal or continuous. If so, specify them using the  "
            "`--ordinals` or `--continous` options to df-analyze.\n\n"
            f"Columns that likely are identifiers:\n{info} "
        )

    if len(time_cols) > 0:
        info = "\n".join([f"{col:<{w}} {desc}" for col, desc in time_cols.items()])
        messy_inform(
            "Found string-valued features that appear to be datetime data or "
            "time differences. Datetime data cannot currently be handled by "
            "`df-analyze` (or most AutoML or or most automated predictive "
            "approaches) due to special data preprocessing needs (e.g. "
            "Fourier features), splitting (e.g. time-based cross-validation, "
            "forecasting, hindcasting) and in the models used (ARIMA, VAR, "
            "etc.).\n\n"
            f"Columns that are likely timestamps:\n{info}\n\n"
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
            "[http://arxiv.org/abs/2201.12740].\n\n"
            ""
            "`df-analyze` may in the future attempt to automate this via "
            "`sktime` (https://www.sktime.net). "
        )
    if len(cat_cols) > 0:
        info = "\n".join([f"{col:<{w}} {desc}" for col, desc in cat_cols.items()])
        messy_inform(
            "Found string-valued features not specified with `--categoricals` "
            "argument, and that look categorical. These will be one-hot "
            "encoded to allow use in subsequent analyses. To silence this "
            "warning, specify them as categoricals or ordinals manually "
            "either via the CLI or in the spreadsheet file header, "
            "using the `--categoricals` and/or `--ordinals` option.\n\n"
            f"Features that look categorical not specified by `--categoricals`:\n{info}"
        )

    return float_cols, ord_cols, id_cols, time_cols, cat_cols


def inspect_int_columns(
    df: DataFrame,
    int_cols: list[str],
    categoricals: list[str],
    ordinals: list[str],
    _warn: bool = True,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns
    -------
    ord_cols: dict[str, str]
        Columns that may be ordinal
    id_cols: dict[str, str]
        Columns that may be identifiers
    """
    ords, ints = inspect_str_columns(
        df,
        str_cols=int_cols,
        categoricals=categoricals,
        ordinals=ordinals,
        _warn=_warn,
    )[1:3]
    return ords, ints
