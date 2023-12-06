import sys
import traceback
from dataclasses import dataclass
from math import ceil
from shutil import get_terminal_size
from typing import Optional, Union
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
from dateutil.parser import parse
from dateutil.parser._parser import UnknownTimezoneWarning
from joblib import Parallel, delayed
from pandas import DataFrame, Series
from pandas.core.dtypes.dtypes import CategoricalDtype
from sklearn.experimental import enable_iterative_imputer  # noqa
from tqdm import tqdm

from src._constants import N_CAT_LEVEL_MIN


@dataclass
class InflationInfo:
    col: str
    to_deflate: list[str]
    to_keep: list[str]
    n_deflate: int
    n_keep: int
    n_total: int


@dataclass
class ColumnDescriptions:
    col: str
    const: Optional[str] = None
    time: Optional[str] = None
    ord: Optional[str] = None
    id: Optional[str] = None
    float: Optional[str] = None
    cat: Optional[str] = None


@dataclass
class InspectionResults:
    floats: dict[str, str]
    ords: dict[str, str]
    ids: dict[str, str]
    times: dict[str, str]
    consts: dict[str, str]  # true constants
    cats: dict[str, str]
    int_ords: dict[str, str]
    int_ids: dict[str, str]
    big_cats: dict[str, int]
    nyan_cats: dict[str, str]
    inflation: list[InflationInfo]
    bin_cats: list[str]
    multi_cats: list[str]


class InspectionError(Exception):
    pass


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


def get_str_cols(df: DataFrame, target: str) -> list[str]:
    X = df.drop(columns=target, errors="ignore")
    return X.select_dtypes(include=["object", "string[python]", "category"]).columns.tolist()


def get_int_cols(df: DataFrame, target: str) -> list[str]:
    X = df.drop(columns=target, errors="ignore")
    return X.select_dtypes(include="int").columns.tolist()


def get_unq_counts(df: DataFrame, target: str) -> tuple[dict[str, int], dict[str, int]]:
    X = df.drop(columns=target, errors="ignore").infer_objects().convert_dtypes()
    unique_counts = {}
    nanless_counts = {}
    for colname in X.columns:
        nans = df[colname].isna().sum() > 0
        try:
            unqs = np.unique(df[colname])
        except TypeError:  # happens when can't sort for unique
            unqs = np.unique(df[colname].astype(str))
        unique_counts[colname] = len(unqs)
        nanless_counts[colname] = len(unqs) - int(nans)
    return unique_counts, nanless_counts


def is_timelike(s: str) -> bool:
    # https://stackoverflow.com/a/25341965 for this...
    try:
        int(s)
        return False
    except Exception:
        ...
    try:
        float(s)
        return False
    except Exception:
        ...
    try:
        with catch_warnings():
            filterwarnings("ignore", category=UnknownTimezoneWarning)
            parse(s, fuzzy=False)
        return True
    except ValueError:
        return False


def looks_timelike(series: Series) -> tuple[bool, str]:
    # we don't want to interpret integer-like data as times, even though
    # they could be e.g. Unix timestamps or something like that
    if converts_to_int(series):
        return False, "Converts to int"

    # This seems to be another false positive from dateutil.parse
    if converts_to_float(series):
        return False, "Converts to float"

    series = series.astype(str)

    # We are mostly worried about timestamps we do NOT want to convert to
    # categoricals. Thus before checking that most data parses as datetime,
    # we check that we are below out inflation threshold and do not flag
    # these, even if they are timelike.
    level_counts = np.unique(series, return_counts=True)[1]
    if np.all(level_counts > N_CAT_LEVEL_MIN):
        return False, "Looks like well-sampled categorical"

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
            return True, f"{p*100:< 2.2f}% of data appears parseable as datetime data"
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

    if isinstance(series.dtype, CategoricalDtype):
        if len(series.dtype.categories) > (len(series) / 2):
            return True, "More unique values than one half of number of non-NaN samples"
        return False, "Is Pandas categorical type already"

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
    try:
        series.astype(int)
        return True
    except Exception:
        ...
    with catch_warnings():
        filterwarnings(
            "ignore",
            message=".*invalid value encountered in cast.*",
            category=RuntimeWarning,
        )
        converted = series.convert_dtypes(
            infer_objects=True,
            convert_string=True,
            convert_integer=True,
            convert_boolean=True,
            convert_floating=False,  # type: ignore
        )
    return converted.dtype.kind in ["i", "b"]


def converts_to_float(series: Series) -> bool:
    try:
        converted = series.astype(float).astype(str)
        if np.all(converted == series.to_numpy()):
            return True

    except Exception:
        ...
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
        odd_vals = np.unique(series.astype(str)[idx]).tolist()
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


def looks_constant(series: Series) -> tuple[bool, str]:
    if len(np.unique(series.astype(str))) == 1:
        return True, "Single value even if including NaNs"
    return False, "Two or more unique values including NaNs"


def detect_big_cats(
    df: DataFrame, unique_counts: dict[str, int], all_cats: list[str], _warn: bool = True
) -> tuple[dict[str, int], list[InflationInfo]]:
    """Detect categoricals with more than 20 levels, and "inflating" categoricals
    (see notes).

    Notes
    -----
    Inflated categoricals we deflate by converting all inflated levels to NaN.
    """
    big_cats = [col for col in all_cats if (col in unique_counts) and (unique_counts[col] >= 20)]
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

    # dict is {colname: list[columns to deflate...]}
    inflation_info: list[InflationInfo] = []
    for col in all_cats:
        unqs, cnts = np.unique(df[col].astype(str), return_counts=True)
        n_total = len(unqs)
        if n_total <= 2:  # do not mangle boolean indicators
            continue
        keep_idx = cnts >= N_CAT_LEVEL_MIN
        if np.all(keep_idx):
            continue
        n_keep = keep_idx.sum()
        n_deflate = len(keep_idx) - n_keep
        inflation_info.append(
            InflationInfo(
                col=col,
                to_deflate=unqs[~keep_idx].tolist(),
                to_keep=unqs[keep_idx].tolist(),
                n_deflate=n_deflate,
                n_keep=n_keep,
                n_total=n_total,
            )
        )
    return {col: unique_counts[col] for col in big_cats}, inflation_info


def inspect_str_column(
    series: Series,
    cats: list[str],
    ords: list[str],
) -> Union[ColumnDescriptions, tuple[str, Exception]]:
    col = str(series.name)
    try:
        result = ColumnDescriptions(col)

        is_const, desc = looks_constant(series)
        if is_const:
            result.const = desc
            return result

        maybe_time, desc = looks_timelike(series)
        if maybe_time:
            result.time = desc
            return result  # time is bad enough we don't need other checks

        maybe_ord, desc = looks_ordinal(series)
        if maybe_ord and (col not in ords) and (col not in cats):
            result.ord = desc

        maybe_id, desc = looks_id_like(series)
        if maybe_id:
            result.id = desc

        maybe_float, desc = looks_floatlike(series)
        if maybe_float:
            result.float = desc

        if maybe_float or maybe_id or maybe_time:
            return result

        maybe_cat, desc = looks_categorical(series)
        if maybe_cat and (col not in ords) and (col not in cats):
            result.cat = desc

        return result
    except Exception as e:
        traceback.print_exc()
        return col, e


def inspect_str_columns(
    df: DataFrame,
    str_cols: list[str],
    categoricals: list[str],
    ordinals: list[str],
    _warn: bool = True,
) -> tuple[
    dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str]
]:
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
    const_cols: dict[str, str]
        Columns with one value (either all NaN or one value with no NaNs).
    """
    float_cols: dict[str, str] = {}
    ord_cols: dict[str, str] = {}
    id_cols: dict[str, str] = {}
    time_cols: dict[str, str] = {}
    cat_cols: dict[str, str] = {}
    const_cols: dict[str, str] = {}
    cats = set(categoricals)
    ords = set(ordinals)

    # args = tqdm()
    descs: list[Union[ColumnDescriptions, tuple[str, Exception]]] = Parallel(n_jobs=-1)(
        delayed(inspect_str_column)(df[col], cats, ords)
        for col in tqdm(
            str_cols,
            desc="Inspecting features",
            total=len(str_cols),
            disable=len(str_cols) < 50,
        )
    )  # type: ignore
    for desc in descs:
        if isinstance(desc, tuple):
            col, error = desc
            raise InspectionError(
                f"Could not interpret data in feature {col}. Additional information "
                "should be above."
            ) from error
        if desc.float is not None:
            float_cols[desc.col] = desc.float
        if desc.ord is not None:
            ord_cols[desc.col] = desc.ord
        if desc.id is not None:
            id_cols[desc.col] = desc.id
        if desc.time is not None:
            time_cols[desc.col] = desc.time
        if desc.cat is not None:
            cat_cols[desc.col] = desc.cat
        if desc.const is not None:
            const_cols[desc.col] = desc.const

    # for col in tqdm(
    #     str_cols, desc="Inspecting features", total=len(str_cols), disable=len(str_cols) < 50
    # ):
    #     is_const, desc = looks_constant(df[col])
    #     if is_const:
    #         const_cols[col] = desc
    #         continue

    #     maybe_time, desc = looks_timelike(df[col])
    #     if maybe_time:
    #         time_cols[col] = desc
    #         continue  # time is bad enough we don't need other checks

    #     maybe_ord, desc = looks_ordinal(df[col])
    #     if maybe_ord and (col not in ords) and (col not in cats):
    #         ord_cols[col] = desc

    #     maybe_id, desc = looks_id_like(df[col])
    #     if maybe_id:
    #         id_cols[col] = desc

    #     maybe_float, desc = looks_floatlike(df[col])
    #     if maybe_float:
    #         float_cols[col] = desc

    #     if maybe_float or maybe_id or maybe_time:
    #         continue

    #     maybe_cat, desc = looks_categorical(df[col])
    #     if maybe_cat and (col not in ords) and (col not in cats):
    #         cat_cols[col] = desc

    if not _warn:
        return float_cols, ord_cols, id_cols, time_cols, cat_cols, const_cols

    all_cols = {**float_cols, **ord_cols, **id_cols, **time_cols, **cat_cols, **const_cols}
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

    if len(const_cols) > 0:
        info = "\n".join([f"{col:<{w}} {desc}" for col, desc in const_cols.items()])
        messy_inform(
            "Found string-valued features that are constant (i.e. all values) "
            "are NaN or all values are the same non-NaN value). These contain "
            "no information and will be removed automatically.\n\n"
            f"Features that are constant:\n{info}"
        )

    return float_cols, ord_cols, id_cols, time_cols, cat_cols, const_cols


def inspect_int_columns(
    df: DataFrame,
    int_cols: list[str],
    categoricals: list[str],
    ordinals: list[str],
    _warn: bool = True,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """
    Returns
    -------
    ord_cols: dict[str, str]
        Columns that may be ordinal
    id_cols: dict[str, str]
        Columns that may be identifiers
    """
    results = inspect_str_columns(
        df,
        str_cols=int_cols,
        categoricals=categoricals,
        ordinals=ordinals,
        _warn=_warn,
    )
    return results[1], results[2], results[5]


def inspect_other_columns(
    df: DataFrame,
    other_cols: list[str],
    categoricals: list[str],
    ordinals: list[str],
    _warn: bool = True,
) -> dict[str, str]:
    """
    Returns
    -------
    const_Cols: dict[str, str]
        Columns that are constant
    """
    const_cols: dict[str, str] = {}
    # TODO
    cats = set(categoricals)
    ords = set(ordinals)
    for col in tqdm(
        other_cols, desc="Inspecting features", total=len(other_cols), disable=len(other_cols) < 50
    ):
        is_const, desc = looks_constant(df[col])
        if is_const:
            const_cols[col] = desc
            continue

    if not _warn:
        return const_cols

    all_cols = {**const_cols}
    if len(all_cols) > 0:
        w = max(len(col) for col in all_cols) + 2
    else:
        w = 0

    if len(const_cols) > 0:
        info = "\n".join([f"{col:<{w}} {desc}" for col, desc in const_cols.items()])
        messy_inform(
            "Found features that are constant (i.e. all values) are NaN or "
            "all values are the same non-NaN value). These contain no "
            "information and will be removed automatically.\n\n"
            f"Features that are constant:\n{info}"
        )
    return const_cols


def inspect_data(
    df: DataFrame,
    target: str,
    categoricals: Optional[list[str]] = None,
    ordinals: Optional[list[str]] = None,
    _warn: bool = True,
) -> InspectionResults:
    categoricals = categoricals or []
    ordinals = ordinals or []
    # convert screwy categorical columns which can have all sorts of
    # annoying behaviours when incorrectly labeled as such
    for col in df.columns:
        if df[col].dtype == "category":
            # https://stackoverflow.com/a/70442594
            # Pandas so dumb, why it would silently ignore casting to string
            # and retain the categorical dtype makes no sense, e.g.
            #
            #       df[col] = df[col].astype("string")
            #
            # fails to do anything but silently passes without warning or error
            df[col] = df[col].astype(df[col].cat.categories.to_numpy().dtype)

    str_cols = get_str_cols(df, target)
    int_cols = get_int_cols(df, target)

    df = df.drop(columns=target)
    remain = set(df.columns.to_list()).difference(str_cols).difference(int_cols)
    remain = list(remain)

    floats, ords, ids, times, cats, consts = inspect_str_columns(
        df, str_cols, categoricals, ordinals=ordinals, _warn=_warn
    )
    int_ords, int_ids, int_consts = inspect_int_columns(
        df, int_cols, categoricals, ordinals=ordinals, _warn=_warn
    )
    other_consts = inspect_other_columns(df, remain, categoricals, ordinals=ordinals, _warn=_warn)

    all_consts = {**consts, **int_consts, **other_consts}
    all_cats = [*categoricals, *cats.keys()]
    unique_counts, nanless_cnts = get_unq_counts(df=df, target=target)
    bigs, inflation = detect_big_cats(df, unique_counts, all_cats, _warn=_warn)

    bin_cats = {cat for cat, cnt in nanless_cnts.items() if cnt == 2}
    nyan_cats = {cat for cat, cnt in nanless_cnts.items() if cnt == 1}
    multi_cats = {cat for cat, cnt in nanless_cnts.items() if cnt > 2}

    bin_cats = sorted(bin_cats.intersection(all_cats))
    nyan_cats = sorted(nyan_cats.intersection(all_cats))
    multi_cats = sorted(multi_cats.intersection(all_cats))

    nyan_cats = {col: "Constant categorical" for col in nyan_cats}

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
            "to df-analyze, or eliminate them using the `--drops` option."
        )

    return InspectionResults(
        floats=floats,
        ords=ords,
        ids=ids,
        times=times,
        consts=all_consts,
        cats=cats,
        int_ords=int_ords,
        int_ids=int_ids,
        big_cats=bigs,
        inflation=inflation,
        bin_cats=bin_cats,
        nyan_cats=nyan_cats,
        multi_cats=multi_cats,
    )
