from __future__ import annotations

import re
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
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

TIME_WORDS = [
    r".*time.*",
    r".*duration.*",
    r".*interval.*",
    r".*lag.*",
    r".*date.*",
    r".*elapse.*",
    r".*day[ _\.\-]?",
    r".*month[ _\.\-]?",
    r".*year[ _\.\-]?",
    r".*yr[ _\.\-]?",
    r".*hour[ _\.\-]?",
]
ORD_WORDS = [
    r".*[_ \-]n[_ \-].*",
    r".*[_ \-]num[_ \-]?.*",
    r".*age.*",
    r".*days.*",
    r".*grade.*",
    r".*hours.*",
    r".*hrs.*",
    r".*months.*",
    r".*rank.*",
    r".*seconds.*",
    r".*secs.*",
    r".*size.*",
    r".*years.*",
    r".*yrs.*",
]
ID_WORDS = [r".id$"]
CAT_WORDS = [
    r".*type.*",
    r".*kind.*",
    r".*is[_ \-\.].*",
    r".*has[_ \-\.].*",
    r".*not[_ \-\.].*",
    "city",
    "race",
    "ancestry",
    "town",
    "country",
    "occupation",
    r".*relig.*",
    "state",
    "zip",
    "code",
    "name",
    "gender",
    "sex",
    "status",
]
CONT_WORDS = [
    r".*average.*",
    r".*avg.*",
    r".*pct.*",
    r".*percent.*",
    r".*rate.*",
    r".*ratio.*",
    r".*total.*",
]


class Inference(Enum):
    MaybeOrd = "ord?"
    MaybeCat = "cat?"
    MaybeCont = "cont?"
    MaybeId = "id?"
    MaybeTime = "time?"
    CertainOrd = "ord"
    CertainCat = "cat"
    CertainCont = "cont"
    CertainId = "id"
    CertainTime = "time"
    Const = "const"

    def is_certain(self) -> bool:
        return self in [
            Inference.CertainCat,
            Inference.CertainCont,
            Inference.CertainOrd,
            Inference.CertainId,
            Inference.CertainTime,
            Inference.Const,
        ]


def has_cat_name(series: Series) -> tuple[bool, str]:
    for pattern in CAT_WORDS:
        if re.search(pattern, str(series.name).lower()) is not None:
            return True, str(pattern)
    return False, ""


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
        x = series.astype(int)
        if np.all(x == series):
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
    vmin, vmax = ints.min(), ints.max()
    if len(unq_ints) == 1:
        return True, "Constant integer after dropping NaNs"

    # typical case might be ordinal or ids with some NaNs
    # Two obvious cases here:
    #
    # (1) every non-NaN value is unique because it is an identifier
    # (2) we have a very large ordinal relative to the number of samples
    if np.all(cnts == 1):
        if maybe_large_ordinal(dropped):
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
        if vmin == 0 and vmax == 1:
            return True, f"Binary {{{vmin}, {vmax}}} indicator"
        return True, f"Increasing integers in [{vmin}, {vmax}]"

    # Small chance remains that we are missing some level(s) of an ordinal, so
    # that we have all values in [0, ..., N] except for a couple, making some
    # diffs on the sorted unique values greater than 1.
    if np.mean(diffs == 1) >= 0.8:
        return True, r"80% or more of unique integer values differ only by 1"
    if len(unq_ints) / (vmax - vmin) >= 0.8:
        return True, f"80% or more of values in [{vmin}, {vmax}] are sampled"

    # Here, we just heuristically probably want to warn the user for some
    # likely common cases In most cases, this would be some rating

    if int(vmin) not in [0, 1]:
        return True, "Integers not starting at 0 or 1"

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
        return True, "Data not interpretable as numeric"

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
