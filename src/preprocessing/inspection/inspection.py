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
from src.preprocessing.inspection.containers import (
    ColumnDescriptions,
    ColumnType,
    InflationInfo,
    InspectionInfo,
    InspectionResults,
)
from src.preprocessing.inspection.inference import (
    CAT_WORDS,
    has_cat_name,
    looks_categorical,
    looks_constant,
    looks_floatlike,
    looks_id_like,
    looks_ordinal,
    looks_timelike,
)


class InspectionError(Exception):
    pass


def messy_inform(message: str) -> str:
    cols = get_terminal_size((81, 24))[0]
    sep = "=" * cols
    title = "Found Messy Data"
    underline = "." * (len(title) + 1)
    message = f"\n{sep}\n{title}\n{underline}\n{message}\n{sep}"
    print(message, file=sys.stderr)
    return message


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


def inflation(series: Series) -> float:
    unqs, cnts = np.unique(series.astype(str), return_counts=True)
    index = np.mean(cnts < N_CAT_LEVEL_MIN)
    return index.item()


def detect_big_cats(
    df: DataFrame, unique_counts: dict[str, int], all_cats: list[str], _warn: bool = True
) -> tuple[dict[str, int], list[InflationInfo], Optional[InspectionInfo]]:
    """Detect categoricals with more than 20 levels, and "inflating" categoricals
    (see notes).

    Notes
    -----
    Inflated categoricals we deflate by converting all inflated levels to NaN.
    """
    big_cats = [col for col in all_cats if (col in unique_counts) and (unique_counts[col] >= 20)]
    big_cols = {col: f"{unique_counts[col]} levels" for col in big_cats}
    inspect_info = InspectionInfo(ColumnType.BigCat, big_cols)
    if _warn:
        inspect_info.print_message()

    # dict is {colname: list[columns to deflate...]}
    inflation_infos: list[InflationInfo] = []
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
        inflation_infos.append(
            InflationInfo(
                col=col,
                to_deflate=unqs[~keep_idx].tolist(),
                to_keep=unqs[keep_idx].tolist(),
                n_deflate=n_deflate,
                n_keep=n_keep,
                n_total=n_total,
            )
        )
    return {col: unique_counts[col] for col in big_cats}, inflation_infos, inspect_info


def inspect_str_column(
    series: Series,
    cats: list[str],
    ords: list[str],
) -> Union[ColumnDescriptions, tuple[str, Exception]]:
    col = str(series.name)
    try:
        result = ColumnDescriptions(col)

        # this sould override even user-specified cardinality, as a constant
        # column as we define it here is useless no matter what
        is_const, desc = looks_constant(series)
        if is_const:
            result.const = desc
            return result

        # Likewise, we are not equipped to handle timeseries features, so this
        # too must override use-specified cardinality
        maybe_time, desc = looks_timelike(series)
        if maybe_time:
            result.time = desc
            return result  # time is bad enough we don't need other checks

        # Trust the user and return early for use-specified features
        if col in cats:
            result.cat = "User-specified categorical"
            return result

        if col in ords:
            result.ord = "User-specified ordinal"
            return result

        maybe_ord, desc = looks_ordinal(series)
        if maybe_ord and (col not in ords) and (col not in cats):
            result.ord = desc

        maybe_id, desc = looks_id_like(series)
        if maybe_id:
            result.id = desc

        maybe_float, desc = looks_floatlike(series)
        if maybe_float:
            result.cont = desc

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
    InspectionInfo,
    InspectionInfo,
    InspectionInfo,
    InspectionInfo,
    InspectionInfo,
    InspectionInfo,
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
        if desc.cont is not None:
            float_cols[desc.col] = desc.cont
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

    float_info = InspectionInfo(ColumnType.Continuous, float_cols)
    ord_info = InspectionInfo(ColumnType.Ordinal, ord_cols)
    id_info = InspectionInfo(ColumnType.Id, id_cols)
    time_info = InspectionInfo(ColumnType.Time, time_cols)
    cat_info = InspectionInfo(ColumnType.Categorical, cat_cols)
    const_info = InspectionInfo(ColumnType.Const, const_cols)

    if _warn:
        id_info.print_message()
        time_info.print_message()
        const_info.print_message()
        float_info.print_message()
        ord_info.print_message()
        cat_info.print_message()

    return float_info, ord_info, id_info, time_info, cat_info, const_info


def inspect_int_columns(
    df: DataFrame,
    int_cols: list[str],
    categoricals: list[str],
    ordinals: list[str],
    _warn: bool = True,
) -> tuple[InspectionInfo, InspectionInfo, InspectionInfo]:
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
) -> InspectionInfo:
    """
    Returns
    -------
    const_Cols: dict[str, str]
        Columns that are constant
    """
    const_cols: dict[str, str] = {}
    # TODO
    # cats = set(categoricals)
    # ords = set(ordinals)
    for col in tqdm(
        other_cols, desc="Inspecting features", total=len(other_cols), disable=len(other_cols) < 50
    ):
        is_const, desc = looks_constant(df[col])
        if is_const:
            const_cols[col] = desc
            continue

    info = InspectionInfo(ColumnType.Const, const_cols)
    if _warn:
        info.print_message()
    return info


def coerce_ambiguous_cols(
    df: DataFrame,
    ambigs: list[str],
    _warn: bool = True,
) -> tuple[
    InspectionInfo,
    InspectionInfo,
    InspectionInfo,
]:
    """
    Coerce columns that cannot be clearly determined to be categorical,
    orfinal, or float to either float or categorical.


    Returns
    -------
    float_cols: dict[str, str]
        Columns coerced to float
    ord_cols: dict[str, str]
        Columns coerced to float
    cat_cols: dict[str, str]
        Columns coerced to categorical

    Notes
    -----
    This shoud generally only happen with columns where values are like:

        0.0, 1.0, 2.0, 1.0, 3.0, ...

    I.e. conversion to integer can happen without loss of precision. We examine
    the string-form of the feature and if any contain "." assume the feature
    was meant as a float, although incompetent data entry or saving was always
    possible in this case.

    In general, categoricals are expensive, so we do not want to coerce to
    categorical without good reason. We only do so when ALL the following hold:

    1. the coerced categorical has zero inflation (all levels have 50+ samples)
    2. the feature name strongly suggests a categorical interpretation

    Also, ordinals are ultimately treated no differently than continuous
    currently.
    """

    ...
    float_cols: dict[str, str] = {}
    ord_cols: dict[str, str] = {}
    cat_cols: dict[str, str] = {}

    for col in ambigs:
        series = df[col]
        cat_named, match = has_cat_name(series)
        if cat_named and (inflation(series) <= 0.0):
            cat_cols[col] = f"Coerced to categorical since well-sampled and matches pattern {match}"
            continue
        if series.astype(str).str.contains(".", regex=False).any():
            float_cols[col] = "Coerced to float due to presence of decimal"
        else:
            ord_cols[
                col
            ] = "Coerced to ordinal due to no decimal, feature name or undersampled levels"

    float_info = InspectionInfo(ColumnType.Continuous, float_cols)
    ord_info = InspectionInfo(ColumnType.Ordinal, ord_cols)
    cat_info = InspectionInfo(ColumnType.Categorical, cat_cols)
    return float_info, ord_info, cat_info


def convert_categorical(series: Series) -> Series:
    # https://stackoverflow.com/a/70442594
    # Pandas so dumb, why it would silently ignore casting to string and retain
    # the categorical dtype makes no sense, e.g.
    #
    #       df[col] = df[col].astype("string")
    #
    # fails to do anything but silently passes without warning or error
    if series.dtype == "category":
        return series.astype(series.cat.categories.to_numpy().dtype)
    return series


def convert_categoricals(df: DataFrame, target: str) -> DataFrame:
    # convert screwy categorical columns which can have all sorts of
    # annoying behaviours when incorrectly labeled as such
    df = df.copy()
    for col in df.columns:
        if str(col) == target:
            continue
        df[col] = convert_categorical(df[col])
    return df


def inspect_data(
    df: DataFrame,
    target: str,
    categoricals: Optional[list[str]] = None,
    ordinals: Optional[list[str]] = None,
    _warn: bool = True,
) -> tuple[InspectionResults, InspectionResults]:
    """
    Attempt to infer column types

    Notes
    -----
    We must infer types of columns not specified as ordinal or categorical by
    the user in order to determine one-hot encoding vs. normalization. However,
    we must also check that user-specified categoricals and ordinals can in
    fact be treated as such.

    There are thus two classes of features: user-specified and unspecified.
    For unspecified features, we need only resolve *ambiguous* types. However,
    for specified features

    .                             all columns
    .                                 |
    .                                 v
    .            user-specified ------------- unspecified
    .     _____________|______                _____|________
    .     |                  |                |            |
    .     v                  v                v            v
    . infer-agree     infer-disagree       ambiguous   unambiguous
    .     |                  |                |            |
    .     v                  |                v            v
    . final type             |             coercion    final type
    .                        |                |
    .     ___________________|________        v
    .     |                          |    final type
    .     v                          v
    .    user                      user
    .   ordinal                 categorical
    .     |                          |
    .     v                          v
    .   infer                      infer
    .     |___________        _______|____________________________
    .     |          |        |              |                   |
    .  certain   uncertain    |              |                   |
    .     |         cat       |              |                   |
    .     |         or     certain       uncertain             ambig
    .     |        ambig      |             ord               cat/ord
    .     |          |        |         _____|___         _______|________
    .     v          v        v         |       |         |
    .  remove      trust    remove     big     ???       big
    .    or        user       or       cat               cat
    .  coerce        |      coerce      |
    .                v                  |
    .           final ordinal        deflate
    .


    """
    categoricals = categoricals or []
    ordinals = ordinals or []

    df = convert_categoricals(df, target)
    df = df.drop(columns=target)

    cats, ords = set(categoricals), set(ordinals)

    all_cols = set(df.columns.to_list())
    user_cols = [col for col in all_cols if ((col in cats) or (col in ords))]
    unk_cols = list(all_cols.difference(user_cols))

    floats, ords, ids, times, cats, consts = inspect_str_columns(
        df, unk_cols, categoricals=categoricals, ordinals=ordinals, _warn=_warn
    )
    user_floats, user_ords, user_ids, user_times, user_cats, user_consts = inspect_str_columns(
        df, user_cols, categoricals=[], ordinals=[], _warn=_warn
    )

    # all_consts = InspectionInfo.merge(consts, int_consts, other_consts)
    all_consts = InspectionInfo.merge(consts, user_consts)
    all_cats = [*categoricals, *cats.descs.keys()]
    unique_counts, nanless_cnts = get_unq_counts(df=df, target=target)
    bigs, inflation = detect_big_cats(df, unique_counts, all_cats, _warn=_warn)[:-1]
    user_bigs, user_inflation = detect_big_cats(df, unique_counts, categoricals, _warn=_warn)[:-1]

    bin_cats = {cat for cat, cnt in nanless_cnts.items() if cnt == 2}
    nyan_cats = {cat for cat, cnt in nanless_cnts.items() if cnt == 1}
    multi_cats = {cat for cat, cnt in nanless_cnts.items() if cnt > 2}

    user_bin_cats = {cat for cat, cnt in nanless_cnts.items() if cnt == 2}
    user_nyan_cats = {cat for cat, cnt in nanless_cnts.items() if cnt == 1}
    user_multi_cats = {cat for cat, cnt in nanless_cnts.items() if cnt > 2}

    bin_cats = sorted(bin_cats.intersection(all_cats))
    nyan_cats = sorted(nyan_cats.intersection(all_cats))
    multi_cats = sorted(multi_cats.intersection(all_cats))

    user_bin_cats = sorted(user_bin_cats.intersection(categoricals))
    user_nyan_cats = sorted(user_nyan_cats.intersection(categoricals))
    user_multi_cats = sorted(user_multi_cats.intersection(categoricals))

    nyan_cols = {col: "Constant when dropping NaNs" for col in nyan_cats}
    nyan_info = InspectionInfo(ColumnType.Nyan, nyan_cols)

    user_nyan_cols = {col: "Constant when dropping NaNs" for col in user_nyan_cats}
    user_nyan_info = InspectionInfo(ColumnType.Nyan, user_nyan_cols)

    all_ordinals = set(ords.descs.keys()).union(int_ords.descs.keys())
    ambiguous = all_ordinals.intersection(cats.descs.keys())
    ambiguous.difference_update(categoricals)
    ambiguous.difference_update(ordinals)
    ambiguous = sorted(ambiguous)
    if len(ambiguous) > 0:
        float_coerced, ord_coerced, cat_coerced = coerce_ambiguous_cols(
            df, ambigs=ambiguous, _warn=_warn
        )
        messy_inform(
            "df-analyze could not determine the types of some features. These "
            "have been coerced to our best guess for the appropriate type. See "
            "reports for details. This warning cannot be silenced unless you "
            "properly identify these features with the `--categoricals` and "
            "`--ordinals` options to df-analyze.\n\n"
            f"Ambiguous columns: {ambiguous}."
        )
        floats = InspectionInfo.merge(floats, float_coerced)
        ords = InspectionInfo.merge(ords, ord_coerced)
        cats = InspectionInfo.merge(cats, cat_coerced)

    main_results = InspectionResults(
        conts=floats,
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
        nyan_cats=nyan_info,
        multi_cats=multi_cats,
    )
    check_results = InspectionResults(
        conts=user_floats,
        ords=user_ords,
        ids=user_ids,
        times=user_times,
        consts=user_consts,
        cats=user_cats,
        int_ords={},  # type: ignore
        int_ids={},  # type: ignore
        big_cats=user_bigs,
        inflation=user_inflation,
        bin_cats=user_bin_cats,
        nyan_cats=user_nyan_info,
        multi_cats=user_multi_cats,
    )
    return main_results, check_results
