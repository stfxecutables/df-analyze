from __future__ import annotations

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
from src.preprocessing.inspection.inference import (
    looks_categorical,
    looks_constant,
    looks_floatlike,
    looks_id_like,
    looks_ordinal,
    looks_timelike,
)
from src.preprocessing.inspection.text import (
    BIG_INFO,
    CAT_INFO,
    CONST_INFO,
    FLOAT_INFO,
    ID_INFO,
    NYAN_INFO,
    ORD_INFO,
    TIME_INFO,
)


class ColumnType(Enum):
    Id = "id"
    Time = "time"
    Const = "const"
    Float = "float"
    Ordinal = "ord"
    Categorical = "cat"
    BigCat = "big_cat"
    Nyan = "const_nan"
    Other = "other"

    def fmt(self, info: str) -> str:
        fmt = {
            ColumnType.Id: ID_INFO,
            ColumnType.Time: TIME_INFO,
            ColumnType.Const: CONST_INFO,
            ColumnType.Float: FLOAT_INFO,
            ColumnType.Ordinal: ORD_INFO,
            ColumnType.Categorical: CAT_INFO,
            ColumnType.BigCat: BIG_INFO,
            ColumnType.Other: NYAN_INFO,
        }[self]
        return fmt.format(info=info)


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


class InspectionInfo:
    """For when df-analyze detects (but does not resolve) data problems"""

    def __init__(
        self,
        kind: ColumnType,
        descs: dict[str, str],
    ) -> None:
        self.kind = kind
        self.descs = descs
        self.pad = self.get_width(self.descs)
        self.lines = [f"{col:<{self.pad}} {desc}" for col, desc in self.descs.items()]
        self.is_empty = len(self.descs) == 0

    def print_message(self) -> None:
        if self.is_empty:
            return

        cols = get_terminal_size((81, 24))[0]
        sep = "=" * cols
        underline = "." * (len(self.__class__.__name__) + 1)
        info = "\n".join(self.lines)
        message = self.kind.fmt(info)
        formatted = f"\n{sep}\n{self.__class__.__name__}\n{underline}\n{message}\n{sep}"
        print(formatted, file=sys.stderr)

    @staticmethod
    def merge(*infos: InspectionInfo) -> InspectionInfo:
        if len(infos) == 0:
            raise ValueError("Nothing to merge")
        kind = infos[0].kind
        if not all(info.kind == kind for info in infos):
            raise ValueError("Cannot merge when kinds differ.")
        descs = {}
        for info in infos:
            descs.update(info.descs)
        return InspectionInfo(kind=kind, descs=descs)

    def get_width(self, *cols: dict[str, str]) -> int:
        all_cols = {}
        for d in cols:
            all_cols.update(d)
        if len(all_cols) > 0:
            return max(len(col) for col in all_cols) + 2
        return 0


@dataclass
class InspectionResults:
    floats: InspectionInfo
    ords: InspectionInfo
    ids: InspectionInfo
    times: InspectionInfo
    consts: InspectionInfo
    cats: InspectionInfo
    int_ords: InspectionInfo
    int_ids: InspectionInfo
    big_cats: dict[str, int]
    nyan_cats: InspectionInfo
    inflation: list[InflationInfo]
    bin_cats: list[str]
    multi_cats: list[str]


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

    float_info = InspectionInfo(ColumnType.Float, float_cols)
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

    # all_consts = {**consts.descs, **int_consts.descs, **other_consts.descs}
    all_consts = InspectionInfo.merge(consts, int_consts, other_consts)
    all_cats = [*categoricals, *cats.descs.keys()]
    unique_counts, nanless_cnts = get_unq_counts(df=df, target=target)
    bigs, inflation, big_info = detect_big_cats(df, unique_counts, all_cats, _warn=_warn)

    bin_cats = {cat for cat, cnt in nanless_cnts.items() if cnt == 2}
    nyan_cats = {cat for cat, cnt in nanless_cnts.items() if cnt == 1}
    multi_cats = {cat for cat, cnt in nanless_cnts.items() if cnt > 2}

    bin_cats = sorted(bin_cats.intersection(all_cats))
    nyan_cats = sorted(nyan_cats.intersection(all_cats))
    multi_cats = sorted(multi_cats.intersection(all_cats))

    nyan_cols = {col: "Constant when dropping NaNs" for col in nyan_cats}
    nyan_info = InspectionInfo(ColumnType.Nyan, nyan_cols)

    all_ordinals = set(ords.descs.keys()).union(int_ords.descs.keys())
    ambiguous = all_ordinals.intersection(cats.descs.keys())
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
        nyan_cats=nyan_info,
        multi_cats=multi_cats,
    )
