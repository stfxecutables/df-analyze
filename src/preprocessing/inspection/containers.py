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
    Continuous = "cont"
    Ordinal = "ord"
    Categorical = "cat"
    BigCat = "big_cat"
    Nyan = "const_nan"
    UserCategorical = "user_cat"
    UserOrdinal = "user_ord"
    Other = "other"

    def fmt(self, info: str) -> str:
        fmt = {
            ColumnType.Id: ID_INFO,
            ColumnType.Time: TIME_INFO,
            ColumnType.Const: CONST_INFO,
            ColumnType.Continuous: FLOAT_INFO,
            ColumnType.Ordinal: ORD_INFO,
            ColumnType.Categorical: CAT_INFO,
            ColumnType.BigCat: BIG_INFO,
            ColumnType.UserCategorical: "User-specified categorical",
            ColumnType.UserOrdinal: "User-specified ordinal",
            ColumnType.Nyan: NYAN_INFO,
            ColumnType.Other: None,
        }[self]
        if fmt is None:
            raise ValueError(f"Formatting not implemeted for enum {self}")
        if self in [ColumnType.UserCategorical, ColumnType.UserOrdinal]:
            return fmt

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
    cont: Optional[str] = None
    cat: Optional[str] = None


def get_width(*cols: dict[str, str]) -> int:
    all_cols = {}
    for d in cols:
        all_cols.update(d)
    if len(all_cols) > 0:
        return max(len(col) for col in all_cols) + 2
    return 0


class InspectionInfo:
    """For when df-analyze detects (but does not resolve) data problems"""

    def __init__(
        self,
        kind: ColumnType,
        descs: dict[str, str],
    ) -> None:
        self.kind = kind
        self.descs = descs
        self.pad = get_width(self.descs)
        self.is_empty = len(self.descs) == 0

    def lines(self, pad: Optional[int] = None) -> list[str]:
        pad = pad or self.pad
        return [f"{col:<{pad}} {desc}" for col, desc in self.descs.items()]

    def print_message(self, pad: Optional[int] = None) -> None:
        if self.is_empty:
            return

        cols = get_terminal_size((81, 24))[0]
        sep = "=" * cols
        underline = "." * (len(self.__class__.__name__) + 1)
        info = "\n".join(self.lines(pad))
        message = self.kind.fmt(info)
        formatted = f"\n{sep}\n{self.__class__.__name__}\n{underline}\n{message}\n{sep}"
        print(formatted, file=sys.stderr)

    def to_df(self) -> DataFrame:
        kind = self.kind.value
        df = DataFrame(self.descs.items(), columns=["feature_name", "reason"])
        df.insert(1, "inferred", kind)
        return df

    def __str__(self) -> str:
        cname = f"{self.kind.name}{self.__class__.__name__}"
        cols = [*self.descs.keys()]
        if len(cols) > 3:
            fmt = str(cols[:3]).replace("]", ", ... ")
        else:
            fmt = str(cols[:3])
        fmt = fmt.replace("[", "").replace("]", "")

        return f"{cname}({fmt})"

    __repr__ = __str__

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


@dataclass
class InspectionResults:
    conts: InspectionInfo
    ords: InspectionInfo
    ids: InspectionInfo
    times: InspectionInfo
    consts: InspectionInfo
    cats: InspectionInfo
    nyan_cats: InspectionInfo
    inflation: list[InflationInfo]
    big_cats: dict[str, int]
    bin_cats: list[str]
    multi_cats: list[str]

    def ordered_basic_infos(
        self
    ) -> tuple[
        InspectionInfo,
        InspectionInfo,
        InspectionInfo,
        InspectionInfo,
        InspectionInfo,
        InspectionInfo,
    ]:
        return (
            self.ids,
            self.times,
            self.consts,
            self.conts,
            self.ords,
            self.cats,
        )

    def print_basic_infos(self, pad: Optional[int] = None) -> None:
        basics = self.ordered_basic_infos()
        pad = pad or get_width(*[basic.descs for basic in basics])
        for info in basics:
            info.print_message(pad)

    def basic_df(self) -> DataFrame:
        dfs = []
        for info in self.ordered_basic_infos():
            dfs.append(info.to_df())
        df = pd.concat(dfs, axis=0, ignore_index=True)
        idx = df["reason"].str.contains("Coerced")
        coerced = df[idx].sort_values(by="feature_name")
        df = df[~idx].sort_values(by="feature_name")
        return pd.concat([df, coerced], axis=0, ignore_index=True)
