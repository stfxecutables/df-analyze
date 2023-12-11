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
from pandas import DataFrame, Index, Series
from pandas.core.dtypes.dtypes import CategoricalDtype
from sklearn.experimental import enable_iterative_imputer  # noqa
from tqdm import tqdm

from src._constants import N_CAT_LEVEL_MIN
from src.preprocessing.inspection.inference import Inference, InferredKind
from src.preprocessing.inspection.text import (
    BIG_INFO,
    BINARY_INFO,
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
    Binary = "bin"
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
            ColumnType.Binary: BINARY_INFO,
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
    const: Optional[Inference] = None
    time: Optional[Inference] = None
    bin: Optional[Inference] = None
    ord: Optional[Inference] = None
    id: Optional[Inference] = None
    cont: Optional[Inference] = None
    cat: Optional[Inference] = None


def get_width(*cols: dict[str, str]) -> int:
    all_cols = {}
    for d in cols:
        all_cols.update(d)
    if len(all_cols) > 0:
        return max(len(col) for col in all_cols) + 2
    return 0


class InspectionInfo:
    """Container holding ColumnType and Inferences for that column"""

    def __init__(
        self,
        kind: ColumnType,
        infos: dict[str, Inference],
    ) -> None:
        self.kind = kind
        self.infos = infos
        self.cols = set([*self.infos.keys()])
        self.pad = get_width({col: info.reason for col, info in self.infos.items()})
        self.is_empty = len(self.infos) == 0

    def lines(self, pad: Optional[int] = None) -> list[str]:
        pad = pad or self.pad
        return [f"{col:<{pad}} {info.reason}" for col, info in self.infos.items()]

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
        df = DataFrame(
            [[col, infer.kind.value, infer.reason] for col, infer in self.infos.items()],
            columns=["feature_name", "inferred", "reason"],
        )
        return df

    def __str__(self) -> str:
        cname = f"{self.kind.name}{self.__class__.__name__}"
        cols = [*self.infos.keys()]
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
        new_infos = {}
        for info in infos:
            new_infos.update(info.infos)
        return InspectionInfo(kind=kind, infos=new_infos)

    @staticmethod
    def conflicts(
        *infos: InspectionInfo
    ) -> tuple[dict[str, list[Inference]], dict[str, list[Inference]]]:
        cols = set()
        for info in infos:
            cols.update(info.cols)

        certains: dict[str, list[Inference]] = {col: [] for col in cols}
        maybes: dict[str, list[Inference]] = {col: [] for col in cols}
        for col in cols:
            for info in infos:
                if col not in info.cols:
                    continue
                infer = info.infos[col]
                if infer.is_certain():
                    certains[col].append(infer)
                else:
                    maybes[col].append(infer)
        for col in certains:
            if len(certains[col]) > 1:
                raise RuntimeError(f"Conflicting certainties for column {col}: {certains[col]}")

        return certains, maybes


@dataclass
class InspectionResults:
    conts: InspectionInfo
    ords: InspectionInfo
    ids: InspectionInfo
    times: InspectionInfo
    consts: InspectionInfo
    cats: InspectionInfo
    binaries: InspectionInfo
    big_cats: dict[str, int]
    multi_cats: list[str]
    inflation: list[InflationInfo]
    user_cats: set[str]
    user_ords: set[str]

    def ordered_basic_infos(
        self
    ) -> tuple[
        InspectionInfo,
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
            self.binaries,
            self.conts,
            self.ords,
            self.cats,
        )

    def print_basic_infos(self, pad: Optional[int] = None) -> None:
        basics = self.ordered_basic_infos()
        pad = pad or max(info.pad for info in basics)
        for info in basics:
            info.print_message(pad)

    def basic_df(self) -> DataFrame:
        dfs = []
        for info in self.ordered_basic_infos():
            dfs.append(info.to_df())
        df = pd.concat(dfs, axis=0, ignore_index=True)
        df.index = Index(df["feature_name"].copy())
        df.drop(columns=["feature_name"], inplace=True)

        df.insert(0, "user", "")
        for col in self.user_cats:
            df.loc[col, "user"] = "cat"
        for col in self.user_ords:
            df.loc[col, "user"] = "ord"

        idx = df["reason"].str.contains("Coerced")
        coerced = df[idx].sort_index()
        df = df[~idx].sort_index().sort_values(by=["user", "inferred"])

        return pd.concat([df, coerced], axis=0, ignore_index=False)
