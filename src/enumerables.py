from __future__ import annotations

from argparse import ArgumentError
from enum import Enum
from math import isnan
from typing import Any, Generic, TypeVar
from warnings import warn

T = TypeVar("T")


class CliArgument(Enum, Generic[T]):
    def to_string(self, value: T | None = None) -> str:
        return self.value

    @staticmethod
    def from_str(s: str) -> CliArgument:
        return CliArgument(s)


class CVSplit(CliArgument):
    KFold3 = "3-fold"
    KFold5 = "5-fold"
    KFold10 = "10-fold"
    KFold20 = "20-fold"
    Holdout = "holdout"

    def to_string(self, value: float) -> str:
        if self is not CVSplit.Holdout:
            return self.value
        return f"{value*100}%-holdout"

    @staticmethod
    def from_str(s: str) -> CliArgument:
        try:
            cv = float(s)
        except Exception as e:
            raise ValueError(
                "Could not convert a `... -size` argument (e.g. --htune-val-size) value to float"
            ) from e
        # validate
        if isnan(cv):
            raise ValueError("NaN is not a valid size")
        if cv <= 0:
            raise ValueError("`... -size` arguments (e.g. --htune-val-size) must be positive")
        if cv == 1:
            raise ValueError(
                "'1' is not a valid value for `... -size` arguments (e.g. --htune-val-size)."
            )

        if 0 < cv < 1:
            return CVSplit.Holdout

        if (cv > 1) and not cv.is_integer():
            raise ValueError(
                "`... -size` arguments (e.g. --htune-val-size) greater than 1, as it specifies the `k` in k-fold"
            )

        if cv != round(cv):
            raise ValueError(
                "`--htune-val-size` must be an integer if greater than 1, as it specifies the `k` in k-fold"
            )
        if cv > 10:
            warn(
                "`--htune-val-size` greater than 10 is not recommended.",
                category=UserWarning,
            )
        if cv > 1:
            return int(cv)

        return CliArgument(s)
