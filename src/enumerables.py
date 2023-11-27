from __future__ import annotations

from enum import Enum
from math import isnan
from typing import Any
from warnings import warn


class NanHandling(Enum):
    Drop = "drop"
    Mean = "mean"
    Median = "median"
    Impute = "impute"


class WrapperSelection(Enum):
    StepUp = "step-up"
    StepDown = "step-down"


class FilterSelection(Enum):
    Relief = "relief"
    Univariate = "univariate"


class EstimatorKind(Enum):
    Linear = "lin"
    SVM = "svm"
    KNN = "knn"
    RF = "rf"
    LGBM = "lgbm"
    MLP = "mlp"

    def regressor(self) -> Any:
        """Return the regressor form"""

    def classifier(self) -> Any:
        """Return the classifier form"""


class CVSplit(Enum):
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
    def from_str(s: str) -> CVSplit:
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
            return CVSplit(cv)

        return CVSplit(s)
