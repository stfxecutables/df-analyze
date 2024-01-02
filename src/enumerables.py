from __future__ import annotations

from enum import Enum, EnumMeta
from math import isnan
from random import choice, randint
from typing import Any, Generic, Type, TypeVar
from warnings import warn

import numpy as np

T = TypeVar("T", bound="RandEnum")


class RandEnum(Generic[T]):
    @classmethod
    def random(cls: Type[T]) -> T:
        if not isinstance(cls, EnumMeta):
            raise ValueError("Undefined")
        return choice([*cls])  # type: ignore

    @classmethod
    def random_n(cls: Type[T]) -> tuple[T, ...]:
        n = randint(1, len(cls) - 1)  # type: ignore
        if not isinstance(cls, EnumMeta):
            raise ValueError("Undefined")
        return tuple(np.random.choice([*cls], size=n, replace=False).tolist())  # type: ignore


class DfAnalyzeClassifier(RandEnum, Enum):
    KNN = "knn"
    LGBM = "lgbm"
    LR = "lr"
    SGD = "sgd"
    MLP = "mlp"
    SVM = "svm"


class DfAnalyzeRegressor(RandEnum, Enum):
    KNN = "knn"
    LGBM = "lgbm"
    ElasticNet = "elastic"
    SGD = "sgd"
    MLP = "mlp"
    SVM = "svm"


class NanHandling(RandEnum, Enum):
    Drop = "drop"
    Mean = "mean"
    Median = "median"
    Impute = "impute"


class EstimationMode(RandEnum, Enum):
    Classify = "classify"
    Regress = "regress"


class FeatureCleaning(RandEnum, Enum):
    Constant = "constant"
    Correlated = "correlated"
    LowInfo = "lowinfo"


class FeatureSelection(RandEnum, Enum):
    Minimal = "minimal"
    StepDown = "step-down"
    StepUp = "step-up"
    PCA = "pca"
    kPCA = "kpca"


class WrapperSelection(RandEnum, Enum):
    StepUp = "step-up"
    StepDown = "step-down"


class FilterSelection(RandEnum, Enum):
    Relief = "relief"
    Association = "assoc"
    Prediction = "pred"


class EmbeddedSelection(RandEnum, Enum):
    LightGBM = "lgbm"
    LASSO = "lasso"


class RegScore(RandEnum, Enum):
    MAE = "MAE"
    MSqE = "MSqE"
    MdAE = "MdAE"
    R2 = "R2"
    VarExp = "Var exp"

    # MAPE = "MAPE"
    def minimum(self) -> float:
        return {
            RegScore.MAE: 0.0,
            RegScore.MSqE: 0.0,
            RegScore.MdAE: 0.0,
            RegScore.R2: -np.inf,
            RegScore.VarExp: 0.0,
        }[self]

    def higher_is_better(self) -> bool:
        return self in [RegScore.R2, RegScore.VarExp]


class ClsScore(Enum, RandEnum):
    Accuracy = "acc"
    AUROC = "auroc"
    Sensitivity = "sens"
    Specificity = "spec"

    def higher_is_better(self) -> bool:
        return True

    def minimum(self) -> float:
        return {
            ClsScore.Accuracy: 0.0,
            ClsScore.AUROC: 0.5,
            ClsScore.Sensitivity: 0.0,
            ClsScore.Specificity: 0.0,
        }[self]


class EstimatorKind(Enum, RandEnum):
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
