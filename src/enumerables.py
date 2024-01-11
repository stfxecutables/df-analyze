from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, EnumMeta
from math import isnan
from random import choice, randint
from typing import TYPE_CHECKING, Any, Generic, Optional, Type, TypeVar, no_type_check
from warnings import warn

import numpy as np
from numpy import ndarray
from pandas import Series
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)

from src.scoring import (
    sensitivity,
    specificity,
)

if TYPE_CHECKING:
    from src.models.base import DfAnalyzeModel

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

    @classmethod
    def random_none(cls: Type[T]) -> Optional[T]:
        if not isinstance(cls, EnumMeta):
            raise ValueError("Undefined")
        return choice([*cls, None])  # type: ignore

    @classmethod
    def choices(cls: Type[T]) -> list[str]:
        return [x.value for x in cls]  # type: ignore

    @no_type_check
    def __lt__(self: T, other: Type[T]) -> bool:
        cls = self.__class__
        if not isinstance(other, cls):
            raise ValueError(f"Can only compare {cls} to other objects of type {cls}")
        return self.name < other.name  # type: ignore

    def __gt__(self, other):
        return not (self < other)


class DfAnalyzeClassifier(RandEnum, Enum):
    KNN = "knn"
    LGBM = "lgbm"
    RF = "rf"
    LR = "lr"
    SGD = "sgd"
    MLP = "mlp"
    SVM = "svm"
    Dummy = "dummy"

    def get_model(self) -> Type[DfAnalyzeModel]:
        from src.models.dummy import DummyClassifier
        from src.models.knn import KNNClassifier
        from src.models.lgbm import (
            LightGBMClassifier,
            LightGBMRFClassifier,
        )
        from src.models.linear import LRClassifier, SGDClassifier
        from src.models.mlp import MLPEstimator
        from src.models.svm import SVMClassifier

        return {
            DfAnalyzeClassifier.KNN: KNNClassifier,
            DfAnalyzeClassifier.LGBM: LightGBMClassifier,
            DfAnalyzeClassifier.RF: LightGBMRFClassifier,
            DfAnalyzeClassifier.LR: LRClassifier,
            DfAnalyzeClassifier.SGD: SGDClassifier,
            DfAnalyzeClassifier.MLP: MLPEstimator,
            DfAnalyzeClassifier.SVM: SVMClassifier,
            DfAnalyzeClassifier.Dummy: DummyClassifier,
        }[self]

    @staticmethod
    def defaults() -> tuple[DfAnalyzeClassifier, ...]:
        return (
            DfAnalyzeClassifier.Dummy,
            DfAnalyzeClassifier.KNN,
            DfAnalyzeClassifier.LGBM,
            DfAnalyzeClassifier.SGD,
        )


class DfAnalyzeRegressor(RandEnum, Enum):
    KNN = "knn"
    LGBM = "lgbm"
    RF = "rf"
    ElasticNet = "elastic"
    SGD = "sgd"
    MLP = "mlp"
    SVM = "svm"
    Dummy = "dummy"

    def get_model(self) -> Type[DfAnalyzeModel]:
        from src.models.dummy import DummyRegressor
        from src.models.knn import KNNRegressor
        from src.models.lgbm import (
            LightGBMRegressor,
            LightGBMRFRegressor,
        )
        from src.models.linear import ElasticNetRegressor, SGDRegressor
        from src.models.mlp import MLPEstimator
        from src.models.svm import SVMRegressor

        return {
            DfAnalyzeRegressor.KNN: KNNRegressor,
            DfAnalyzeRegressor.LGBM: LightGBMRegressor,
            DfAnalyzeRegressor.RF: LightGBMRFRegressor,
            DfAnalyzeRegressor.ElasticNet: ElasticNetRegressor,
            DfAnalyzeRegressor.SGD: SGDRegressor,
            DfAnalyzeRegressor.MLP: MLPEstimator,
            DfAnalyzeRegressor.SVM: SVMRegressor,
            DfAnalyzeRegressor.Dummy: DummyRegressor,
        }[self]

    @staticmethod
    def defaults() -> tuple[DfAnalyzeRegressor, ...]:
        return (
            DfAnalyzeRegressor.Dummy,
            DfAnalyzeRegressor.KNN,
            DfAnalyzeRegressor.LGBM,
            DfAnalyzeRegressor.SGD,
        )


@dataclass
class ClassifierScorer(RandEnum, Enum):
    Accuracy = "acc"
    AUROC = "auroc"
    Sensitivity = "sens"
    Specificity = "spec"
    F1 = "f1"
    BalanceAccuracy = "bal-acc"

    @staticmethod
    def get_scores(y_true: Series, y_pred: Series, y_prob: ndarray) -> dict[str, float]:
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        raws = {
            ClassifierScorer.Accuracy.value: accuracy_score(y_true, y_pred),
            ClassifierScorer.AUROC.value: roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovr"
            ),
            ClassifierScorer.Sensitivity.value: sensitivity(y_true, y_pred),
            ClassifierScorer.Specificity.value: specificity(y_true, y_pred),
            ClassifierScorer.F1.value: f1_score(y_true, y_pred, average="macro"),
            ClassifierScorer.BalanceAccuracy.value: balanced_accuracy_score(y_true, y_pred),
        }
        return {raw: float(value) for raw, value in raws.items()}

    @staticmethod
    def null_scores() -> dict[str, float]:
        raws = {
            ClassifierScorer.Accuracy.value: np.nan,
            ClassifierScorer.AUROC.value: np.nan,
            ClassifierScorer.Sensitivity.value: np.nan,
            ClassifierScorer.Specificity.value: np.nan,
            ClassifierScorer.F1.value: np.nan,
            ClassifierScorer.BalanceAccuracy.value: np.nan,
        }
        return {raw: float(value) for raw, value in raws.items()}


@dataclass
class RegressorScorer(RandEnum, Enum):
    MAE = "mae"
    MSqE = "msqe"
    MdAE = "mdae"
    R2 = "r2"
    VarExp = "var-exp"

    @staticmethod
    def get_scores(y_true: Series, y_pred: Series) -> dict[str, float]:
        raws = {
            RegressorScorer.MAE.value: mean_absolute_error(y_true, y_pred),
            RegressorScorer.MSqE.value: mean_squared_error(y_true, y_pred),
            RegressorScorer.MdAE.value: median_absolute_error(y_true, y_pred),
            RegressorScorer.R2.value: r2_score(y_true, y_pred),
            RegressorScorer.VarExp.value: explained_variance_score(y_true, y_pred),
        }
        return {raw: float(value) for raw, value in raws.items()}

    @staticmethod
    def null_scores() -> dict[str, float]:
        raws = {
            RegressorScorer.MAE.value: np.nan,
            RegressorScorer.MSqE.value: np.nan,
            RegressorScorer.MdAE.value: np.nan,
            RegressorScorer.R2.value: np.nan,
            RegressorScorer.VarExp.value: np.nan,
        }
        return {raw: float(value) for raw, value in raws.items()}


class EmbedSelectionModel(RandEnum, Enum):
    LGBM = "lgbm"
    Linear = "linear"


class NanHandling(RandEnum, Enum):
    Drop = "drop"
    Mean = "mean"
    Median = "median"
    Impute = "impute"


class Normalization(RandEnum, Enum):
    MinMax = "minmax"
    Robust = "robust"


class EstimationMode(RandEnum, Enum):
    Classify = "classify"
    Regress = "regress"


class FeatureCleaning(RandEnum, Enum):
    Correlated = "correlated"
    LowInfo = "lowinfo"


class FeatureSelection(RandEnum, Enum):
    Filter = "filter"
    Embedded = "embed"
    Wrapper = "wrap"


class ModelFeatureSelection(RandEnum, Enum):
    Embedded = "embed"
    Wrapper = "wrap"
    NoSelection = "none"


class WrapperSelection(RandEnum, Enum):
    StepUp = "step-up"
    StepDown = "step-down"
    # Genetic = "genetic"
    # ParticleSwarm = "swarm"


class WrapperSelectionModel(RandEnum, Enum):
    Linear = "linear"
    LGBM = "lgbm"


class DimensionReduction(RandEnum, Enum):
    PCA = "pca"
    kPCA = "kpca"


class FilterSelection(RandEnum, Enum):
    Relief = "relief"
    Association = "assoc"
    Prediction = "pred"


class EmbeddedSelection(RandEnum, Enum):
    LightGBM = "lgbm"
    LASSO = "lasso"


class RegScore(RandEnum, Enum):
    MAE = "mae"
    MSqE = "msqe"
    MdAE = "mdae"
    R2 = "r2"
    VarExp = "var-exp"

    @staticmethod
    def default() -> RegScore:
        return RegScore.MAE

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

    def longname(self) -> str:
        return {
            RegScore.MAE: "Mean Absolute Error",
            RegScore.MSqE: "Mean Squared Error",
            RegScore.MdAE: "Median Absolute Error",
            RegScore.R2: "R-Squared",
            RegScore.VarExp: "Percent Variance Explained",
        }[self]


class ClsScore(RandEnum, Enum):
    Accuracy = "acc"
    AUROC = "auroc"
    Sensitivity = "sens"
    Specificity = "spec"

    @staticmethod
    def default() -> ClsScore:
        return ClsScore.Accuracy

    def higher_is_better(self) -> bool:
        return True

    def minimum(self) -> float:
        return {
            ClsScore.Accuracy: 0.0,
            ClsScore.AUROC: 0.5,
            ClsScore.Sensitivity: 0.0,
            ClsScore.Specificity: 0.0,
        }[self]

    def longname(self) -> str:
        return {
            ClsScore.Accuracy: "Accuracy",
            ClsScore.AUROC: "Area Under the ROC Curve",
            ClsScore.Sensitivity: "Sensitivity",
            ClsScore.Specificity: "Specificity",
        }[self]


class EstimatorKind(RandEnum, Enum):
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
