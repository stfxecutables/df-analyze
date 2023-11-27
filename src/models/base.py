from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Type, Union

import numpy as np
import optuna
from optuna import Study, Trial, create_study
from optuna.samplers import TPESampler
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate as cv

from src._constants import SEED

NEG_MAE = "neg_mean_absolute_error"


class DfAnalyzeModel(ABC):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__()
        self.is_classifier: bool = True
        self.needs_proba: bool = False  # NOTE: maybe unneeded?
        self.model_cls: Type[Any]
        self.model: Optional[Any] = None
        self.fixed_args: dict[str, Any] = {}
        self.default_args: dict[str, Any] = {}
        self.model_args: Mapping = model_args or dict()

    def optuna_objective(
        self, X_train: DataFrame, y_train: Series, n_folds: int = 3
    ) -> Callable[[Trial], float]:
        def objective(trial: Trial) -> float:
            kf = StratifiedKFold if self.is_classifier else KFold
            _cv = kf(n_splits=n_folds, shuffle=True, random_state=SEED)
            args = {**self.fixed_args, **self.default_args, **self.optuna_args(trial)}
            estimator = self.model_cls(**args)
            scoring = "accuracy" if self.is_classifier else NEG_MAE
            scores = cv(
                estimator,  # type: ignore
                X=X_train,
                y=y_train,
                scoring=scoring,
                cv=_cv,
                n_jobs=1,
            )
            return float(np.mean(scores["test_score"]))

        return objective

    @abstractmethod
    def optuna_args(self, trial: Trial) -> dict[str, Union[str, float, int]]:
        return {}

    def htune_optuna(
        self,
        X_train: DataFrame,
        y_train: Series,
        n_trials: int = 100,
        verbosity: int = optuna.logging.ERROR,
    ) -> Study:
        study = create_study(direction="maximize", sampler=TPESampler())
        optuna.logging.set_verbosity(verbosity)
        objective = self.optuna_objective(X_train=X_train, y_train=y_train)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        return study

    def fit(self, X_train: DataFrame, y_train: Series) -> None:
        if self.model is None:
            kwargs = {**self.fixed_args, **self.default_args, **self.model_args}
            self.model = self.model_cls(**kwargs)
        self.model.fit(X_train, y_train)

    def score(self, X: DataFrame, y: Series) -> float:
        if self.model is None:
            raise RuntimeError("Need to call `model.fit()` before calling `.score()`")
        return self.model.score(X, y)

    def predict(self, X: DataFrame) -> Series:
        if self.model is None:
            raise RuntimeError("Need to call `model.fit()` before calling `.predict()`")
        return self.model.predict(X)

    def wrapper_select(
        self, X_train: DataFrame, y_train: Series, n_feat: int, method: str
    ) -> Series:
        raise NotImplementedError()

    def predict_proba(self, X: DataFrame, y: Series) -> Series:
        if not self.is_classifier:
            raise ValueError("Cannot get probabilities for a regression model.")

        if self.model is None:
            raise RuntimeError("Need to call `model.fit()` before calling `.predict_proba()`")

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"No `predict_proba` method found on model {self.model}")
        return self.model.predict_proba(X, y)
