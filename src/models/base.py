from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Optional, Type, TypeVar, Union

import numpy as np
import optuna
from optuna import Study, Trial, create_study
from optuna.samplers import TPESampler
from pandas import DataFrame, Series
from sklearn.calibration import CalibratedClassifierCV as CVCalibrate
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate as cv

from src._constants import SEED

NEG_MAE = "neg_mean_absolute_error"


class DfAnalyzeModel(ABC):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__()
        self.is_classifier: bool = True
        self.needs_calibration: bool = False
        self.model_cls: Type[Any] = type(None)
        self.model: Optional[Any] = None
        self.fixed_args: dict[str, Any] = {}
        self.default_args: dict[str, Any] = {}
        self.model_args: Mapping = model_args or {}

        self.tuned_args: Optional[dict[str, Any]] = None
        self.tuned_model: Optional[Any] = None
        self.is_refit = False

    def optuna_objective(
        self, X_train: DataFrame, y_train: Series, n_folds: int = 3
    ) -> Callable[[Trial], float]:
        def objective(trial: Trial) -> float:
            kf = StratifiedKFold if self.is_classifier else KFold
            _cv = kf(n_splits=n_folds, shuffle=True, random_state=SEED)
            args = {**self.fixed_args, **self.default_args, **self.optuna_args(trial)}
            estimator = self.model_cls(**args)
            scoring = "accuracy" if self.is_classifier else NEG_MAE
            time.sleep(1)  # secs
            scores = cv(
                estimator,  # type: ignore
                X=X_train,
                y=y_train,
                scoring=scoring,
                cv=_cv,
                n_jobs=1,
            )
            time.sleep(1)  # secs
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
        n_jobs: int = -1,
        verbosity: int = optuna.logging.ERROR,
    ) -> Study:
        if self.tuned_args is not None:
            raise RuntimeError(
                f"Model {self.__class__.__name__} has already been tuned with Optuna"
            )

        study = create_study(direction="maximize", sampler=TPESampler())
        optuna.logging.set_verbosity(verbosity)
        objective = self.optuna_objective(X_train=X_train, y_train=y_train)
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        print("Optuna tuning completed")
        self.tuned_args = study.best_params
        self.refit(X=X_train, y=y_train, overrides=self.tuned_args)

        return study

    def fit(self, X_train: DataFrame, y_train: Series) -> None:
        if self.model is None:
            kwargs = {**self.fixed_args, **self.default_args, **self.model_args}
            self.model = self.model_cls(**kwargs)
        self.model.fit(X_train, y_train)

    def refit(self, X: DataFrame, y: Series, overrides: Optional[Mapping] = None) -> None:
        overrides = overrides or {}
        kwargs = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **overrides,
        }
        self.tuned_model = self.model_cls(**kwargs)

        if self.needs_calibration:
            self.tuned_model = CVCalibrate(self.tuned_model, method="sigmoid", cv=5, n_jobs=5)

        self.tuned_model.fit(X, y)

    def htune_eval(
        self,
        X_train: DataFrame,
        y_train: Series,
        X_test: DataFrame,
        y_test: Series,
    ) -> Any:
        # TODO: need to specify valiation method, and return confidences, etc.
        # Actually maybe just want to call refit in here...
        if self.tuned_args is None:
            raise RuntimeError("Cannot evaluate tuning because model has not been tuned.")

        self.refit(X_train, y_train, overrides=self.tuned_args)
        # TODO: return Platt-scaling or probability estimates
        return self.score(X_test, y_test)

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

    def predict_proba(self, X: DataFrame) -> Series:
        if not self.is_classifier:
            raise ValueError("Cannot get probabilities for a regression model.")

        if self.tuned_args is None or self.tuned_model is None:
            raise RuntimeError("Need to tune estimator before calling `.predict_proba()`")

        return self.tuned_model.predict_proba(X)
