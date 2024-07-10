from __future__ import annotations

# fmt: off
from logging import ERROR
import sys  # isort: skip
from pathlib import Path
from typing import Any, Mapping, Optional

import optuna
from optuna import Study, Trial

from df_analyze.enumerables import Scorer

ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from pandas import DataFrame, Series
from sklearn.dummy import DummyClassifier as SklearnDummyClassifier
from sklearn.dummy import DummyRegressor as SklearnDummyRegressor

from df_analyze.models.base import DfAnalyzeModel


class DummyEstimator(DfAnalyzeModel):
    shortname = "dummy-est"
    longname = "Dummy Estimator"

    def htune_optuna(
        self,
        X_train: DataFrame,
        y_train: Series,
        metric: Scorer,
        n_trials: int = 100,
        n_jobs: int = -1,
        verbosity: int = optuna.logging.ERROR,
    ) -> Study:
        """
        Too many jobs will blow memory and also since all grids are at most 4
        waste time and compute.
        """
        return super().htune_optuna(
            X_train, y_train, metric, n_trials, n_jobs=4, verbosity=verbosity
        )


class DummyRegressor(DummyEstimator):
    shortname = "dummy"
    longname = "Dummy Regressor"
    timeout_s = 5 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = SklearnDummyRegressor
        self.fixed_args = dict()
        self.grid = {
            "strategy": ["mean", "median"],
        }

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            strategy=trial.suggest_categorical(
                "strategy", ["mean", "median", "quantile"]
            ),
        )


class DummyClassifier(DummyEstimator):
    shortname = "dummy"
    longname = "Dummy Classifier"
    timeout_s = 5 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = SklearnDummyClassifier
        self.fixed_args = dict()
        self.grid = {"strategy": ["most_frequent", "prior", "stratified", "uniform"]}

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            strategy=trial.suggest_categorical(
                "strategy", ["most_frequent", "prior", "stratified", "uniform"]
            ),
        )
