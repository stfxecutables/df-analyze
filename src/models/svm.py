from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Callable, Mapping, Optional

from optuna import Trial

from pandas import DataFrame, Series  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import time
from warnings import catch_warnings, filterwarnings

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate as cv
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

from src._constants import SEED
from src.models.base import NEG_MAE, DfAnalyzeModel


class SVMEstimator(DfAnalyzeModel):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.needs_calibration = True
        self.fixed_args = dict(cache_size=1000)

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            kernel=trial.suggest_categorical("kernel", choices=["rbf", "linear"]),
            # kernel=trial.suggest_categorical("kernel", choices=["rbf"]),
            # combination of large C and small gamma can multiply training time
            # by orders of magnitude...
            C=trial.suggest_float("C", 1e-10, 1e3, log=True),
            gamma=trial.suggest_float("gamma", 1e-7, 1e5, log=True),
        )

    def optuna_objective(
        self, X_train: DataFrame, y_train: Series, n_folds: int = 3
    ) -> Callable[[Trial], float]:
        def objective(trial: Trial) -> float:
            kf = StratifiedKFold if self.is_classifier else KFold
            _cv = kf(n_splits=n_folds, shuffle=True, random_state=SEED)
            opt_args = self.optuna_args(trial)
            args = {**self.fixed_args, **self.default_args, **opt_args}
            is_rbf = opt_args["kernel"] = "rbf"
            # plain SVC is excruciatingly slow even on small datasets for many
            # C and gamma values. So we switch to Linear here despite the fact
            # that, in theory, it confounds Optuna by ignoring a lot of gamma
            # values.
            # TODO: Switch to SGDClassifer and Kernel Approximation here
            # for large N
            if self.is_classifier:
                model_cls = SVC if is_rbf else LinearSVC
            else:
                model_cls = SVR if is_rbf else LinearSVR
            if not is_rbf:
                args.pop("cache_size")
                args.pop("gamma")
                args.pop("kernel")
            estimator = model_cls(**args)
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


class SVMClassifier(SVMEstimator):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = SVC

    def fit(self, X_train: DataFrame, y_train: Series) -> None:
        if self.model is None:
            kwargs = {**self.fixed_args, **self.model_args}
            self.model = self.model_cls(**kwargs)
        self.model.fit(X_train, y_train)


class SVMRegressor(SVMEstimator):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = SVR
