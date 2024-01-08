from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from abc import ABC, abstractmethod
from math import ceil
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Type, Union

import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from optuna import Study, Trial, create_study
from optuna.logging import _get_library_root_logger
from optuna.pruners import MedianPruner
from optuna.samplers import GridSampler, TPESampler
from optuna.trial import FrozenTrial
from pandas import DataFrame, Series
from sklearn.calibration import CalibratedClassifierCV as CVCalibrate
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import KFold, StratifiedKFold

from src._constants import SEED
from src.enumerables import WrapperSelection
from src.hypertune import (
    ClassifierScorer,
    RegressorScorer,
)

NEG_MAE = "neg_mean_absolute_error"

OPT_LOGGER = _get_library_root_logger()


class EarlyStopping:
    # TODO: return k-fold values as they come in and do early stopping on
    # trials with a bad initial fold?
    def __init__(self, patience: int = 10, min_trials: int = 50) -> None:
        self.patience: int = patience
        self.min_trials: int = min_trials
        self.has_stopped: bool = False

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """https://github.com/optuna/optuna/issues/1001#issuecomment-1351766030"""
        if self.has_stopped:
            # raise optuna.exceptions.TrialPruned()
            study.stop()

        current_trial = trial.number
        if current_trial < self.min_trials:
            return
        best_trial = study.best_trial.number

        # best_score = study.best_value  # TODO: patience
        should_stop = (current_trial - best_trial) >= self.patience
        if should_stop:
            if not self.has_stopped:
                OPT_LOGGER.info(
                    f"Completed {self.patience} trials without metric improvement. "
                    f"Stopping early at trial {current_trial}. Some trials may still "
                    "need to finish, and produce a better result. If so, that better "
                    f"result will be used rather than trial {current_trial}."
                )
            self.has_stopped = True
            study.stop()


class DfAnalyzeModel(ABC):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__()
        self.is_classifier: bool = True
        self.needs_calibration: bool = False
        self.model: Optional[Any] = None
        self.fixed_args: dict[str, Any] = {}
        self.default_args: dict[str, Any] = {}
        self.model_args: Mapping = model_args or {}
        self.grid: Optional[dict[str, Any]] = None

        self.tuned_args: Optional[dict[str, Any]] = None
        self.tuned_model: Optional[Any] = None
        self.is_refit = False
        self.shortname = ""
        self.longname = ""

    @abstractmethod
    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[Type[Any], dict[str, Any]]:
        """Allows for conditioning the model based on args (e.g. SVC vs. LinearSVC
        depending on kernel, and also subsequent removal or addition of necessary
        args because of this.

        Returns
        -------
        model_cls: Type[Any]
            The model class needed based on `full_args`

        clean_args: dict[str, Any]
            The args that now work for the returned `model_cls`

        """
        ...

    def optuna_objective(
        self, X_train: DataFrame, y_train: Series, n_folds: int = 5
    ) -> Callable[[Trial], float]:
        X = np.asarray(X_train)
        y = np.asarray(y_train)

        def objective(trial: Trial) -> float:
            kf = StratifiedKFold if self.is_classifier else KFold
            _cv = kf(n_splits=n_folds, shuffle=True, random_state=SEED)
            opt_args = self.optuna_args(trial)
            full_args = {**self.fixed_args, **self.default_args, **opt_args}
            scores = []
            for step, (idx_train, idx_test) in enumerate(_cv.split(X_train, y_train)):
                X_tr, y_tr = X[idx_train], y[idx_train]
                X_test, y_test = X[idx_test], y[idx_test]
                model_cls, clean_args = self.model_cls_args(full_args)
                estimator = model_cls(**clean_args)
                estimator.fit(X_tr, y_tr)
                preds = estimator.predict(X_test)
                scorer = acc if self.is_classifier else mae
                score = scorer(preds, y_test)
                score = score if self.is_classifier else -score
                scores.append(score)
                # allows pruning
                trial.report(float(np.mean(scores)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

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

        grid = self.grid
        study = create_study(
            direction="maximize",  # handled in objective
            sampler=GridSampler(grid) if grid is not None else TPESampler(),
            pruner=MedianPruner(n_warmup_steps=0, n_min_trials=5),
        )
        optuna.logging.set_verbosity(verbosity)
        objective = self.optuna_objective(X_train=X_train, y_train=y_train)
        cbs = [EarlyStopping(patience=15, min_trials=50)]
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=cbs if grid is None else [],
            timeout=64_800,  # 18h * 60min/h * 60 sec/min, leave 6 hr on 24-hour job
            n_jobs=n_jobs,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        self.tuned_args = study.best_params
        self.refit_tuned(X=X_train, y=y_train, tuned_args=self.tuned_args)

        return study

    def fit(self, X_train: DataFrame, y_train: Series) -> None:
        if self.model is None:
            kwargs = {**self.fixed_args, **self.default_args, **self.model_args}
            self.model = self.model_cls_args(kwargs)[0](**kwargs)
        self.model.fit(X_train, y_train)

    def refit_tuned(self, X: DataFrame, y: Series, tuned_args: Optional[Mapping] = None) -> None:
        tuned_args = tuned_args or {}
        kwargs = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **tuned_args,
        }
        self.tuned_model = self.model_cls_args(kwargs)[0](**kwargs)

        if self.needs_calibration:
            self.tuned_model = CVCalibrate(self.tuned_model, method="sigmoid", cv=5, n_jobs=5)

        self.tuned_model.fit(X, y)

    def htune_eval(
        self,
        X_train: DataFrame,
        y_train: Series,
        X_test: DataFrame,
        y_test: Series,
    ) -> DataFrame:
        """
        Returns
        -------
        df: DataFrame
            DataFrame with columns: [trainset, holdout, 5-fold] and index as
            the scorers, values as the scorer metric values.
        """
        # TODO: need to specify valiation method, and return confidences, etc.
        # Actually maybe just want to call refit in here...
        if self.tuned_model is None:
            raise RuntimeError("Cannot evaluate tuning because model has not been tuned.")
        preds_test = self.tuned_predict(X_test)
        preds_train = self.tuned_predict(X_train)
        if self.is_classifier:
            probs_test = self.predict_proba(X_test)
            probs_train = self.predict_proba(X_train)
            scorer = ClassifierScorer
            holdout_scores = scorer.get_scores(y_true=y_test, y_pred=preds_test, y_prob=probs_test)
            train_scores = scorer.get_scores(y_true=y_train, y_pred=preds_train, y_prob=probs_train)
        else:
            scorer = RegressorScorer
            holdout_scores = scorer.get_scores(y_true=y_test, y_pred=preds_test)
            train_scores = scorer.get_scores(y_true=y_train, y_pred=preds_train)

        if self.is_classifier:
            ss = StratifiedKFold(n_splits=5)
        else:
            ss = KFold(n_splits=5)

        scores = []
        for idx_train, idx_test in ss.split(y_test, y_test):
            df_train = X_test.loc[idx_train]
            df_test = X_test.loc[idx_test]
            targ_train = y_test.loc[idx_train]
            targ_test = y_test.loc[idx_test]
            self.refit_tuned(X=df_train, y=targ_train, tuned_args=self.tuned_args)
            preds_test = self.tuned_predict(X=df_test)
            if self.is_classifier:
                probs_test = self.predict_proba(X=df_test)
                scorer = ClassifierScorer
                scores.append(
                    scorer.get_scores(y_true=targ_test, y_pred=preds_test, y_prob=probs_test)
                )
            else:
                scorer = RegressorScorer
                scores.append(scorer.get_scores(y_true=targ_test, y_pred=preds_test))

        holdout = Series(holdout_scores, name="holdout")
        train = Series(train_scores, name="trainset")
        scores = pd.concat([Series(score) for score in scores], axis=1)
        means = scores.mean(axis=1)
        means.name = "5-fold"
        df = pd.concat([train, holdout, means], axis=1)
        df.index.name = "metric"
        df = df.reset_index()
        return df

    def score(self, X: DataFrame, y: Series) -> float:
        if self.model is None:
            raise RuntimeError("Need to call `model.fit()` before calling `.score()`")
        return self.model.score(X, y)

    def tuned_score(self, X: DataFrame, y: Series) -> float:
        if self.tuned_model is None:
            raise RuntimeError("Need to tune model before calling `.tuned_score()`")
        return self.tuned_model.score(X, y)

    def tuned_scores(self, X: DataFrame, y: Series) -> Series:
        if self.tuned_model is None:
            raise RuntimeError("Need to tune model before calling `.tuned_scores()`")
        preds = self.tuned_predict(X)

        return self.tuned_model.score(X, y)

    def tuned_cv_scores(self, X: DataFrame, y: Series) -> Series:
        if self.tuned_model is None:
            raise RuntimeError("Need to tune model before calling `.tuned_scores()`")
        preds = self.tuned_predict(X)

        return self.tuned_model.score(X, y)

    def predict(self, X: DataFrame) -> Series:
        if self.model is None:
            raise RuntimeError("Need to call `model.fit()` before calling `.predict()`")
        return self.model.predict(X)

    def tuned_predict(self, X: DataFrame) -> Series:
        if self.tuned_model is None:
            raise RuntimeError("Need to call `model.tune()` before calling `.tuned_predict()`")
        return self.tuned_model.predict(X)

    def wrapper_select(
        self,
        X_train: DataFrame,
        y_train: Series,
        n_feat: Optional[Union[float, int]] = None,
        method: WrapperSelection = WrapperSelection.StepUp,
    ) -> Optional[tuple[list[str], dict[str, float]]]:
        n_select = get_n_select(X_train, n_feat=n_feat)
        if n_select is None:
            return None

        raise NotImplementedError()

    def predict_proba(self, X: DataFrame) -> ndarray:
        if not self.is_classifier:
            raise ValueError("Cannot get probabilities for a regression model.")

        if self.tuned_args is None or self.tuned_model is None:
            raise RuntimeError("Need to tune estimator before calling `.predict_proba()`")

        return self.tuned_model.predict_proba(X)


def get_n_select(X_train: DataFrame, n_feat: Optional[Union[float, int]] = None) -> Optional[int]:
    n_features = X_train.shape[1]
    msg = (
        "`n_feat` must be either None, an integer in [1, n_features - 1] "
        f"(i.e. an integer in [1, {n_features - 1}] for the current data) "
        "representing the number of features, or a float in (0, 1] "
        f"representing a percentage of features to select. Got: {n_feat}"
    )
    if n_feat is None:
        n_select = min(20, n_features - 1)
    elif isinstance(n_feat, Integral):
        if not 0 < n_feat < n_features:
            raise ValueError(msg)
        n_select = n_feat
    elif isinstance(n_feat, Real):
        if not 0 < n_feat <= 1:
            raise ValueError(msg)
        n_select = ceil(n_features * n_feat)
    else:
        raise ValueError(msg)

    if n_select >= n_features:  # from ceil or None case
        return None
