from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Callable, Mapping

from optuna import Trial

from pandas import DataFrame  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate as cv
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from src._constants import SEED
from src.models.base import NEG_MAE, DfAnalyzeModel


class KNNEstimator(DfAnalyzeModel):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.fixed_args = dict()

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 50, 1),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical("metric", ["cosine", "l1", "l2", "corr"]),
        )

    def optuna_objective(
        self, X_train: DataFrame, y_train: DataFrame, n_folds: int = 3
    ) -> Callable[[Trial], float]:
        # precompute to save huge amounts of compute at cost of potentially GBs
        # of memory
        distances = {
            "cosine": cosine_distances(X_train),
            "l1": manhattan_distances(X_train),
            "l2": euclidean_distances(X_train),
            "corr": np.abs(np.corrcoef(X_train)),
        }

        def objective(trial: Trial) -> float:
            kf = StratifiedKFold if self.is_classifier else KFold
            _cv = kf(n_splits=n_folds, shuffle=True, random_state=SEED)
            trial_args = self.optuna_args(trial)
            metric = str(trial_args.pop("metric"))
            # NOTE: we force precomputed for repeated trials
            estimator = self.model_cls(**self.fixed_args, **trial_args, metric="precomputed")
            scoring = "accuracy" if self.is_classifier else NEG_MAE
            scores = cv(
                estimator,  # type: ignore
                X=distances[metric],
                y=y_train,
                scoring=scoring,
                cv=_cv,
                n_jobs=1,
            )
            return float(np.mean(scores["test_score"]))

        return objective


class KNNClassifier(KNNEstimator):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = KNeighborsClassifier


class KNNRegressor(KNNEstimator):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = KNeighborsRegressor
