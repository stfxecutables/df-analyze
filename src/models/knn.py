from __future__ import annotations


# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Any, Mapping, Type

import optuna
from optuna import Study, Trial
from pandas import DataFrame, Series

ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import Optional

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from src.models.base import DfAnalyzeModel


class KNNEstimator(DfAnalyzeModel):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.needs_calibration = False
        self.fixed_args = dict(n_jobs=1)
        self.model_cls: Type[Any] = type(None)
        self.shortname = "knn"
        self.longname = "K-Neighbours Estimator"
        self.grid = {
            "n_neighbors": [1, 5, 10, 25, 50],
            "weights": ["uniform", "distance"],
            "metric": ["cosine", "l2", "correlation"],
        }

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, {}

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 50, step=1),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical("metric", ["cosine", "l1", "l2", "correlation"]),
        )

    def htune_optuna(
        self,
        X_train: DataFrame,
        y_train: Series,
        n_trials: int = 100,
        n_jobs: int = -1,
        verbosity: int = optuna.logging.ERROR,
    ) -> Study:
        if self.grid is None:
            raise ValueError("Impossible!")
        n_hp = 1
        for opts in self.grid.values():
            n_hp *= len(opts)
        return super().htune_optuna(
            X_train, y_train, n_trials=n_hp, n_jobs=n_jobs, verbosity=verbosity
        )

    # def optuna_objective(
    #     self, X_train: DataFrame, y_train: DataFrame, n_folds: int = 3
    # ) -> Callable[[Trial], float]:
    #     # precompute to save huge amounts of compute at cost of potentially GBs
    #     # of memory
    #     distances = {
    #         "cosine": cosine_distances(X_train),
    #         "l1": manhattan_distances(X_train),
    #         "l2": euclidean_distances(X_train),
    #         "correlation": np.abs(np.corrcoef(X_train)),
    #     }
    #     y = np.asarray(y_train)

    #     def objective(trial: Trial) -> float:
    #         kf = StratifiedKFold if self.is_classifier else KFold
    #         _cv = kf(n_splits=n_folds, shuffle=True, random_state=SEED)
    #         opt_args = self.optuna_args(trial)
    #         metric = str(opt_args.pop("metric"))
    #         clean_args = {**self.fixed_args, **self.default_args, **opt_args}
    #         clean_args["metric"] = "precomputed"
    #         # NOTE: we force precomputed for repeated trials
    #         X = distances[metric]

    #         scores = []
    #         for step, (idx_train, idx_test) in enumerate(_cv.split(X_train, y_train)):
    #             X_tr, y_tr = X[idx_train][:, idx_train], y[idx_train]
    #             X_test, y_test = X[idx_test][:, idx_test], y[idx_test]
    #             estimator = self.model_cls(**clean_args)
    #             estimator.fit(X_tr, y_tr)
    #             score = estimator.score(np.copy(X_test), np.copy(y_test))
    #             score = score if self.is_classifier else -score
    #             scores.append(score)
    #             # allows pruning
    #             trial.report(float(np.mean(scores)), step=step)
    #             if trial.should_prune():
    #                 raise optuna.TrialPruned()
    #         return float(np.mean(scores))

    #         estimator = self.model_cls(**full_args)

    #         scoring = "accuracy" if self.is_classifier else NEG_MAE
    #         scores = cv(
    #             estimator,  # type: ignore
    #             X=distances[metric],
    #             y=y_train,
    #             scoring=scoring,
    #             cv=_cv,
    #             n_jobs=1,
    #         )
    #         return float(np.mean(scores["test_score"]))

    #     return objective


class KNNClassifier(KNNEstimator):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = KNeighborsClassifier
        self.shortname = "knn"
        self.longname = "K-Neighbours Classifier"


class KNNRegressor(KNNEstimator):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = KNeighborsRegressor
        self.shortname = "knn"
        self.longname = "K-Neighbours Regressor"
