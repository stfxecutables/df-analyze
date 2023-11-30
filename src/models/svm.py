from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Mapping, Optional

from optuna import Trial

from pandas import DataFrame, Series  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from sklearn.svm import SVC, SVR

from src.models.base import DfAnalyzeModel


class SVMEstimator(DfAnalyzeModel):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.needs_calibration = True
        self.fixed_args = dict(cache_size=1000)

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            # kernel=trial.suggest_categorical("kernel", choices=["rbf", "linear"]),
            kernel=trial.suggest_categorical("kernel", choices=["rbf"]),
            # combination of large C and small gamma can multiply training time
            # by orders of magnitude...
            C=trial.suggest_float("C", 1e-10, 1e3, log=True),
            gamma=trial.suggest_float("gamma", 1e-7, 1e5, log=True),
        )


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
