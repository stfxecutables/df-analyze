from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Type, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import DataFrame, Series
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC, SVR

from src.hypertune import CLASSIFIER_TEST_SCORERS as CLS_SCORING
from src.hypertune import REGRESSION_TEST_SCORERS as REG_SCORING

AnyModel = Union[
    LGBMClassifier,
    LGBMRegressor,
    DummyClassifier,
    DummyRegressor,
    ElasticNetCV,
    LinearRegression,
    LogisticRegressionCV,
    SVC,
    SVR,
]


class UnivariatePredictor(ABC):
    """Mixin needed for unified interface"""

    def __init__(self) -> None:
        self.model: Type[AnyModel]
        self.grid: dict[str, list[Any]] = {}
        self.fixed_args: dict[str, Any] = {}
        self.is_classifier: bool = False
        self.__opt: Optional[GridSearchCV] = None

    def evaluate(self, encoded: DataFrame, target: str) -> DataFrame:
        opt = self.optimizer
        X = encoded.drop(columns="target")
        y = encoded[target]
        opt.fit(X, y)
        return self.get_best_scores(opt)

    @property
    def scorers(self) -> dict[str, Callable]:
        return CLS_SCORING if self.is_classifier else REG_SCORING

    @property
    def refit(self) -> str:
        return "acc" if self.is_classifier else "MAE"

    @property
    def optimizer(self) -> GridSearchCV:
        if self.__opt is None:
            self.__opt = GridSearchCV(
                estimator=self.model(**self.fixed_args),  # type: ignore
                param_grid=self.grid,
                scoring=self.scorers,
                refit=self.refit,
                cv=5,
                n_jobs=-1,
            )
        return self.__opt

    def get_best_scores(self, opt: GridSearchCV) -> Series:
        """Returns"""
        df = DataFrame(opt.cv_results_)
        filters = [f"mean_test_{scorer}" for scorer in self.scorers]
        df = df[filters].rename(columns=lambda s: s.replace("mean_test_", ""))
        if self.is_classifier:
            return df.iloc[np.argmax(df["acc"])]
        return df.iloc[np.argmin(df["MAE"])]


def logspace(start: int, stop: int) -> list[float]:
    n = stop - start + 1
    return np.logspace(start, stop, num=n, endpoint=True).tolist()


class ElasticNetRegressor(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = ElasticNetCV
        self.grid = {"l1_ratio": [[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]]}
        self.is_classifier = False


class SVMRegressor(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVR
        self.grid = {
            "kernel": ["rbf", "linear"],
            "gamma": logspace(-5, 5),
            "C": logspace(-10, 3),
        }
        self.is_classifier = False


class SVMClassifier(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVC
        self.grid = {
            "kernel": ["rbf", "linear"],
            "gamma": logspace(-5, 5),
            "C": logspace(-10, 3),
        }
        self.fixed_args = dict(probability=True)
        self.is_classifier = True


if __name__ == "__main__":
    X = np.random.uniform(0, 1, [200, 10])
    y = np.random.standard_exponential([200])
    y_cls = KBinsDiscretizer(n_bins=5, encode="ordinal").fit_transform(y.reshape(-1, 1)).ravel()
    target = "target"
    y = Series(y, name=target)
    y_cls = Series(y_cls, name=target)
    df = pd.concat([DataFrame(X), y], axis=1)
    df_cls = pd.concat([DataFrame(X), y_cls], axis=1)

    # model = SVMRegressor()
    # model.evaluate(df, target)

    model = SVMClassifier()
    model.evaluate(df_cls, target)
