from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Mapping, Optional

from optuna import Trial

ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from sklearn.linear_model import ElasticNet, LogisticRegression

from src.models.base import DfAnalyzeModel

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html


class ElasticNetRegressor(DfAnalyzeModel):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = ElasticNet
        self.fixed_args = dict(max_iter=2000)

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            alpha=trial.suggest_float("alpha", 0.01, 1.0),
            l1_ratio=trial.suggest_float("l1_ratio", 0.01, 1.0),
        )


class LRClassifier(DfAnalyzeModel):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = LogisticRegression
        self.fixed_args = dict(max_iter=2000, penalty="elasticnet", solver="saga")
        self.default_args = dict(l1_ratio=0.5)

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            l1_ratio=trial.suggest_float("l1_ratio", 0.01, 1.0),
            C=trial.suggest_float("C", 1e-10, 1e2, log=True),
        )
