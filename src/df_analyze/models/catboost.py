from __future__ import annotations

# fmt: off
import os
import sys  # isort: skip
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import optuna
from numpy import ndarray
from optuna import Study, Trial
from pandas import DataFrame, Series

ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from df_analyze._constants import SEED
from df_analyze.device import DeviceInfo
from df_analyze.enumerables import Scorer
from df_analyze.models import gpu as gpu_backends
from df_analyze.models.base import DfAnalyzeModel


class CatBoostEstimator(DfAnalyzeModel):
    shortname = "catboost"
    longname = "CatBoost Estimator"
    timeout_s = 3600

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.fixed_args = dict(
            verbose=False,
            allow_writing_files=False,
            random_seed=SEED,
        )
        self.default_args = dict(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
        )

    def _catboost_cls(self) -> type:
        try:
            from catboost import CatBoostClassifier as CBC
            from catboost import CatBoostRegressor as CBR
        except ImportError as e:
            raise ImportError(
                "CatBoost support requires the optional `catboost` package. "
                "Install project dependencies before selecting model `catboost`."
            ) from e
        return CBC if self.is_classifier else CBR

    def set_device_info(self, device_info: DeviceInfo) -> None:
        super().set_device_info(device_info)
        if device_info.cuda_backends_enabled and gpu_backends.catboost_cuda_available():
            self.fixed_args["task_type"] = "GPU"
            self.fixed_args["devices"] = os.environ.get(
                "DF_ANALYZE_CATBOOST_DEVICES", "0"
            )
        else:
            self.fixed_args["task_type"] = "CPU"
            self.fixed_args.pop("devices", None)
            if device_info.cuda_backends_enabled:
                gpu_backends.warn_unavailable_gpu_backend(
                    "CatBoost",
                    "CatBoost GPU requires a working task_type='GPU' backend.",
                )

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        return (
            "catboost-gpu"
            if self.fixed_args.get("task_type") == "GPU"
            else "catboost-cpu"
        )

    def uses_accelerated_backend(
        self, tuned_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        return self.fixed_args.get("task_type") == "GPU"

    def cuda_tuning_trial_budget(self) -> Optional[int]:
        if self.fixed_args.get("task_type") == "GPU":
            return 75
        return None

    def target_score_stopping_min_trials(self) -> int:
        if self.fixed_args.get("task_type") == "GPU":
            return 5
        return 1

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self._catboost_cls(), dict(full_args)

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            iterations=trial.suggest_int("iterations", 100, 500, step=100),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
        )

    def htune_optuna(
        self,
        X_train: DataFrame,
        y_train: Series,
        g_train: Optional[Series],
        metric: Scorer,
        n_trials: int = 100,
        n_jobs: int = -1,
        verbosity: int = optuna.logging.ERROR,
    ) -> Study:
        if self.fixed_args.get("task_type") == "GPU":
            n_jobs = 1
        return super().htune_optuna(
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            metric=metric,
            n_trials=n_trials,
            n_jobs=n_jobs,
            verbosity=verbosity,
        )

    @staticmethod
    def _as_1d(value: Any) -> ndarray:
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.ravel()
        return arr

    def predict(self, X: DataFrame) -> ndarray:
        return self._as_1d(super().predict(X))

    def tuned_predict(self, X: DataFrame) -> ndarray:
        return self._as_1d(super().tuned_predict(X))

    def predict_proba(self, X: DataFrame) -> ndarray:
        return np.asarray(super().predict_proba(X))


class CatBoostClassifier(CatBoostEstimator):
    shortname = "catboost"
    longname = "CatBoost Classifier"
    timeout_s = 3600

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True


class CatBoostRegressor(CatBoostEstimator):
    shortname = "catboost"
    longname = "CatBoost Regressor"
    timeout_s = 3600

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
