from __future__ import annotations

# fmt: off
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


class XGBoostEstimator(DfAnalyzeModel):
    shortname = "xgboost"
    longname = "XGBoost Estimator"
    timeout_s = 3600

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.fixed_args = dict(
            verbosity=0,
            random_state=SEED,
            n_jobs=-1,
        )
        self.default_args = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
        )

    def _xgboost_cls(self) -> type:
        try:
            from xgboost import XGBClassifier as XGBC
            from xgboost import XGBRegressor as XGBR
        except ImportError as e:
            raise ImportError(
                "XGBoost support requires the optional `xgboost` package. "
                "Install project dependencies before selecting model `xgboost`."
            ) from e
        return XGBC if self.is_classifier else XGBR

    def set_device_info(self, device_info: DeviceInfo) -> None:
        super().set_device_info(device_info)
        self.fixed_args.pop("device", None)
        self.fixed_args.pop("tree_method", None)
        self.fixed_args["n_jobs"] = -1
        if device_info.cuda_backends_enabled:
            if gpu_backends.xgboost_cuda_available():
                self.fixed_args["n_jobs"] = 1
                self.fixed_args.update(gpu_backends.xgboost_cuda_args())
                return
            gpu_backends.warn_unavailable_gpu_backend(
                "XGBoost",
                "XGBoost GPU requires a working device='cuda' backend.",
            )

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        device = str(self.fixed_args.get("device", ""))
        return (
            "xgboost-cuda"
            if device.startswith("cuda")
            else "xgboost-cpu"
        )

    def uses_accelerated_backend(
        self, tuned_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        return str(self.fixed_args.get("device", "")).startswith("cuda")

    def cuda_tuning_trial_budget(self) -> Optional[int]:
        if str(self.fixed_args.get("device", "")).startswith("cuda"):
            return 75
        return None

    def target_score_stopping_min_trials(self) -> int:
        if str(self.fixed_args.get("device", "")).startswith("cuda"):
            return 5
        return 1

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self._xgboost_cls(), dict(full_args)

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float(
                "min_child_weight", 1e-3, 10.0, log=True
            ),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
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
        if str(self.fixed_args.get("device", "")).startswith("cuda"):
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


class XGBoostClassifier(XGBoostEstimator):
    shortname = "xgboost"
    longname = "XGBoost Classifier"
    timeout_s = 3600

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True


class XGBoostRegressor(XGBoostEstimator):
    shortname = "xgboost"
    longname = "XGBoost Regressor"
    timeout_s = 3600

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
