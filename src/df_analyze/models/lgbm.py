from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Mapping, Optional

import optuna
from optuna import Study, Trial

ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import Any, Type

from lightgbm import LGBMClassifier, LGBMRegressor
from pandas import DataFrame, Series

from df_analyze.device import DeviceInfo
from df_analyze.enumerables import Scorer
from df_analyze.models.base import DfAnalyzeModel


def _set_lightgbm_device_type(
    fixed_args: dict[str, Any], _device_info: DeviceInfo
) -> None:
    # LightGBM's CUDA RF backend can hang or run pathologically slowly on some
    # workloads, so df-analyze keeps both lgbm and rf on all-core CPU.
    # Users who still want to benchmark CUDA LightGBM should install it
    # manually and opt in outside df-analyze's automatic setup; see
    # docs/gpu_setup.md for the manual install command.
    fixed_args.pop("device_type", None)
    fixed_args["n_jobs"] = -1


def _lightgbm_backend_name(fixed_args: Mapping[str, Any]) -> str:
    device_type = fixed_args.get("device_type")
    if device_type == "cuda":
        return "lightgbm-cuda"
    return "lightgbm-cpu"


def _lightgbm_uses_accelerated_backend(fixed_args: Mapping[str, Any]) -> bool:
    return fixed_args.get("device_type") == "cuda"


class LightGBMEstimator(DfAnalyzeModel):
    shortname = "lgbm"
    longname = "LightGBM Estimator"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.fixed_args = dict(verbosity=-1, n_jobs=-1)
        self.model_cls: Type[Any] = type(None)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        clean_args = dict(full_args)
        clean_args.pop("device_type", None)
        clean_args["n_jobs"] = -1
        return self.model_cls, clean_args

    def set_device_info(self, device_info: DeviceInfo) -> None:
        super().set_device_info(device_info)
        _set_lightgbm_device_type(self.fixed_args, device_info)

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        return _lightgbm_backend_name(self.fixed_args)

    def uses_accelerated_backend(
        self, tuned_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        return _lightgbm_uses_accelerated_backend(self.fixed_args)

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
        # LightGBM itself uses all CPU cores via n_jobs=-1. Keep Optuna serial
        # so each fold fit owns the CPU instead of oversubscribing threads.
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

    # https://neptune.ai/blog/lightgbm-parameters-guide
    # https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            num_leaves=trial.suggest_int("num_leaves", 2, 256, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 0, 7),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        )


class LightGBMRFEstimator(DfAnalyzeModel):
    shortname = "rf"
    longname = "LightGBM Random Forest Estimator"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        # https://github.com/microsoft/LightGBM/issues/1333
        self.fixed_args = dict(verbosity=-1, n_jobs=-1)
        self.default_args = dict(bagging_freq=1, bagging_fraction=0.75)
        self.model_cls: Type[Any] = type(None)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        clean_args = dict(full_args)
        clean_args.pop("device_type", None)
        clean_args["n_jobs"] = -1
        return self.model_cls, clean_args

    def set_device_info(self, device_info: DeviceInfo) -> None:
        super().set_device_info(device_info)
        _set_lightgbm_device_type(self.fixed_args, device_info)

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        return _lightgbm_backend_name(self.fixed_args)

    def uses_accelerated_backend(
        self, tuned_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        return _lightgbm_uses_accelerated_backend(self.fixed_args)

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
        # LightGBM itself uses all CPU cores via n_jobs=-1. Keep Optuna serial
        # so each fold fit owns the CPU instead of oversubscribing threads.
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

    # https://neptune.ai/blog/lightgbm-parameters-guide
    # https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            num_leaves=trial.suggest_int("num_leaves", 2, 256, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 0, 7),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.05, 0.95),
        )


class LightGBMClassifier(LightGBMEstimator):
    shortname = "lgbm"
    longname = "LightGBM Classifier"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.fixed_args = dict(verbosity=-1, n_jobs=-1)
        self.model_cls = LGBMClassifier


class LightGBMRegressor(LightGBMEstimator):
    shortname = "lgbm"
    longname = "LightGBM Regressor"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.fixed_args = dict(verbosity=-1, n_jobs=-1)
        self.model_cls = LGBMRegressor


class LightGBMRFClassifier(LightGBMRFEstimator):
    shortname = "rf"
    longname = "LightGBM Random Forest Classifier"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.fixed_args.update(dict(boosting_type="rf", verbosity=-1))
        self.model_cls = LGBMClassifier


class LightGBMRFRegressor(LightGBMRFEstimator):
    shortname = "rf"
    longname = "LightGBM Random Forest Regressor"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.fixed_args.update(dict(boosting_type="rf", verbosity=-1))
        self.model_cls = LGBMRegressor
