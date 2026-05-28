from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Any, Mapping, Optional

import optuna
from numpy import ndarray
from optuna import Study, Trial
from pandas import DataFrame, Series

ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from df_analyze.enumerables import Scorer
from df_analyze.models import gpu as gpu_backends
from df_analyze.models.base import DfAnalyzeModel
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html

_CUML_SGD_CLASSIFIER_LOSSES = {
    "hinge": "hinge",
    "log": "log",
    "log_loss": "log",
}
_CUML_SGD_REGRESSOR_LOSSES = {
    "squared_error": "squared_loss",
    "squared_loss": "squared_loss",
}
_CPU_SGD_REGRESSOR_LOSSES = ["squared_error", "huber", "epsilon_insensitive"]
_CPU_SGD_SELECTOR_REGRESSOR_LOSSES = [
    "squared_error",
    "huber",
    "epsilon_insensitive",
    "squared_epsilon_insensitive",
]
_CUML_SGD_DROPPED_ARGS = {
    "average",
    "early_stopping",
    "validation_fraction",
}
_CUML_SGD_ETA0_LOW = 1e-4
_CUML_SGD_ETA0_HIGH = 1e-1
_CUML_SGD_ALPHA_LOW = 1e-6
_CUML_SGD_ALPHA_HIGH = 1e-2


class ElasticNetRegressor(DfAnalyzeModel):
    shortname = "elastic"
    longname = "ElasticNet Regressor"
    timeout_s = 30 * 60  # should never hit this

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = ElasticNet
        self.fixed_args = dict(max_iter=2000)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        if not self.device_info.cuda_backends_enabled:
            return self.model_cls, dict(full_args)

        cuml_cls = gpu_backends.get_cuml_elasticnet_cls()
        if cuml_cls is None:
            gpu_backends.warn_missing_cuml_backend(self.longname)
            return self.model_cls, dict(full_args)
        if not gpu_backends.cuml_elasticnet_available():
            gpu_backends.warn_unavailable_gpu_backend(
                self.longname,
                "The cuML ElasticNet smoke test failed in this environment.",
            )
            return self.model_cls, dict(full_args)

        args = gpu_backends.with_numpy_output(cuml_cls, dict(full_args))
        return cuml_cls, gpu_backends.filter_model_args(cuml_cls, args)

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            alpha=trial.suggest_float("alpha", 0.01, 1.0),
            l1_ratio=trial.suggest_float("l1_ratio", 0.01, 1.0),
        )

    def _uses_cuml_backend(self) -> bool:
        return (
            self.device_info.cuda_backends_enabled
            and gpu_backends.get_cuml_elasticnet_cls() is not None
            and gpu_backends.cuml_elasticnet_available()
        )

    def uses_accelerated_backend(
        self, tuned_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        return self._uses_cuml_backend()

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        return "cuml-elasticnet" if self._uses_cuml_backend() else "sklearn-elasticnet"

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
        if self._uses_cuml_backend():
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

    def predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().predict(X))

    def tuned_predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().tuned_predict(X))


class LRClassifier(DfAnalyzeModel):
    shortname = "lr"
    longname = "Logistic Regression"
    timeout_s = 15 * 60  # if it isn't done will be bad fit at this point

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = LogisticRegression
        self.fixed_args = dict(max_iter=2000, penalty="elasticnet", solver="saga")
        self.default_args = dict(l1_ratio=0.5)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        if not self.device_info.cuda_backends_enabled:
            return self.model_cls, dict(full_args)

        cuml_cls = gpu_backends.get_cuml_logistic_regression_cls()
        if cuml_cls is None:
            gpu_backends.warn_missing_cuml_backend(self.longname)
            return self.model_cls, dict(full_args)
        if not gpu_backends.cuml_logistic_regression_available():
            gpu_backends.warn_unavailable_gpu_backend(
                self.longname,
                "The cuML LogisticRegression smoke test failed in this environment.",
            )
            return self.model_cls, dict(full_args)

        args = gpu_backends.cuml_logistic_regression_args(cuml_cls, dict(full_args))
        return cuml_cls, args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        if self._uses_cuml_backend():
            return dict(
                C=trial.suggest_float("C", 1e-3, 10.0, log=True),
            )
        return dict(
            l1_ratio=trial.suggest_float("l1_ratio", 0.01, 1.0),
            C=trial.suggest_float("C", 1e-10, 1e2, log=True),
        )

    def _uses_cuml_backend(self) -> bool:
        return (
            self.device_info.cuda_backends_enabled
            and gpu_backends.get_cuml_logistic_regression_cls() is not None
            and gpu_backends.cuml_logistic_regression_available()
        )

    def uses_accelerated_backend(
        self, tuned_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        return self._uses_cuml_backend()

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        if self._uses_cuml_backend():
            return "cuml-logistic-regression"
        return "sklearn-logistic-regression"

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
        if self._uses_cuml_backend():
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

    def predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().predict(X))

    def tuned_predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().tuned_predict(X))

    def predict_proba(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().predict_proba(X))


class SGDClassifier(DfAnalyzeModel):
    shortname = "sgd"
    longname = "SGD Linear Classifer"
    timeout_s = 15 * 60  # can't have this eating up all the time

    def __init__(self, model_args: Mapping | None = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = SklearnSGDClassifier
        self.default_args = dict(learning_rate="adaptive", penalty="l2", eta0=3e-4)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        # cuML MBSGDClassifier does not document predict_proba(), while df-analyze
        # classifier evaluation can require probabilities. Keep this path sklearn.
        return self.model_cls, dict(full_args)

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            loss=trial.suggest_categorical("loss", ["log_loss", "modified_huber"]),
            eta0=trial.suggest_float("eta0", 1e-5, 5.0, log=True),
            early_stopping=trial.suggest_categorical("early_stopping", [True, False]),
        )

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        return "sklearn-sgd-classifier"


class SGDRegressor(DfAnalyzeModel):
    shortname = "sgd"
    longname = "SGD Linear Regressor"
    timeout_s = 15 * 60  # can't have this eating up all the time

    def __init__(self, model_args: Mapping | None = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = SklearnSGDRegressor
        self.default_args = dict(eta0=3e-4)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        if not self.device_info.cuda_backends_enabled:
            return self.model_cls, dict(full_args)

        cuml_cls = gpu_backends.get_cuml_mbsgd_regressor_cls()
        if cuml_cls is None:
            gpu_backends.warn_missing_cuml_backend(self.longname)
            return self.model_cls, dict(full_args)
        if not gpu_backends.cuml_mbsgd_regressor_available():
            gpu_backends.warn_unavailable_gpu_backend(
                self.longname,
                "The cuML MBSGDRegressor smoke test failed in this environment.",
            )
            return self.model_cls, dict(full_args)

        args = _cuml_sgd_regressor_args(full_args, cuml_cls)
        if args is None:
            return self.model_cls, dict(full_args)
        return cuml_cls, args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        if self._uses_cuml_backend():
            return dict(
                loss=trial.suggest_categorical("loss", ["squared_error"]),
                eta0=trial.suggest_float(
                    "eta0",
                    _CUML_SGD_ETA0_LOW,
                    _CUML_SGD_ETA0_HIGH,
                    log=True,
                ),
                penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            )
        return dict(
            loss=trial.suggest_categorical("loss", self._loss_choices()),
            eta0=trial.suggest_float("eta0", 1e-5, 5.0, log=True),
            penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            early_stopping=trial.suggest_categorical("early_stopping", [True, False]),
        )

    def _loss_choices(self) -> list[str]:
        if self._uses_cuml_backend():
            return ["squared_error"]
        return list(_CPU_SGD_REGRESSOR_LOSSES)

    def _uses_cuml_backend(
        self, full_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        if not self.device_info.cuda_backends_enabled:
            return False
        if not gpu_backends.cuml_mbsgd_regressor_available():
            return False
        cuml_cls = gpu_backends.get_cuml_mbsgd_regressor_cls()
        if cuml_cls is None:
            return False
        if full_args is None:
            return True
        return _cuml_sgd_regressor_args(dict(full_args), cuml_cls) is not None

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        full_args = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **(tuned_args or {}),
        }
        if self._uses_cuml_backend(full_args):
            return "cuml-mbsgd-regressor"
        return "sklearn-sgd-regressor"

    def uses_accelerated_backend(
        self, tuned_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        return self._uses_cuml_backend(tuned_args)

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
        if self._uses_cuml_backend():
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

    def predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().predict(X))

    def tuned_predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().tuned_predict(X))


class SGDClassifierSelector(DfAnalyzeModel):
    shortname = "sgd-select"
    longname = "SGD Linear Selector"
    timeout_s = 10 * 60  # can't have this eating up all the time

    def __init__(self, model_args: Mapping | None = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = SklearnSGDClassifier
        self.default_args = dict(learning_rate="adaptive", penalty="l1", eta0=3e-4)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        if not self.device_info.cuda_backends_enabled:
            return self.model_cls, dict(full_args)

        cuml_cls = gpu_backends.get_cuml_mbsgd_classifier_cls()
        if cuml_cls is None:
            gpu_backends.warn_missing_cuml_backend(self.longname)
            return self.model_cls, dict(full_args)
        if not gpu_backends.cuml_mbsgd_classifier_available():
            gpu_backends.warn_unavailable_gpu_backend(
                self.longname,
                "The cuML MBSGDClassifier smoke test failed in this environment.",
            )
            return self.model_cls, dict(full_args)

        args = _cuml_sgd_classifier_args(full_args, cuml_cls)
        if args is None:
            return self.model_cls, dict(full_args)
        return cuml_cls, args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        if self._uses_cuml_backend():
            return dict(
                loss=trial.suggest_categorical("loss", ["hinge", "log_loss"]),
                eta0=trial.suggest_float(
                    "eta0",
                    _CUML_SGD_ETA0_LOW,
                    _CUML_SGD_ETA0_HIGH,
                    log=True,
                ),
                alpha=trial.suggest_float(
                    "alpha",
                    _CUML_SGD_ALPHA_LOW,
                    _CUML_SGD_ALPHA_HIGH,
                    log=True,
                ),
            )
        args = dict(
            loss=trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
            eta0=trial.suggest_float("eta0", 1e-5, 5.0, log=True),
            alpha=trial.suggest_float("alpha", 1e-6, 1.0, log=True),
            early_stopping=trial.suggest_categorical("early_stopping", [True, False]),
            average=trial.suggest_int("average", 0, 20),
        )
        # prevent a FutureWarning
        if args["average"] == 0:
            args["average"] = False
        return args

    def _uses_cuml_backend(
        self, full_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        if not self.device_info.cuda_backends_enabled:
            return False
        if not gpu_backends.cuml_mbsgd_classifier_available():
            return False
        cuml_cls = gpu_backends.get_cuml_mbsgd_classifier_cls()
        if cuml_cls is None:
            return False
        if full_args is None:
            return True
        return _cuml_sgd_classifier_args(dict(full_args), cuml_cls) is not None

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        full_args = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **(tuned_args or {}),
        }
        if self._uses_cuml_backend(full_args):
            return "cuml-mbsgd-classifier"
        return "sklearn-sgd-classifier"

    def uses_accelerated_backend(
        self, tuned_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        return self._uses_cuml_backend(tuned_args)

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
        if self._uses_cuml_backend():
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

    def predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().predict(X))

    def tuned_predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().tuned_predict(X))


class SGDRegressorSelector(DfAnalyzeModel):
    shortname = "sgd-select"
    longname = "SGD Linear Selector"
    timeout_s = 10 * 60  # can't have this eating up all the time

    def __init__(self, model_args: Mapping | None = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = SklearnSGDRegressor
        self.default_args = dict(learning_rate="adaptive", penalty="l1", eta0=3e-4)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        if not self.device_info.cuda_backends_enabled:
            return self.model_cls, dict(full_args)

        cuml_cls = gpu_backends.get_cuml_mbsgd_regressor_cls()
        if cuml_cls is None:
            gpu_backends.warn_missing_cuml_backend(self.longname)
            return self.model_cls, dict(full_args)
        if not gpu_backends.cuml_mbsgd_regressor_available():
            gpu_backends.warn_unavailable_gpu_backend(
                self.longname,
                "The cuML MBSGDRegressor smoke test failed in this environment.",
            )
            return self.model_cls, dict(full_args)

        args = _cuml_sgd_regressor_args(full_args, cuml_cls)
        if args is None:
            return self.model_cls, dict(full_args)
        return cuml_cls, args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        if self._uses_cuml_backend():
            return dict(
                loss=trial.suggest_categorical("loss", ["squared_error"]),
                eta0=trial.suggest_float(
                    "eta0",
                    _CUML_SGD_ETA0_LOW,
                    _CUML_SGD_ETA0_HIGH,
                    log=True,
                ),
                alpha=trial.suggest_float(
                    "alpha",
                    _CUML_SGD_ALPHA_LOW,
                    _CUML_SGD_ALPHA_HIGH,
                    log=True,
                ),
            )
        args = dict(
            loss=trial.suggest_categorical("loss", self._loss_choices()),
            eta0=trial.suggest_float("eta0", 1e-5, 5.0, log=True),
            alpha=trial.suggest_float("alpha", 1e-6, 1.0, log=True),
            early_stopping=trial.suggest_categorical("early_stopping", [True, False]),
            average=trial.suggest_int("average", 0, 20),
        )
        # prevent a FutureWarning
        if args["average"] == 0:
            args["average"] = False
        return args

    def _loss_choices(self) -> list[str]:
        if self._uses_cuml_backend():
            return ["squared_error"]
        return list(_CPU_SGD_SELECTOR_REGRESSOR_LOSSES)

    def _uses_cuml_backend(
        self, full_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        if not self.device_info.cuda_backends_enabled:
            return False
        if not gpu_backends.cuml_mbsgd_regressor_available():
            return False
        cuml_cls = gpu_backends.get_cuml_mbsgd_regressor_cls()
        if cuml_cls is None:
            return False
        if full_args is None:
            return True
        return _cuml_sgd_regressor_args(dict(full_args), cuml_cls) is not None

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        full_args = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **(tuned_args or {}),
        }
        if self._uses_cuml_backend(full_args):
            return "cuml-mbsgd-regressor"
        return "sklearn-sgd-regressor"

    def uses_accelerated_backend(
        self, tuned_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        return self._uses_cuml_backend(tuned_args)

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
        if self._uses_cuml_backend():
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

    def predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().predict(X))

    def tuned_predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().tuned_predict(X))


def _cuml_sgd_common_args(args: dict[str, Any]) -> dict[str, Any]:
    # sklearn's SGD estimators accept early-stopping and averaging controls
    # that cuML's MBSGD estimators may not support. Keep only parameters that
    # map cleanly, then filter again against the installed cuML signature.
    clean = {k: v for k, v in args.items() if k not in _CUML_SGD_DROPPED_ARGS}
    if "max_iter" in clean and "epochs" not in clean:
        clean["epochs"] = clean.pop("max_iter")
    return clean


def _cuml_sgd_classifier_args(
    args: dict[str, Any], model_cls: type
) -> dict[str, Any] | None:
    clean = _cuml_sgd_common_args(args)
    loss = clean.get("loss")
    if isinstance(loss, str):
        if loss not in _CUML_SGD_CLASSIFIER_LOSSES:
            return None
        clean["loss"] = _CUML_SGD_CLASSIFIER_LOSSES[loss]
    clean = gpu_backends.with_numpy_output(model_cls, clean)
    return gpu_backends.filter_model_args(model_cls, clean)


def _cuml_sgd_regressor_args(
    args: dict[str, Any], model_cls: type
) -> dict[str, Any] | None:
    clean = _cuml_sgd_common_args(args)
    loss = clean.get("loss")
    if isinstance(loss, str):
        if loss not in _CUML_SGD_REGRESSOR_LOSSES:
            return None
        clean["loss"] = _CUML_SGD_REGRESSOR_LOSSES[loss]
    clean = gpu_backends.with_numpy_output(model_cls, clean)
    return gpu_backends.filter_model_args(model_cls, clean)
