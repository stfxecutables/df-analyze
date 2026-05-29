from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import platform
import warnings
from typing import Any, Mapping, Optional, Type

import optuna
from numpy import ndarray
from optuna import Study, Trial
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from df_analyze.device import DeviceInfo
from df_analyze.enumerables import Scorer
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.models import gpu as gpu_backends


_WARNED_MISSING_CUML = False
_CPU_KNN_METRICS = ["cosine", "l1", "l2", "correlation"]
_CPU_KNN_GRID_METRICS = ["cosine", "l2", "correlation"]


class KNNEstimator(DfAnalyzeModel):
    shortname = "knn"
    longname = "K-Neighbours Estimator"
    timeout_s = 30 * 60  # 30 minutes is enough given the grid

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        n_jobs = 1 if platform.system().lower() == "darwin" else -1
        self.is_classifier = False
        self.needs_calibration = False
        self.fixed_args = dict(n_jobs=n_jobs)
        self.model_cls: Type[Any] = type(None)
        self._cpu_grid = {
            "n_neighbors": [1, 5, 10, 25, 50],
            "weights": ["uniform", "distance"],
            "metric": list(_CPU_KNN_GRID_METRICS),
        }
        self._cuda_grid = {
            **self._cpu_grid,
            "metric": gpu_backends.cuml_knn_metric_choices(_CPU_KNN_GRID_METRICS),
        }
        self.grid = dict(self._cpu_grid)

    def set_device_info(self, device_info: DeviceInfo) -> None:
        super().set_device_info(device_info)
        self.grid = dict(self._cuda_grid if self._uses_cuml_backend() else self._cpu_grid)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        if not self.device_info.cuda_backends_enabled:
            return self.model_cls, dict(full_args)

        cuml_cls = gpu_backends.get_cuml_knn_cls(self.is_classifier)
        if cuml_cls is None:
            _warn_missing_cuml()
            return self.model_cls, dict(full_args)
        if not gpu_backends.cuml_knn_available(self.is_classifier):
            gpu_backends.warn_unavailable_gpu_backend(
                "KNN",
                "The cuML KNN smoke test failed in this environment.",
            )
            return self.model_cls, dict(full_args)

        if not self._cuml_metric_supported(full_args):
            metric = full_args.get("metric")
            gpu_backends.warn_unavailable_gpu_backend(
                "KNN",
                f"cuML KNN metric '{metric}' is not enabled in df-analyze.",
            )
            return self.model_cls, dict(full_args)

        return cuml_cls, self._cuml_args(full_args, cuml_cls)

    def _cuml_args(self, full_args: dict[str, Any], model_cls: type) -> dict[str, Any]:
        args = dict(full_args)
        args.pop("n_jobs", None)
        if isinstance(args.get("metric"), str):
            args["metric"] = gpu_backends.normalize_cuml_knn_metric(args["metric"])
        args = gpu_backends.with_numpy_output(model_cls, args)
        return gpu_backends.filter_model_args(model_cls, args)

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 50, step=1),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical("metric", self._metric_choices()),
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
        if self.grid is None:
            raise ValueError("Impossible!")
        self.grid = dict(self._cuda_grid if self._uses_cuml_backend() else self._cpu_grid)
        n_hp = 1
        for opts in self.grid.values():
            n_hp *= len(opts)
        if self._uses_cuml_backend():
            n_jobs = 1
        else:
            n_jobs = -1 if self.fixed_args["n_jobs"] == 1 else 1
        return super().htune_optuna(
            X_train,
            y_train,
            g_train,
            metric=metric,
            n_trials=n_hp,
            n_jobs=n_jobs,
            verbosity=verbosity,
        )

    def _metric_choices(self) -> list[str]:
        if not self._uses_cuml_backend():
            return list(_CPU_KNN_METRICS)
        return gpu_backends.cuml_knn_metric_choices(_CPU_KNN_METRICS)

    def _cuml_metric_supported(
        self, full_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        if full_args is None:
            return True
        metric = full_args.get("metric")
        if metric is None:
            return True
        return isinstance(metric, str) and gpu_backends.cuml_knn_metric_available(
            metric, is_classifier=self.is_classifier
        )

    def _uses_cuml_backend(
        self, full_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        if not self.device_info.cuda_backends_enabled:
            return False
        if gpu_backends.get_cuml_knn_cls(self.is_classifier) is None:
            return False
        if not gpu_backends.cuml_knn_available(self.is_classifier):
            return False
        return self._cuml_metric_supported(full_args)

    def uses_accelerated_backend(
        self, tuned_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        return self._uses_cuml_backend(tuned_args)

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        full_args = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **(tuned_args or {}),
        }
        return "cuml-knn" if self._uses_cuml_backend(full_args) else "sklearn-knn"

    def predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().predict(X))

    def tuned_predict(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().tuned_predict(X))

    def predict_proba(self, X: DataFrame) -> ndarray:
        return gpu_backends.as_numpy(super().predict_proba(X))


def _warn_missing_cuml() -> None:
    global _WARNED_MISSING_CUML
    if _WARNED_MISSING_CUML:
        return
    warnings.warn(
        "CUDA was selected for KNN, but optional RAPIDS cuML KNN is unavailable. "
        "Falling back to sklearn KNN.",
        RuntimeWarning,
        stacklevel=3,
    )
    _WARNED_MISSING_CUML = True


class KNNClassifier(KNNEstimator):
    shortname = "knn"
    longname = "K-Neighbours Classifier"
    timeout_s = 30 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = KNeighborsClassifier


class KNNRegressor(KNNEstimator):
    shortname = "knn"
    longname = "K-Neighbours Regressor"
    timeout_s = 30 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = KNeighborsRegressor
        self.shortname = "knn"
        self.longname = "K-Neighbours Regressor"
