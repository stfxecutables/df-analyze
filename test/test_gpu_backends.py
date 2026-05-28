from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from optuna import Trial
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

import df_analyze.models.knn as knn_module
from df_analyze.device import device_info_from_resolved
from df_analyze.enumerables import ClassifierScorer
from df_analyze.models import gpu as gpu_backends
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.models.knn import KNNClassifier
from df_analyze.models.lgbm import LightGBMClassifier, LightGBMRFClassifier
from df_analyze.models.linear import (
    ElasticNetRegressor,
    LRClassifier,
    SGDClassifier,
    SGDClassifierSelector,
    SGDRegressor,
    SGDRegressorSelector,
)
from df_analyze.models.svm import SVMClassifier, SVMRegressor
from df_analyze.models.xgboost import XGBoostClassifier


def tiny_tuning_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y = pd.Series([0, 1, 0, 1], name="target")
    return X, y


def report_model(
    report: dict[str, Any],
    *,
    model: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    matches = [
        item
        for item in report["models"]
        if (model is None or item["model"] == model)
        and (backend is None or item["backend"] == backend)
    ]
    assert len(matches) == 1
    return matches[0]


class RecordingTrial:
    def __init__(self) -> None:
        self.choices: dict[str, list[Any]] = {}
        self.floats: dict[str, tuple[float, float, dict[str, Any]]] = {}
        self.ints: dict[str, tuple[int, int, dict[str, Any]]] = {}

    def suggest_int(self, name: str, low: int, high: int, **kwargs: Any) -> int:
        self.ints[name] = (low, high, dict(kwargs))
        return low

    def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> float:
        self.floats[name] = (low, high, dict(kwargs))
        return low

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        self.choices[name] = list(choices)
        return choices[0]


@pytest.mark.fast
def test_base_model_uses_cleaned_fit_and_refit_args() -> None:
    class RecordingEstimator:
        init_kwargs: dict[str, Any] = {}

        def __init__(self, **kwargs: Any) -> None:
            type(self).init_kwargs = kwargs

        def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
            return None

    class CleaningModel(DfAnalyzeModel):
        def model_cls_args(
            self, full_args: dict[str, Any]
        ) -> tuple[type, dict[str, Any]]:
            clean_args = dict(full_args)
            clean_args.pop("bad", None)
            clean_args["cleaned"] = True
            return RecordingEstimator, clean_args

        def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
            return {}

    X, y = tiny_tuning_data()
    model = CleaningModel(model_args={"bad": "remove", "keep": 1})

    model.fit(X, y)
    assert RecordingEstimator.init_kwargs == {"keep": 1, "cleaned": True}

    model.refit_tuned(X, y, tuned_args={"bad": "remove", "tuned": 2})
    assert RecordingEstimator.init_kwargs == {
        "keep": 1,
        "tuned": 2,
        "cleaned": True,
    }


@pytest.mark.fast
def test_knn_cpu_preserves_tuned_args() -> None:
    model = KNNClassifier()
    args = {"n_neighbors": 7, "weights": "distance", "metric": "cosine", "n_jobs": -1}

    model_cls, clean_args = model.model_cls_args(args)

    assert model_cls is KNeighborsClassifier
    assert clean_args == args


@pytest.mark.fast
def test_knn_cuda_uses_cuml_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCumlKNN:
        def __init__(
            self,
            n_neighbors: int = 5,
            weights: str = "uniform",
            metric: str = "euclidean",
            output_type: str | None = None,
        ) -> None:
            self.args = {
                "n_neighbors": n_neighbors,
                "weights": weights,
                "metric": metric,
                "output_type": output_type,
            }

    monkeypatch.setattr(
        gpu_backends, "get_cuml_knn_cls", lambda is_classifier: FakeCumlKNN
    )
    monkeypatch.setattr(
        gpu_backends, "cuml_knn_available", lambda is_classifier=True: True
    )
    model = KNNClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model_cls, clean_args = model.model_cls_args(
        {"n_neighbors": 11, "weights": "distance", "metric": "l2", "n_jobs": -1}
    )

    assert model_cls is FakeCumlKNN
    assert clean_args == {
        "n_neighbors": 11,
        "weights": "distance",
        "metric": "euclidean",
        "output_type": "numpy",
    }
    assert model.device_info.resolved == "cuda"
    assert model.backend_name() == "cuml-knn"


@pytest.mark.fast
def test_knn_cuda_falls_back_when_cuml_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "get_cuml_knn_cls", lambda is_classifier: None)
    monkeypatch.setattr(knn_module, "_WARNED_MISSING_CUML", False)
    model = KNNClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))
    args = {"n_neighbors": 3, "weights": "uniform", "metric": "cosine", "n_jobs": -1}

    with pytest.warns(RuntimeWarning, match="cuML"):
        model_cls, clean_args = model.model_cls_args(args)

    assert model_cls is KNeighborsClassifier
    assert clean_args == args
    assert model.backend_name() == "sklearn-knn"


@pytest.mark.fast
def test_knn_cuda_tuning_forces_single_job(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCumlKNN:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    calls: list[int] = []

    def fake_htune_optuna(
        self: DfAnalyzeModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        g_train: pd.Series | None,
        metric: ClassifierScorer,
        n_trials: int = 100,
        n_jobs: int = -1,
        verbosity: int = 0,
    ) -> object:
        calls.append(n_jobs)
        return object()

    monkeypatch.setattr(
        gpu_backends, "get_cuml_knn_cls", lambda is_classifier: FakeCumlKNN
    )
    monkeypatch.setattr(
        gpu_backends, "cuml_knn_available", lambda is_classifier=True: True
    )
    monkeypatch.setattr(DfAnalyzeModel, "htune_optuna", fake_htune_optuna)
    X, y = tiny_tuning_data()
    model = KNNClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model.htune_optuna(
        X_train=X,
        y_train=y,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_jobs=-1,
    )

    assert calls == [1]


@pytest.mark.fast
def test_knn_cpu_metric_search_space_stays_full() -> None:
    trial = RecordingTrial()
    model = KNNClassifier()

    args = model.optuna_args(trial)  # type: ignore[arg-type]

    assert args["metric"] == "cosine"
    assert trial.choices["metric"] == ["cosine", "l1", "l2", "correlation"]


@pytest.mark.fast
def test_knn_cuda_metric_search_space_is_cuml_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCumlKNN:
        pass

    monkeypatch.setattr(
        gpu_backends, "get_cuml_knn_cls", lambda is_classifier: FakeCumlKNN
    )
    monkeypatch.setattr(
        gpu_backends, "cuml_knn_available", lambda is_classifier=True: True
    )
    trial = RecordingTrial()
    model = KNNClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    args = model.optuna_args(trial)  # type: ignore[arg-type]

    assert args["metric"] == "l2"
    assert trial.choices["metric"] == ["l2"]
    assert model.grid["metric"] == ["l2"]


@pytest.mark.fast
def test_knn_cuda_unsupported_metric_reports_sklearn_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCumlKNN:
        pass

    monkeypatch.setattr(
        gpu_backends, "get_cuml_knn_cls", lambda is_classifier: FakeCumlKNN
    )
    monkeypatch.setattr(
        gpu_backends, "cuml_knn_available", lambda is_classifier=True: True
    )
    gpu_backends._WARNED_UNAVAILABLE_GPU_BACKENDS.clear()
    model = KNNClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))
    args = {"n_neighbors": 3, "weights": "uniform", "metric": "cosine", "n_jobs": -1}

    with pytest.warns(RuntimeWarning, match="KNN"):
        model_cls, clean_args = model.model_cls_args(args)

    assert model_cls is KNeighborsClassifier
    assert clean_args == args
    assert model.backend_name({"metric": "cosine"}) == "sklearn-knn"
    assert model.backend_name({"metric": "l2"}) == "cuml-knn"


@pytest.mark.fast
def test_lgbm_cuda_request_stays_on_all_core_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_boosting_types: list[str] = []

    monkeypatch.setattr(
        gpu_backends,
        "lightgbm_best_device_type",
        lambda boosting_type="gbdt": checked_boosting_types.append(boosting_type)
        or ("cuda", ""),
    )
    model = LightGBMClassifier()

    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.fixed_args["n_jobs"] == -1
    assert "device_type" not in model.fixed_args
    assert model.backend_name() == "lightgbm-cpu"
    assert checked_boosting_types == []
    _, clean_args = model.model_cls_args({**model.fixed_args, "device_type": "cuda"})
    assert clean_args["n_jobs"] == -1
    assert "device_type" not in clean_args


@pytest.mark.fast
def test_lgbm_rf_cuda_request_stays_on_all_core_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_boosting_types: list[str] = []

    monkeypatch.setattr(
        gpu_backends,
        "lightgbm_best_device_type",
        lambda boosting_type="gbdt": checked_boosting_types.append(boosting_type)
        or ("cuda", ""),
    )
    model = LightGBMRFClassifier()

    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.fixed_args["boosting_type"] == "rf"
    assert model.fixed_args["n_jobs"] == -1
    assert "device_type" not in model.fixed_args
    assert model.backend_name() == "lightgbm-cpu"
    assert checked_boosting_types == []
    _, clean_args = model.model_cls_args({**model.fixed_args, "device_type": "cuda"})
    assert clean_args["n_jobs"] == -1
    assert "device_type" not in clean_args


@pytest.mark.fast
def test_as_numpy_converts_common_gpu_like_outputs() -> None:
    class FakeGet:
        def __init__(self, value: Any) -> None:
            self.value = value

        def get(self) -> Any:
            return self.value

    class FakeToNumpy:
        def to_numpy(self) -> np.ndarray:
            return np.array([3, 4])

    class FakeValuesGet:
        values = FakeGet([5, 6])

    assert np.array_equal(gpu_backends.as_numpy(np.array([1, 2])), np.array([1, 2]))
    assert np.array_equal(gpu_backends.as_numpy(FakeGet([1, 2])), np.array([1, 2]))
    assert np.array_equal(gpu_backends.as_numpy(FakeToNumpy()), np.array([3, 4]))
    assert np.array_equal(gpu_backends.as_numpy(FakeValuesGet()), np.array([5, 6]))


@pytest.mark.fast
def test_cuml_capability_probe_requires_successful_fit_and_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWorkingKNN:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def fit(self, X: np.ndarray, y: np.ndarray) -> FakeWorkingKNN:
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.8, 0.2]]), (len(X), 1))

    class FakeBrokenKNN(FakeWorkingKNN):
        def fit(self, X: np.ndarray, y: np.ndarray) -> FakeBrokenKNN:
            raise RuntimeError("no CUDA runtime")

    gpu_backends.cuml_knn_available.cache_clear()
    monkeypatch.setattr(
        gpu_backends, "get_cuml_knn_cls", lambda is_classifier: FakeWorkingKNN
    )
    assert gpu_backends.cuml_knn_available(is_classifier=True) is True

    gpu_backends.cuml_knn_available.cache_clear()
    monkeypatch.setattr(
        gpu_backends, "get_cuml_knn_cls", lambda is_classifier: FakeBrokenKNN
    )
    assert gpu_backends.cuml_knn_available(is_classifier=True) is False


@pytest.mark.fast
def test_cuml_logistic_probe_uses_df_analyze_cuda_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_args: dict[str, Any] = {}

    class FakeCumlLogisticRegression:
        def __init__(
            self,
            max_iter: int = 1000,
            penalty: str = "l2",
            solver: str | None = None,
            tol: float = 1e-4,
            C: float = 1.0,
            l1_ratio: float | None = None,
            output_type: str | None = None,
        ) -> None:
            seen_args.update(
                {
                    "max_iter": max_iter,
                        "penalty": penalty,
                        "solver": solver,
                        "tol": tol,
                        "C": C,
                        "l1_ratio": l1_ratio,
                        "output_type": output_type,
                }
            )

        def fit(
            self, X: np.ndarray, y: np.ndarray
        ) -> FakeCumlLogisticRegression:
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.8, 0.2]]), (len(X), 1))

    monkeypatch.setattr(
        gpu_backends,
        "get_cuml_logistic_regression_cls",
        lambda: FakeCumlLogisticRegression,
    )

    available, reason = gpu_backends.cuml_logistic_regression_availability()

    assert available is True
    assert reason == ""
    assert seen_args == {
        "max_iter": 20,
        "penalty": "l2",
        "solver": "qn",
        "tol": 1e-2,
        "C": 1.0,
        "l1_ratio": None,
        "output_type": "numpy",
    }


@pytest.mark.fast
def test_backend_availability_report_skips_cuda_smoke_checks_for_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "cuda_available", lambda: False)
    monkeypatch.setattr(gpu_backends, "cuda_driver_available", lambda: False)
    monkeypatch.setattr(gpu_backends, "mps_available", lambda: False)
    monkeypatch.setattr(
        gpu_backends, "tabpfn_api_status", lambda: (False, "tabpfn missing")
    )
    monkeypatch.setattr(
        gpu_backends, "tabpfn_v25_api_status", lambda: (False, "tabpfn missing")
    )
    report = gpu_backends.backend_availability_report(
        device_info_from_resolved(
            "cpu",
            "cpu",
            torch_cuda_available=False,
            cuda_driver_available=False,
            mps_is_available=False,
        )
    )

    assert report["device"] == {
        "requested": "cpu",
        "resolved": "cpu",
        "torch_device": "cpu",
        "cuda_ready": False,
        "torch_cuda_ready": False,
        "nvidia_driver_ready": False,
        "mps_ready": False,
    }
    assert report["checks"]["cuda_backend_checks"] is False
    assert "cuml_knn_available" not in report
    assert "xgboost_cuda_available" not in report
    tabpfn = report_model(report, model="tabpfn")
    assert tabpfn["checked"] is True
    assert tabpfn["available"] is False
    assert tabpfn["reason"] == "tabpfn missing; tabpfn missing"


@pytest.mark.fast
def test_backend_availability_report_records_cuda_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "cuda_available", lambda: True)
    monkeypatch.setattr(gpu_backends, "cuda_driver_available", lambda: True)
    monkeypatch.setattr(gpu_backends, "mps_available", lambda: False)
    monkeypatch.setattr(gpu_backends, "tabpfn_api_status", lambda: (True, ""))
    monkeypatch.setattr(gpu_backends, "tabpfn_v25_api_status", lambda: (True, ""))
    monkeypatch.setattr(
        gpu_backends,
        "cuml_knn_availability",
        lambda is_classifier=True: (
            is_classifier,
            "" if is_classifier else "regressor smoke failed",
        ),
    )
    monkeypatch.setattr(
        gpu_backends,
        "cuml_logistic_regression_availability",
        lambda: (False, "lr import failed"),
    )
    monkeypatch.setattr(
        gpu_backends, "cuml_elasticnet_availability", lambda: (True, "")
    )
    monkeypatch.setattr(
        gpu_backends, "cuml_mbsgd_classifier_availability", lambda: (True, "")
    )
    monkeypatch.setattr(
        gpu_backends,
        "cuml_mbsgd_regressor_availability",
        lambda: (False, "mbsgd fit failed"),
    )
    monkeypatch.setattr(
        gpu_backends,
        "lightgbm_cuda_availability",
        lambda: (False, "lightgbm cuda unavailable"),
    )
    monkeypatch.setattr(
        gpu_backends,
        "lightgbm_cuda_rf_availability",
        lambda: (False, "lightgbm rf cuda unavailable"),
    )
    monkeypatch.setattr(
        gpu_backends,
        "lightgbm_best_device_type",
        lambda boosting_type="gbdt": (
            None,
            "lightgbm rf cuda unavailable"
            if boosting_type == "rf"
            else "lightgbm cuda unavailable",
        ),
    )
    monkeypatch.setattr(
        gpu_backends, "catboost_cuda_availability", lambda: (True, "")
    )
    monkeypatch.setattr(
        gpu_backends,
        "xgboost_cuda_availability",
        lambda: (False, "xgboost cuda unavailable"),
    )

    report = gpu_backends.backend_availability_report(
        device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True)
    )

    assert report["device"]["torch_cuda_ready"] is True
    assert report["device"]["nvidia_driver_ready"] is True
    assert report["device"]["cuda_ready"] is True
    assert report["checks"]["cuda_backend_checks"] is True
    assert report_model(
        report, model="knn", backend="cuml-knn-classifier"
    )["available"] is True
    knn_reg = report_model(report, model="knn", backend="cuml-knn-regressor")
    assert knn_reg["available"] is False
    assert knn_reg["reason"] == "regressor smoke failed"
    assert report_model(
        report, model="lr", backend="cuml-logistic-regression"
    )["reason"] == "lr import failed"
    assert report_model(
        report, model="elastic", backend="cuml-elasticnet"
    )["available"] is True
    assert report_model(
        report, model="sgd", backend="cuml-mbsgd-regressor"
    )["reason"] == "mbsgd fit failed"
    assert report_model(
        report, model="sgd-selector", backend="cuml-mbsgd-regressor"
    )["reason"] == "mbsgd fit failed"
    assert report_model(
        report, model="tabpfn", backend="tabpfn-cuda"
    )["available"] is True
    assert report_model(
        report, model="catboost", backend="catboost-gpu"
    )["available"] is True
    assert report_model(
        report, model="xgboost", backend="xgboost-cuda"
    )["reason"] == "xgboost cuda unavailable"
    assert report_model(report, model="lgbm")["backend"] == "lightgbm-cpu"
    assert report_model(report, model="rf")["backend"] == "lightgbm-rf-cpu"


@pytest.mark.fast
def test_backend_availability_report_only_checks_selected_cuda_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_probe(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        raise AssertionError("unexpected backend probe")

    monkeypatch.setattr(gpu_backends, "cuda_available", lambda: True)
    monkeypatch.setattr(gpu_backends, "cuda_driver_available", lambda: True)
    monkeypatch.setattr(gpu_backends, "mps_available", lambda: False)
    monkeypatch.setattr(gpu_backends, "tabpfn_api_status", fail_probe)
    monkeypatch.setattr(gpu_backends, "tabpfn_v25_api_status", fail_probe)
    monkeypatch.setattr(gpu_backends, "cuml_knn_availability", fail_probe)
    monkeypatch.setattr(
        gpu_backends, "cuml_logistic_regression_availability", fail_probe
    )
    monkeypatch.setattr(gpu_backends, "cuml_elasticnet_availability", fail_probe)
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_classifier_availability", fail_probe)
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_regressor_availability", fail_probe)
    monkeypatch.setattr(gpu_backends, "lightgbm_cuda_availability", fail_probe)
    monkeypatch.setattr(gpu_backends, "lightgbm_cuda_rf_availability", fail_probe)
    monkeypatch.setattr(gpu_backends, "lightgbm_best_device_type", fail_probe)
    monkeypatch.setattr(gpu_backends, "catboost_cuda_availability", fail_probe)
    monkeypatch.setattr(
        gpu_backends, "xgboost_cuda_availability", lambda: (True, "")
    )

    report = gpu_backends.backend_availability_report(
        device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True),
        selected_models=["xgboost"],
    )

    assert report["checks"]["scope"] == "selected"
    assert report["selected_models"] == ["xgboost"]
    assert report["checks"]["cuda_backend_checks"] is True
    assert report["models"] == [
        {
            "model": "xgboost",
            "backend": "xgboost-cuda",
            "device": "cuda",
            "checked": True,
            "available": True,
            "fallback_backend": "xgboost-cpu",
        }
    ]


@pytest.mark.fast
def test_backend_availability_report_keeps_main_sgd_classifier_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_probe(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        raise AssertionError("unexpected backend probe")

    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_classifier_availability", fail_probe)
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_regressor_availability", fail_probe)

    report = gpu_backends.backend_availability_report(
        device_info_from_resolved(
            "auto",
            "cuda",
            torch_cuda_available=True,
            cuda_driver_available=True,
        ),
        selected_models=["sgd-classifier"],
    )

    assert report["checks"]["cuda_backend_checks"] is False
    assert report["models"] == [
        {
            "model": "sgd",
            "backend": "sklearn-sgd-classifier",
            "device": "cpu",
            "checked": False,
            "available": True,
            "note": (
                "classifier SGD stays on CPU because probability outputs may be "
                "required."
            ),
        }
    ]


@pytest.mark.fast
def test_backend_availability_report_checks_sgd_selector_classifier_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_probe(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        raise AssertionError("unexpected backend probe")

    monkeypatch.setattr(
        gpu_backends, "cuml_mbsgd_classifier_availability", lambda: (True, "")
    )
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_regressor_availability", fail_probe)

    report = gpu_backends.backend_availability_report(
        device_info_from_resolved(
            "auto",
            "cuda",
            torch_cuda_available=True,
            cuda_driver_available=True,
        ),
        selected_models=["sgd-selector-classifier"],
    )

    assert report["checks"]["cuda_backend_checks"] is True
    assert report["models"] == [
        {
            "model": "sgd-selector",
            "backend": "cuml-mbsgd-classifier",
            "device": "cuda",
            "checked": True,
            "available": True,
            "fallback_backend": "sklearn-sgd-classifier",
        }
    ]


@pytest.mark.fast
def test_backend_availability_report_can_skip_smoke_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_probe(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        raise AssertionError("unexpected backend probe")

    monkeypatch.setattr(gpu_backends, "tabpfn_api_status", fail_probe)
    monkeypatch.setattr(gpu_backends, "tabpfn_v25_api_status", fail_probe)
    monkeypatch.setattr(gpu_backends, "cuml_knn_availability", fail_probe)
    monkeypatch.setattr(gpu_backends, "lightgbm_cuda_availability", fail_probe)
    monkeypatch.setattr(gpu_backends, "catboost_cuda_availability", fail_probe)
    monkeypatch.setattr(gpu_backends, "xgboost_cuda_availability", fail_probe)

    report = gpu_backends.backend_availability_report(
        device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True),
        selected_models=["knn", "lgbm", "catboost", "xgboost", "tabpfn"],
        run_smoke_checks=False,
    )

    assert report["checks"]["cuda_backend_checks"] is False
    assert "tabpfn_api_available" not in report
    assert "cuml_knn_available" not in report
    assert "xgboost_cuda_available" not in report
    assert report_model(report, model="lgbm")["backend"] == "lightgbm-cpu"
    assert report_model(
        report, model="catboost", backend="catboost-gpu"
    )["checked"] is False
    assert report_model(
        report, model="xgboost", backend="xgboost-cuda"
    )["checked"] is False


@pytest.mark.fast
def test_backend_availability_report_separates_driver_cuda_from_torch_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_probe(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        raise AssertionError("unexpected backend probe")

    monkeypatch.setattr(gpu_backends, "tabpfn_api_status", fail_probe)
    monkeypatch.setattr(gpu_backends, "tabpfn_v25_api_status", fail_probe)
    monkeypatch.setattr(
        gpu_backends,
        "xgboost_cuda_availability",
        lambda: (True, ""),
    )

    report = gpu_backends.backend_availability_report(
        device_info_from_resolved(
            "auto",
            "cuda",
            torch_cuda_available=False,
            cuda_driver_available=True,
        ),
        selected_models=["xgboost"],
    )

    assert report["device"]["resolved"] == "cuda"
    assert report["device"]["torch_device"] == "cpu"
    assert report["device"]["torch_cuda_ready"] is False
    assert report["device"]["nvidia_driver_ready"] is True
    assert report["device"]["cuda_ready"] is True
    assert report["checks"]["cuda_backend_checks"] is True
    assert report_model(
        report, model="xgboost", backend="xgboost-cuda"
    )["available"] is True


@pytest.mark.fast
def test_xgboost_cuda_uses_single_cpu_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_backends, "xgboost_cuda_available", lambda: True)
    monkeypatch.setattr(
        gpu_backends,
        "xgboost_cuda_args",
        lambda: {"device": "cuda", "tree_method": "hist"},
    )
    model = XGBoostClassifier()

    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.fixed_args["n_jobs"] == 1
    assert model.fixed_args["device"] == "cuda"

    model.set_device_info(device_info_from_resolved("cpu", "cpu"))

    assert model.fixed_args["n_jobs"] == -1
    assert "device" not in model.fixed_args

@pytest.mark.fast
def test_xgboost_booster_cuda_detection_rejects_cpu_fallback() -> None:
    class FakeBooster:
        def save_config(self) -> str:
            return '{"learner": {"generic_param": {"device": "cpu"}}}'

    class FakeModel:
        def get_booster(self) -> FakeBooster:
            return FakeBooster()

    assert gpu_backends._xgboost_booster_uses_cuda(FakeModel()) is False


@pytest.mark.fast
def test_xgboost_booster_cuda_detection_accepts_cuda_device() -> None:
    class FakeBooster:
        def save_config(self) -> str:
            return '{"learner": {"generic_param": {"device": "cuda:0"}}}'

    class FakeModel:
        def get_booster(self) -> FakeBooster:
            return FakeBooster()

    assert gpu_backends._xgboost_booster_uses_cuda(FakeModel()) is True


@pytest.mark.fast
def test_lr_cuda_uses_cuml_and_drops_sklearn_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCumlLogisticRegression:
        def __init__(
            self,
            max_iter: int = 1000,
            penalty: str = "l2",
            C: float = 1.0,
            solver: str = "qn",
            tol: float = 1e-4,
            l1_ratio: float | None = None,
            output_type: str | None = None,
        ) -> None:
            self.args = {
                "max_iter": max_iter,
                "penalty": penalty,
                "C": C,
                "solver": solver,
                "tol": tol,
                "l1_ratio": l1_ratio,
                "output_type": output_type,
            }

    monkeypatch.setattr(
        gpu_backends,
        "get_cuml_logistic_regression_cls",
        lambda: FakeCumlLogisticRegression,
    )
    monkeypatch.setattr(
        gpu_backends, "cuml_logistic_regression_available", lambda: True
    )
    model = LRClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model_cls, clean_args = model.model_cls_args(
        {
            "max_iter": 2000,
            "penalty": "elasticnet",
            "solver": "saga",
            "l1_ratio": 0.25,
            "C": 2.0,
        }
    )

    assert model_cls is FakeCumlLogisticRegression
    assert clean_args == {
        "max_iter": 2000,
        "penalty": "l2",
        "solver": "qn",
        "tol": 1e-2,
        "C": 2.0,
        "output_type": "numpy",
    }
    assert model.backend_name() == "cuml-logistic-regression"


@pytest.mark.fast
def test_lr_cuda_falls_back_when_cuml_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "get_cuml_logistic_regression_cls", lambda: None)
    gpu_backends._WARNED_MISSING_CUML_BACKENDS.clear()
    model = LRClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))
    args = {"max_iter": 2000, "penalty": "elasticnet", "solver": "saga"}

    with pytest.warns(RuntimeWarning, match="cuML"):
        model_cls, clean_args = model.model_cls_args(args)

    assert model_cls is LogisticRegression
    assert clean_args == args
    assert model.backend_name() == "sklearn-logistic-regression"


@pytest.mark.fast
def test_elasticnet_cuda_uses_cuml(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCumlElasticNet:
        def __init__(
            self,
            alpha: float = 1.0,
            l1_ratio: float = 0.5,
            max_iter: int = 1000,
            output_type: str | None = None,
        ) -> None:
            self.args = {
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "max_iter": max_iter,
                "output_type": output_type,
            }

    monkeypatch.setattr(
        gpu_backends, "get_cuml_elasticnet_cls", lambda: FakeCumlElasticNet
    )
    monkeypatch.setattr(gpu_backends, "cuml_elasticnet_available", lambda: True)
    model = ElasticNetRegressor()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model_cls, clean_args = model.model_cls_args(
        {"alpha": 0.2, "l1_ratio": 0.7, "max_iter": 2000}
    )

    assert model_cls is FakeCumlElasticNet
    assert clean_args == {
        "alpha": 0.2,
        "l1_ratio": 0.7,
        "max_iter": 2000,
        "output_type": "numpy",
    }
    assert model.backend_name() == "cuml-elasticnet"


@pytest.mark.fast
@pytest.mark.parametrize("resolved", ["cpu", "mps"])
def test_elasticnet_cpu_and_mps_keep_sklearn(resolved: str) -> None:
    model = ElasticNetRegressor()
    model.set_device_info(device_info_from_resolved("auto", resolved))
    args = {"alpha": 0.2, "l1_ratio": 0.7, "max_iter": 2000}

    model_cls, clean_args = model.model_cls_args(args)

    assert model_cls is ElasticNet
    assert clean_args == args
    assert model.backend_name() == "sklearn-elasticnet"


@pytest.mark.fast
@pytest.mark.parametrize(
    ("model", "full_args", "expected_cls", "expected_args"),
    [
        (
            SVMClassifier(),
            {"kernel": "rbf", "C": 4.0, "gamma": 0.1, "cache_size": 1000},
            SVC,
            {"kernel": "rbf", "C": 4.0, "gamma": 0.1, "cache_size": 1000},
        ),
        (
            SVMClassifier(),
            {"kernel": "linear", "C": 4.0, "gamma": 0.1, "cache_size": 1000},
            LinearSVC,
            {"C": 4.0},
        ),
        (
            SVMRegressor(),
            {"kernel": "rbf", "C": 3.0, "gamma": 0.1, "cache_size": 1000},
            SVR,
            {"kernel": "rbf", "C": 3.0, "gamma": 0.1, "cache_size": 1000},
        ),
        (
            SVMRegressor(),
            {"kernel": "linear", "C": 3.0, "gamma": 0.1, "cache_size": 1000},
            LinearSVR,
            {"C": 3.0},
        ),
    ],
)
def test_svm_cuda_keeps_sklearn_backend(
    model: SVMClassifier | SVMRegressor,
    full_args: dict[str, Any],
    expected_cls: type,
    expected_args: dict[str, Any],
) -> None:
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model_cls, clean_args = model.model_cls_args(full_args)

    assert model_cls is expected_cls
    assert clean_args == expected_args
    assert model.backend_name(full_args) == "sklearn-svm"


@pytest.mark.fast
def test_model_prediction_outputs_are_converted_to_numpy() -> None:
    class FakeGet:
        def __init__(self, value: Any) -> None:
            self.value = value

        def get(self) -> Any:
            return self.value

    class FakePredictor:
        def predict(self, X: pd.DataFrame) -> FakeGet:
            return FakeGet([0, 1])

        def predict_proba(self, X: pd.DataFrame) -> FakeGet:
            return FakeGet([[0.8, 0.2], [0.1, 0.9]])

    X, _ = tiny_tuning_data()
    X = X.iloc[:2]
    model = LRClassifier()
    model.model = FakePredictor()
    model.tuned_model = FakePredictor()
    model.tuned_args = {}

    assert np.array_equal(model.predict(X), np.array([0, 1]))
    assert np.array_equal(model.tuned_predict(X), np.array([0, 1]))
    assert np.array_equal(
        model.predict_proba(X),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


@pytest.mark.fast
def test_cuml_mbsgd_lazy_import_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMBSGDClassifier:
        pass

    class FakeMBSGDRegressor:
        pass

    linear_model = ModuleType("cuml.linear_model")
    linear_model.MBSGDClassifier = FakeMBSGDClassifier  # type: ignore[attr-defined]
    linear_model.MBSGDRegressor = FakeMBSGDRegressor  # type: ignore[attr-defined]
    cuml = ModuleType("cuml")
    monkeypatch.setitem(sys.modules, "cuml", cuml)
    monkeypatch.setitem(sys.modules, "cuml.linear_model", linear_model)

    assert gpu_backends.get_cuml_mbsgd_classifier_cls() is FakeMBSGDClassifier
    assert gpu_backends.get_cuml_mbsgd_regressor_cls() is FakeMBSGDRegressor


@pytest.mark.fast
def test_sgd_regressor_cuda_maps_args_and_drops_sklearn_only_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMBSGDRegressor:
        def __init__(
            self,
            loss: str = "squared_loss",
            penalty: str = "l2",
            eta0: float = 0.001,
            epochs: int = 1000,
            output_type: str | None = None,
        ) -> None:
            self.args = {
                "loss": loss,
                "penalty": penalty,
                "eta0": eta0,
                "epochs": epochs,
                "output_type": output_type,
            }

    monkeypatch.setattr(
        gpu_backends, "get_cuml_mbsgd_regressor_cls", lambda: FakeMBSGDRegressor
    )
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_regressor_available", lambda: True)
    model = SGDRegressor()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model_cls, clean_args = model.model_cls_args(
        {
            "loss": "squared_error",
            "penalty": "l1",
            "eta0": 0.1,
            "max_iter": 25,
            "early_stopping": True,
            "average": 3,
        }
    )

    assert model_cls is FakeMBSGDRegressor
    assert clean_args == {
        "loss": "squared_loss",
        "penalty": "l1",
        "eta0": 0.1,
        "epochs": 25,
        "output_type": "numpy",
    }
    assert model.backend_name({"loss": "squared_error"}) == "cuml-mbsgd-regressor"


@pytest.mark.fast
def test_sgd_regressor_cuda_loss_search_space_is_cuml_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMBSGDRegressor:
        pass

    monkeypatch.setattr(
        gpu_backends, "get_cuml_mbsgd_regressor_cls", lambda: FakeMBSGDRegressor
    )
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_regressor_available", lambda: True)
    trial = RecordingTrial()
    model = SGDRegressor()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    args = model.optuna_args(trial)  # type: ignore[arg-type]

    assert args["loss"] == "squared_error"
    assert trial.choices["loss"] == ["squared_error"]
    assert trial.floats["eta0"] == (1e-4, 1e-1, {"log": True})
    assert "early_stopping" not in args


@pytest.mark.fast
def test_sgd_regressor_cpu_loss_search_space_stays_full() -> None:
    trial = RecordingTrial()
    model = SGDRegressor()

    args = model.optuna_args(trial)  # type: ignore[arg-type]

    assert args["loss"] == "squared_error"
    assert trial.choices["loss"] == [
        "squared_error",
        "huber",
        "epsilon_insensitive",
    ]


@pytest.mark.fast
def test_sgd_regressor_cuda_falls_back_for_unsupported_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMBSGDRegressor:
        def __init__(self, loss: str = "squared_loss") -> None:
            self.loss = loss

    monkeypatch.setattr(
        gpu_backends, "get_cuml_mbsgd_regressor_cls", lambda: FakeMBSGDRegressor
    )
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_regressor_available", lambda: True)
    model = SGDRegressor()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))
    args = {"loss": "huber", "eta0": 0.1, "early_stopping": True}

    model_cls, clean_args = model.model_cls_args(args)

    assert model_cls is SklearnSGDRegressor
    assert clean_args == args
    assert model.backend_name(args) == "sklearn-sgd-regressor"


@pytest.mark.fast
def test_sgd_selector_classifier_cuda_maps_log_and_hinge_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMBSGDClassifier:
        def __init__(
            self,
            loss: str = "hinge",
            penalty: str = "l2",
            alpha: float = 0.0001,
            eta0: float = 0.001,
            output_type: str | None = None,
        ) -> None:
            self.args = {
                "loss": loss,
                "penalty": penalty,
                "alpha": alpha,
                "eta0": eta0,
                "output_type": output_type,
            }

    monkeypatch.setattr(
        gpu_backends, "get_cuml_mbsgd_classifier_cls", lambda: FakeMBSGDClassifier
    )
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_classifier_available", lambda: True)
    model = SGDClassifierSelector()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model_cls, clean_args = model.model_cls_args(
        {
            "loss": "log_loss",
            "penalty": "l1",
            "alpha": 0.01,
            "eta0": 0.2,
            "early_stopping": True,
            "average": 5,
        }
    )

    assert model_cls is FakeMBSGDClassifier
    assert clean_args == {
        "loss": "log",
        "penalty": "l1",
        "alpha": 0.01,
        "eta0": 0.2,
        "output_type": "numpy",
    }
    assert model.backend_name({"loss": "log_loss"}) == "cuml-mbsgd-classifier"


@pytest.mark.fast
def test_sgd_selector_classifier_cuda_search_space_is_cuml_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMBSGDClassifier:
        pass

    monkeypatch.setattr(
        gpu_backends, "get_cuml_mbsgd_classifier_cls", lambda: FakeMBSGDClassifier
    )
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_classifier_available", lambda: True)
    trial = RecordingTrial()
    model = SGDClassifierSelector()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    args = model.optuna_args(trial)  # type: ignore[arg-type]

    assert args["loss"] == "hinge"
    assert trial.choices["loss"] == ["hinge", "log_loss"]
    assert trial.floats["eta0"] == (1e-4, 1e-1, {"log": True})
    assert trial.floats["alpha"] == (1e-6, 1e-2, {"log": True})
    assert "early_stopping" not in args
    assert "average" not in args


@pytest.mark.fast
def test_sgd_selector_classifier_cuda_falls_back_for_unsupported_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMBSGDClassifier:
        def __init__(self, loss: str = "hinge") -> None:
            self.loss = loss

    monkeypatch.setattr(
        gpu_backends, "get_cuml_mbsgd_classifier_cls", lambda: FakeMBSGDClassifier
    )
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_classifier_available", lambda: True)
    model = SGDClassifierSelector()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))
    args = {"loss": "squared_hinge", "average": 3, "early_stopping": False}

    model_cls, clean_args = model.model_cls_args(args)

    assert model_cls is SklearnSGDClassifier
    assert clean_args == args
    assert model.backend_name(args) == "sklearn-sgd-classifier"


@pytest.mark.fast
def test_sgd_selector_regressor_cuda_loss_search_space_is_cuml_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMBSGDRegressor:
        pass

    monkeypatch.setattr(
        gpu_backends, "get_cuml_mbsgd_regressor_cls", lambda: FakeMBSGDRegressor
    )
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_regressor_available", lambda: True)
    trial = RecordingTrial()
    model = SGDRegressorSelector()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    args = model.optuna_args(trial)  # type: ignore[arg-type]

    assert args["loss"] == "squared_error"
    assert trial.choices["loss"] == ["squared_error"]
    assert trial.floats["eta0"] == (1e-4, 1e-1, {"log": True})
    assert trial.floats["alpha"] == (1e-6, 1e-2, {"log": True})
    assert "early_stopping" not in args
    assert "average" not in args


@pytest.mark.fast
def test_sgd_selector_regressor_cpu_loss_search_space_stays_full() -> None:
    trial = RecordingTrial()
    model = SGDRegressorSelector()

    args = model.optuna_args(trial)  # type: ignore[arg-type]

    assert args["loss"] == "squared_error"
    assert trial.choices["loss"] == [
        "squared_error",
        "huber",
        "epsilon_insensitive",
        "squared_epsilon_insensitive",
    ]


@pytest.mark.fast
def test_main_sgd_classifier_stays_sklearn_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMBSGDClassifier:
        pass

    monkeypatch.setattr(
        gpu_backends, "get_cuml_mbsgd_classifier_cls", lambda: FakeMBSGDClassifier
    )
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_classifier_available", lambda: True)
    model = SGDClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model_cls, clean_args = model.model_cls_args({"loss": "log_loss"})

    assert model_cls is SklearnSGDClassifier
    assert clean_args == {"loss": "log_loss"}
    assert model.backend_name({"loss": "log_loss"}) == "sklearn-sgd-classifier"


@pytest.mark.fast
def test_sgd_selector_regressor_backend_name_tracks_supported_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMBSGDRegressor:
        def __init__(self, loss: str = "squared_loss") -> None:
            self.loss = loss

    monkeypatch.setattr(
        gpu_backends, "get_cuml_mbsgd_regressor_cls", lambda: FakeMBSGDRegressor
    )
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_regressor_available", lambda: True)
    model = SGDRegressorSelector()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.backend_name({"loss": "squared_error"}) == "cuml-mbsgd-regressor"
    assert model.backend_name({"loss": "huber"}) == "sklearn-sgd-regressor"
