from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from df_analyze.device import device_info_from_resolved
from df_analyze.enumerables import (
    ClassifierScorer,
    DfAnalyzeClassifier,
    DfAnalyzeRegressor,
)
from df_analyze.models import gpu as gpu_backends
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.models.catboost import CatBoostClassifier, CatBoostRegressor


class FakeCatBoostBase:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FakeCatBoostBase:
        self.fit_shape = X.shape
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.arange(len(X)).reshape(-1, 1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.zeros((len(X), 2), dtype=float)
        probs[:, 0] = 0.75
        probs[:, 1] = 0.25
        return probs


class FakeCatBoostClassifier(FakeCatBoostBase):
    pass


class FakeCatBoostRegressor(FakeCatBoostBase):
    pass


def install_fake_catboost(monkeypatch: pytest.MonkeyPatch) -> None:
    catboost = ModuleType("catboost")
    catboost.CatBoostClassifier = FakeCatBoostClassifier  # type: ignore[attr-defined]
    catboost.CatBoostRegressor = FakeCatBoostRegressor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "catboost", catboost)


def tiny_tuning_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y = pd.Series([0, 1, 0, 1], name="target")
    return X, y


@pytest.mark.fast
def test_catboost_registered_but_not_default() -> None:
    assert DfAnalyzeClassifier.CatBoost.get_model() is CatBoostClassifier
    assert DfAnalyzeRegressor.CatBoost.get_model() is CatBoostRegressor
    assert DfAnalyzeClassifier.CatBoost.value in DfAnalyzeClassifier.choices()
    assert DfAnalyzeRegressor.CatBoost.value in DfAnalyzeRegressor.choices()
    assert DfAnalyzeClassifier.CatBoost.value not in DfAnalyzeClassifier.defaults()
    assert DfAnalyzeRegressor.CatBoost.value not in DfAnalyzeRegressor.defaults()


@pytest.mark.fast
def test_catboost_cuda_device_adds_gpu_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_backends, "catboost_cuda_available", lambda: True)
    model = CatBoostClassifier()

    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.fixed_args["task_type"] == "GPU"
    assert model.fixed_args["devices"] == "0"
    assert model.backend_name() == "catboost-gpu"


@pytest.mark.fast
def test_catboost_driver_only_cuda_can_use_backend_specific_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "catboost_cuda_available", lambda: True)
    model = CatBoostClassifier()

    model.set_device_info(
        device_info_from_resolved(
            "auto",
            "cuda",
            torch_cuda_available=False,
            cuda_driver_available=True,
        )
    )

    assert model.device_info.torch_device == "cpu"
    assert model.device_info.accelerated is False
    assert model.fixed_args["task_type"] == "GPU"
    assert model.backend_name() == "catboost-gpu"


@pytest.mark.fast
def test_catboost_cuda_devices_can_come_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "catboost_cuda_available", lambda: True)
    monkeypatch.setenv("DF_ANALYZE_CATBOOST_DEVICES", "1")
    model = CatBoostClassifier()

    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.fixed_args["task_type"] == "GPU"
    assert model.fixed_args["devices"] == "1"


@pytest.mark.fast
def test_catboost_cuda_falls_back_when_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "catboost_cuda_available", lambda: False)
    gpu_backends._WARNED_UNAVAILABLE_GPU_BACKENDS.clear()
    model = CatBoostClassifier()

    with pytest.warns(RuntimeWarning, match="CatBoost"):
        model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.fixed_args["task_type"] == "CPU"
    assert "devices" not in model.fixed_args
    assert model.backend_name() == "catboost-cpu"


@pytest.mark.fast
@pytest.mark.parametrize("resolved", ["cpu", "mps"])
def test_catboost_cpu_and_mps_use_cpu_args(
    monkeypatch: pytest.MonkeyPatch,
    resolved: str,
) -> None:
    monkeypatch.setattr(gpu_backends, "catboost_cuda_available", lambda: True)
    model = CatBoostClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model.set_device_info(device_info_from_resolved("auto", resolved))

    assert model.fixed_args["task_type"] == "CPU"
    assert "devices" not in model.fixed_args
    assert model.backend_name() == "catboost-cpu"


@pytest.mark.fast
def test_catboost_cuda_tuning_forces_single_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "catboost_cuda_available", lambda: True)
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

    monkeypatch.setattr(DfAnalyzeModel, "htune_optuna", fake_htune_optuna)
    X, y = tiny_tuning_data()
    model = CatBoostClassifier()
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
def test_catboost_cuda_tuning_trial_budget_caps_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "catboost_cuda_available", lambda: True)
    model = CatBoostClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.effective_tuning_n_trials(100) == 75
    assert model.effective_tuning_n_trials(50) == 50

    model.set_device_info(device_info_from_resolved("cpu", "cpu"))
    assert model.effective_tuning_n_trials(100) == 100


@pytest.mark.fast
def test_catboost_cuda_target_score_stopping_requires_minimum_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "catboost_cuda_available", lambda: True)
    model = CatBoostClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.target_score_stopping_min_trials() == 5

    model.set_device_info(device_info_from_resolved("cpu", "cpu"))
    assert model.target_score_stopping_min_trials() == 1


@pytest.mark.fast
def test_catboost_lazy_import_returns_fake_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_catboost(monkeypatch)
    model = CatBoostClassifier()
    args = {"iterations": 100, "depth": 4}

    model_cls, clean_args = model.model_cls_args(args)

    assert model_cls is FakeCatBoostClassifier
    assert clean_args == args


@pytest.mark.fast
def test_catboost_lazy_import_returns_fake_regressor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_catboost(monkeypatch)
    model = CatBoostRegressor()
    args = {"iterations": 100, "depth": 4}

    model_cls, clean_args = model.model_cls_args(args)

    assert model_cls is FakeCatBoostRegressor
    assert clean_args == args


@pytest.mark.fast
def test_catboost_prediction_outputs_are_numpy_1d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_catboost(monkeypatch)
    X, y = tiny_tuning_data()
    model = CatBoostClassifier()
    model.fit(X, y)
    model.refit_tuned(X, y)
    model.tuned_args = {}

    assert np.array_equal(model.predict(X), np.array([0, 1, 2, 3]))
    assert np.array_equal(model.tuned_predict(X), np.array([0, 1, 2, 3]))
    assert model.predict_proba(X).shape == (len(X), 2)
