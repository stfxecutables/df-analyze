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
from df_analyze.models.xgboost import XGBoostClassifier, XGBoostRegressor


class FakeXGBoostBase:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FakeXGBoostBase:
        self.fit_shape = X.shape
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.arange(len(X)).reshape(-1, 1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.zeros((len(X), 2), dtype=float)
        probs[:, 0] = 0.75
        probs[:, 1] = 0.25
        return probs


class FakeXGBoostClassifier(FakeXGBoostBase):
    pass


class FakeXGBoostRegressor(FakeXGBoostBase):
    pass


def install_fake_xgboost(monkeypatch: pytest.MonkeyPatch) -> None:
    xgboost = ModuleType("xgboost")
    xgboost.XGBClassifier = FakeXGBoostClassifier  # type: ignore[attr-defined]
    xgboost.XGBRegressor = FakeXGBoostRegressor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xgboost", xgboost)


def tiny_tuning_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y = pd.Series([0, 1, 0, 1], name="target")
    return X, y


@pytest.mark.fast
def test_xgboost_registered_but_not_default() -> None:
    assert DfAnalyzeClassifier.XGBoost.get_model() is XGBoostClassifier
    assert DfAnalyzeRegressor.XGBoost.get_model() is XGBoostRegressor
    assert DfAnalyzeClassifier.XGBoost.value in DfAnalyzeClassifier.choices()
    assert DfAnalyzeRegressor.XGBoost.value in DfAnalyzeRegressor.choices()
    assert DfAnalyzeClassifier.XGBoost.value not in DfAnalyzeClassifier.defaults()
    assert DfAnalyzeRegressor.XGBoost.value not in DfAnalyzeRegressor.defaults()


@pytest.mark.fast
def test_xgboost_cuda_device_adds_gpu_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_backends, "xgboost_cuda_available", lambda: True)
    model = XGBoostClassifier()

    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.fixed_args["device"] == "cuda"
    assert model.fixed_args["tree_method"] == "hist"
    assert model.backend_name() == "xgboost-cuda"


@pytest.mark.fast
def test_xgboost_cuda_device_can_come_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "xgboost_cuda_available", lambda: True)
    monkeypatch.setenv("DF_ANALYZE_XGBOOST_DEVICE", "cuda:1")
    model = XGBoostClassifier()

    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.fixed_args["device"] == "cuda:1"
    assert model.backend_name() == "xgboost-cuda"


@pytest.mark.fast
def test_xgboost_cuda_falls_back_when_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "xgboost_cuda_available", lambda: False)
    gpu_backends._WARNED_UNAVAILABLE_GPU_BACKENDS.clear()
    model = XGBoostClassifier()

    with pytest.warns(RuntimeWarning, match="XGBoost"):
        model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert "device" not in model.fixed_args
    assert "tree_method" not in model.fixed_args
    assert model.fixed_args["n_jobs"] == -1
    assert model.backend_name() == "xgboost-cpu"


@pytest.mark.fast
@pytest.mark.parametrize("resolved", ["cpu", "mps"])
def test_xgboost_cpu_and_mps_use_cpu_args(
    monkeypatch: pytest.MonkeyPatch,
    resolved: str,
) -> None:
    monkeypatch.setattr(gpu_backends, "xgboost_cuda_available", lambda: True)
    model = XGBoostClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model.set_device_info(device_info_from_resolved("auto", resolved))

    assert "device" not in model.fixed_args
    assert "tree_method" not in model.fixed_args
    assert model.backend_name() == "xgboost-cpu"


@pytest.mark.fast
@pytest.mark.parametrize("device", ["cuda", "cuda:1"])
def test_xgboost_cuda_tuning_forces_single_job(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    monkeypatch.setattr(gpu_backends, "xgboost_cuda_available", lambda: True)
    monkeypatch.setenv("DF_ANALYZE_XGBOOST_DEVICE", device)
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
    model = XGBoostClassifier()
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
def test_xgboost_cuda_tuning_trial_budget_caps_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "xgboost_cuda_available", lambda: True)
    model = XGBoostClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.effective_tuning_n_trials(100) == 75
    assert model.effective_tuning_n_trials(50) == 50

    model.set_device_info(device_info_from_resolved("cpu", "cpu"))
    assert model.effective_tuning_n_trials(100) == 100


@pytest.mark.fast
def test_xgboost_cuda_target_score_stopping_requires_minimum_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_backends, "xgboost_cuda_available", lambda: True)
    model = XGBoostClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model.target_score_stopping_min_trials() == 5

    model.set_device_info(device_info_from_resolved("cpu", "cpu"))
    assert model.target_score_stopping_min_trials() == 1


@pytest.mark.fast
def test_xgboost_lazy_import_returns_fake_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_xgboost(monkeypatch)
    model = XGBoostClassifier()
    args = {"n_estimators": 100, "max_depth": 4}

    model_cls, clean_args = model.model_cls_args(args)

    assert model_cls is FakeXGBoostClassifier
    assert clean_args == args


@pytest.mark.fast
def test_xgboost_lazy_import_returns_fake_regressor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_xgboost(monkeypatch)
    model = XGBoostRegressor()
    args = {"n_estimators": 100, "max_depth": 4}

    model_cls, clean_args = model.model_cls_args(args)

    assert model_cls is FakeXGBoostRegressor
    assert clean_args == args


@pytest.mark.fast
def test_xgboost_prediction_outputs_are_numpy_1d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_xgboost(monkeypatch)
    X, y = tiny_tuning_data()
    model = XGBoostClassifier()
    model.fit(X, y)
    model.refit_tuned(X, y)
    model.tuned_args = {}

    assert np.array_equal(model.predict(X), np.array([0, 1, 2, 3]))
    assert np.array_equal(model.tuned_predict(X), np.array([0, 1, 2, 3]))
    assert model.predict_proba(X).shape == (len(X), 2)
