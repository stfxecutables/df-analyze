from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier

from df_analyze.analysis.univariate.predict import models as pred_models
from df_analyze.analysis.univariate.predict import predict as pred_module
from df_analyze.device import device_info_from_resolved
from df_analyze.models import gpu as gpu_backends


@pytest.mark.fast
def test_univariate_sgd_regressor_uses_cuml_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMBSGDRegressor:
        def __init__(self, output_type: str | None = None) -> None:
            self.output_type = output_type

    monkeypatch.setattr(
        gpu_backends, "get_cuml_mbsgd_regressor_cls", lambda: FakeMBSGDRegressor
    )
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_regressor_available", lambda: True)

    model = pred_models.SGDRegressor(
        device_info=device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True)
    )

    assert model.model is FakeMBSGDRegressor
    assert model.fixed_args["output_type"] == "numpy"
    assert model.grid == {"loss": ["squared_loss"], "penalty": ["l1", "l2"]}
    assert model.uses_gpu_backend is True


@pytest.mark.fast
def test_univariate_sgd_classifier_stays_sklearn_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMBSGDClassifier:
        pass

    monkeypatch.setattr(
        gpu_backends, "get_cuml_mbsgd_classifier_cls", lambda: FakeMBSGDClassifier
    )

    model = pred_models.SGDClassifier(
        device_info=device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True)
    )

    assert model.model is SklearnSGDClassifier
    assert model.uses_gpu_backend is False


@pytest.mark.fast
@pytest.mark.parametrize(
    (
        "resolved",
        "is_classification",
        "cuml_regressor_cls",
        "cuml_regressor_available",
        "expected_n_jobs",
    ),
    [
        ("cpu", True, None, False, 8),
        ("cuda", True, None, False, 8),
        ("cuda", False, None, False, 8),
        ("cuda", False, "fake", True, 1),
    ],
)
def test_feature_target_predictions_uses_device_parallelism_and_passes_device(
    monkeypatch: pytest.MonkeyPatch,
    resolved: str,
    is_classification: bool,
    cuml_regressor_cls: str | None,
    cuml_regressor_available: bool,
    expected_n_jobs: int,
) -> None:
    calls: dict[str, Any] = {"n_jobs": [], "devices": []}
    monkeypatch.delenv(pred_module.PRED_N_JOBS_ENV, raising=False)
    monkeypatch.setattr(pred_module.os, "cpu_count", lambda: 32)

    class FakeMBSGDRegressor:
        def __init__(self, output_type: str | None = None) -> None:
            self.output_type = output_type

    class FakeParallel:
        def __init__(self, n_jobs: int) -> None:
            calls["n_jobs"].append(n_jobs)

        def __call__(self, tasks: Any) -> list[Any]:
            results = []
            for task in tasks:
                func, args, kwargs = task
                results.append(func(*args, **kwargs))
            return results

    def fake_continuous_feature_target_preds(
        continuous: pd.DataFrame,
        column: str,
        target: pd.Series,
        is_classification: bool,
        device_info: Any = None,
    ) -> tuple[pd.DataFrame, None, list[Any]]:
        calls["devices"].append(None if device_info is None else device_info.resolved)
        df = pd.DataFrame({"acc": [1.0]}, index=pd.Index([column], name="feature"))
        return df, None, []

    monkeypatch.setattr(pred_module, "Parallel", FakeParallel)
    monkeypatch.setattr(
        pred_module,
        "continuous_feature_target_preds",
        fake_continuous_feature_target_preds,
    )
    monkeypatch.setattr(
        gpu_backends,
        "get_cuml_mbsgd_regressor_cls",
        lambda: FakeMBSGDRegressor if cuml_regressor_cls is not None else None,
    )
    monkeypatch.setattr(
        gpu_backends,
        "cuml_mbsgd_regressor_available",
        lambda: cuml_regressor_available,
    )
    monkeypatch.setattr(gpu_backends, "warn_missing_cuml_backend", lambda name: None)
    monkeypatch.setattr(
        gpu_backends, "warn_unavailable_gpu_backend", lambda name, reason=None: None
    )
    device_info = device_info_from_resolved("auto", resolved)

    df_cont, df_cat, errs, warns = pred_module.feature_target_predictions(
        categoricals=None,
        continuous=pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]}),
        target=pd.Series([0, 1, 0, 1], name="target"),
        is_classification=is_classification,
        device_info=device_info,
    )

    assert calls["n_jobs"] == [expected_n_jobs]
    assert calls["devices"] == [resolved]
    assert df_cont is not None
    assert df_cat is None
    assert errs == []
    assert warns == []


@pytest.mark.fast
def test_cpu_parallel_n_jobs_accepts_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(pred_module.PRED_N_JOBS_ENV, "12")

    assert pred_module.cpu_parallel_n_jobs() == 12
