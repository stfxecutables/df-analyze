from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import json
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import pytest
from cli_test_helpers import ArgvContext

import df_analyze.device as device
import df_analyze.models.gpu as gpu_report
from df_analyze.cli.cli import ProgramOptions, get_options
from df_analyze.device import (
    device_info_from_json,
    device_info_from_resolved,
    is_fatal_accelerator_error,
)
from df_analyze.enumerables import ClassifierScorer, WrapperSelectionModel
from df_analyze.hypertune import HtuneResult, _device_result_fields, evaluate_tuned
from df_analyze.models import gpu as gpu_backends
from df_analyze.models.base import DfAnalyzeModel, TargetScoreStopping
from df_analyze.models.dummy import DummyClassifier
from df_analyze.models.gandalf import GandalfEstimator
from df_analyze.models.lgbm import (
    LightGBMClassifier,
    LightGBMRegressor,
    LightGBMRFClassifier,
)
from df_analyze.models.linear import (
    ElasticNetRegressor,
    LRClassifier,
    SGDClassifierSelector,
    SGDRegressor,
    SGDRegressorSelector,
)
from df_analyze.models.mlp import MLPEstimator
from df_analyze.selection.stepwise import StepwiseSelector, get_dfanalyze_score


def tiny_tuning_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y = pd.Series([0, 1, 0, 1], name="target")
    return X, y


def capture_base_tuning_n_jobs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
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
    return calls


class MinimalModel(DfAnalyzeModel):
    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return object, {}

    def optuna_args(self, trial: Any) -> dict[str, str | float | int]:
        return {}


class FatalAcceleratorModel(MinimalModel):
    shortname = "fatal"
    longname = "Fatal Accelerator Model"

    def htune_optuna(self, *args: Any, **kwargs: Any) -> object:
        raise RuntimeError("CUDA out of memory")


@pytest.mark.fast
def test_base_htune_uses_effective_tuning_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import df_analyze.models.base as base_module

    class BudgetModel(MinimalModel):
        def cuda_tuning_trial_budget(self) -> int | None:
            return 7

    class FakeStudy:
        best_params: dict[str, Any] = {}
        best_value = 1.0

        def __init__(self) -> None:
            self.optimize_kwargs: dict[str, Any] = {}

        def optimize(self, objective: Any, **kwargs: Any) -> None:
            self.optimize_kwargs = kwargs

    study = FakeStudy()
    monkeypatch.setattr(base_module, "create_study", lambda **kwargs: study)
    monkeypatch.setattr(BudgetModel, "refit_tuned", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        BudgetModel,
        "optuna_objective",
        lambda *args, **kwargs: lambda trial: 1.0,
    )
    X, y = tiny_tuning_data()
    model = BudgetModel()

    returned = model.htune_optuna(
        X_train=X,
        y_train=y,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_trials=100,
        n_jobs=1,
    )

    assert returned is study
    assert study.optimize_kwargs["n_trials"] == 7


@pytest.mark.fast
def test_base_htune_uses_seeded_tpe_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    import df_analyze.models.base as base_module

    seen: dict[str, Any] = {}

    class FakeStudy:
        best_params: dict[str, Any] = {}

        def optimize(self, objective: Any, **kwargs: Any) -> None:
            pass

    def fake_tpe_sampler(*, seed: int) -> str:
        seen["seed"] = seed
        return "seeded-sampler"

    def fake_create_study(**kwargs: Any) -> FakeStudy:
        seen.update(kwargs)
        return FakeStudy()

    monkeypatch.setattr(base_module, "TPESampler", fake_tpe_sampler)
    monkeypatch.setattr(base_module, "create_study", fake_create_study)
    monkeypatch.setattr(MinimalModel, "refit_tuned", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        MinimalModel,
        "optuna_objective",
        lambda *args, **kwargs: lambda trial: 1.0,
    )
    X, y = tiny_tuning_data()

    MinimalModel().htune_optuna(
        X_train=X,
        y_train=y,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_trials=1,
        n_jobs=1,
    )

    assert seen["seed"] == base_module.SEED
    assert seen["sampler"] == "seeded-sampler"


@pytest.mark.fast
def test_base_model_device_info_does_not_imply_accelerated_backend() -> None:
    model = MinimalModel()
    assert model._tuning_n_jobs(-1) == -1
    assert model._calibration_n_jobs() == 5

    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    assert model._tuning_n_jobs(-1) == -1
    assert model._tuning_n_jobs(8) == 8
    assert model._calibration_n_jobs() == 5


@pytest.mark.fast
@pytest.mark.parametrize(
    ("best_value", "higher_is_better", "should_stop"),
    [
        (1.0, True, True),
        (0.999, True, False),
        (0.0, False, True),
        (0.1, False, False),
    ],
)
def test_target_score_stopping_stops_only_at_metric_optimum(
    best_value: float,
    higher_is_better: bool,
    should_stop: bool,
) -> None:
    class FakeStudy:
        def __init__(self, best_value: float) -> None:
            self.best_value = best_value
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    study = FakeStudy(best_value)
    callback = TargetScoreStopping(
        target=1.0 if higher_is_better else 0.0,
        higher_is_better=higher_is_better,
    )

    callback(study, object())  # type: ignore[arg-type]

    assert study.stopped is should_stop


@pytest.mark.fast
def test_target_score_stopping_respects_min_trials() -> None:
    class FakeStudy:
        best_value = 1.0
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    class FakeTrial:
        def __init__(self, number: int) -> None:
            self.number = number

    study = FakeStudy()
    callback = TargetScoreStopping(
        target=1.0,
        higher_is_better=True,
        min_trials=3,
    )

    callback(study, FakeTrial(0))  # type: ignore[arg-type]
    assert study.stopped is False

    callback(study, FakeTrial(2))  # type: ignore[arg-type]
    assert study.stopped is True


def stepwise_n_jobs(
    wrapper_model: WrapperSelectionModel,
    device_name: str,
    is_classification: bool,
) -> int:
    selector = object.__new__(StepwiseSelector)
    selector.options = SimpleNamespace(
        wrapper_model=wrapper_model,
        device_requested=device_name,
        device=device_name,
    )
    selector.prepared = SimpleNamespace(is_classification=is_classification)
    return selector._parallel_n_jobs()


@pytest.mark.fast
def test_auto_resolves_to_cpu_without_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: False)
    monkeypatch.setattr(device, "cuda_driver_available", lambda: False)
    monkeypatch.setattr(device, "mps_available", lambda: False)

    info = device.resolve_device("auto")

    assert info.requested == "auto"
    assert info.resolved == "cpu"
    assert info.torch_device == "cpu"
    assert info.skorch_device == "cpu"
    assert info.lightning_accelerator == "cpu"
    assert info.accelerated is False


@pytest.mark.fast
def test_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(device, "mps_available", lambda: True)

    info = device.resolve_device("auto")

    assert info.resolved == "cuda"
    assert info.lightning_accelerator == "gpu"
    assert info.lightning_devices == 1
    assert info.accelerated is True


@pytest.mark.fast
def test_auto_uses_cuda_driver_when_torch_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: False)
    monkeypatch.setattr(device, "cuda_driver_available", lambda: True)
    monkeypatch.setattr(device, "mps_available", lambda: True)

    info = device.resolve_device("auto")

    assert info.requested == "auto"
    assert info.resolved == "cuda"
    assert info.torch_device == "cpu"
    assert info.skorch_device == "cpu"
    assert info.lightning_accelerator == "cpu"
    assert info.accelerated is False
    assert info.torch_cuda_available is False
    assert info.cuda_driver_available is True
    assert info.cuda_backends_enabled is True


@pytest.mark.fast
def test_device_info_from_resolved_queries_cuda_availability_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: False)
    monkeypatch.setattr(device, "cuda_driver_available", lambda: True)
    monkeypatch.setattr(device, "mps_available", lambda: False)

    info = device_info_from_resolved("auto", "cuda")

    assert info.resolved == "cuda"
    assert info.torch_device == "cpu"
    assert info.skorch_device == "cpu"
    assert info.lightning_accelerator == "cpu"
    assert info.accelerated is False
    assert info.torch_cuda_available is False
    assert info.cuda_driver_available is True
    assert info.cuda_backends_enabled is True


@pytest.mark.fast
def test_device_info_from_resolved_queries_mps_availability_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: False)
    monkeypatch.setattr(device, "cuda_driver_available", lambda: False)
    monkeypatch.setattr(device, "mps_available", lambda: False)

    info = device_info_from_resolved("auto", "mps")

    assert info.resolved == "mps"
    assert info.torch_device == "cpu"
    assert info.skorch_device == "cpu"
    assert info.lightning_accelerator == "cpu"
    assert info.accelerated is False
    assert info.mps_available is False


@pytest.mark.fast
def test_auto_uses_mps_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: False)
    monkeypatch.setattr(device, "cuda_driver_available", lambda: False)
    monkeypatch.setattr(device, "mps_available", lambda: True)

    info = device.resolve_device("auto")

    assert info.resolved == "mps"
    assert info.lightning_accelerator == "mps"
    assert info.lightning_devices == 1
    assert info.accelerated is True


@pytest.mark.fast
def test_mps_requires_built_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mps = SimpleNamespace(
        is_available=lambda: True,
        is_built=lambda: False,
    )
    monkeypatch.setattr(device.torch, "backends", SimpleNamespace(mps=fake_mps))

    assert device.mps_available() is False


@pytest.mark.fast
def test_explicit_cpu_resolves_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(device, "mps_available", lambda: True)

    info = device.resolve_device("cpu")

    assert info.requested == "cpu"
    assert info.resolved == "cpu"
    assert info.accelerated is False


@pytest.mark.fast
def test_explicit_cuda_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: False)
    monkeypatch.setattr(device, "cuda_driver_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        device.resolve_device("cuda")


@pytest.mark.fast
def test_explicit_cuda_allows_driver_when_torch_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: False)
    monkeypatch.setattr(device, "cuda_driver_available", lambda: True)

    info = device.resolve_device("cuda")

    assert info.requested == "cuda"
    assert info.resolved == "cuda"
    assert info.torch_device == "cpu"
    assert info.lightning_accelerator == "cpu"
    assert info.accelerated is False
    assert info.cuda_driver_available is True
    assert info.cuda_backends_enabled is True


@pytest.mark.fast
def test_explicit_mps_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "mps_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS"):
        device.resolve_device("mps")


@pytest.mark.fast
def test_device_info_from_json_restores_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(device, "cuda_driver_available", lambda: True)
    monkeypatch.setattr(device, "mps_available", lambda: False)

    info = device_info_from_json({"requested": "auto", "resolved": "cuda"})

    assert info.requested == "auto"
    assert info.resolved == "cuda"
    assert info.torch_device == "cuda"


@pytest.mark.fast
def test_device_info_from_json_queries_missing_availability_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: False)
    monkeypatch.setattr(device, "cuda_driver_available", lambda: True)
    monkeypatch.setattr(device, "mps_available", lambda: False)

    info = device_info_from_json({"requested": "auto", "resolved": "cuda"})

    assert info.resolved == "cuda"
    assert info.torch_device == "cpu"
    assert info.skorch_device == "cpu"
    assert info.lightning_accelerator == "cpu"
    assert info.accelerated is False
    assert info.cuda_backends_enabled is True


@pytest.mark.fast
def test_device_info_from_json_restores_cpu_torch_device_for_backend_cuda() -> None:
    info = device_info_from_json(
        {
            "requested": "auto",
            "resolved": "cuda",
            "torch_device": "cpu",
        }
    )

    assert info.resolved == "cuda"
    assert info.torch_device == "cpu"
    assert info.skorch_device == "cpu"
    assert info.lightning_accelerator == "cpu"
    assert info.accelerated is False
    assert info.cuda_driver_available is True
    assert info.cuda_backends_enabled is True


@pytest.mark.fast
def test_cuda_driver_available_respects_hidden_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(device.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    assert device.cuda_driver_available() is False


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_cli_accepts_cpu_device() -> None:
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "data.parquet"
        path.touch()
        with ArgvContext(
            "df-analyze.py",
            "--df",
            f"{path}",
            "--device",
            "cpu",
            "--verbosity",
            "0",
        ):
            opts = get_options()

    assert opts.device_requested == "cpu"
    assert opts.device == "cpu"
    assert opts.device_info.torch_device == "cpu"


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_program_options_from_json_preserves_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(device, "mps_available", lambda: False)

    with TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        path = root / "data.parquet"
        outdir = root / "out"
        path.touch()
        with ArgvContext(
            "df-analyze.py",
            "--df",
            f"{path}",
            "--device",
            "auto",
            "--classifiers",
            "dummy",
            "--outdir",
            f"{outdir}",
            "--verbosity",
            "0",
        ):
            opts = get_options()
            monkeypatch.setattr(
                gpu_report,
                "backend_availability_report",
                lambda info, **_: {
                    "requested": info.requested,
                    "resolved": info.resolved,
                },
            )
        opts.to_json()
        saved_root = opts.program_dirs.root
        assert saved_root is not None
        assert (saved_root / "device_report.json").exists()

        monkeypatch.setattr(device, "cuda_available", lambda: False)
        monkeypatch.setattr(device, "mps_available", lambda: False)
        loaded = ProgramOptions.from_json(saved_root)

    assert loaded.device_requested == "auto"
    assert loaded.device == "cuda"
    assert loaded.device_info.torch_device == "cuda"


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_program_options_device_report_runs_selected_smoke_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_backend_availability_report(
        info: Any,
        selected_models: list[str] | None = None,
        *,
        run_smoke_checks: bool = False,
    ) -> dict[str, Any]:
        calls.append(
            {
                "resolved": info.resolved,
                "selected_models": selected_models,
                "run_smoke_checks": run_smoke_checks,
            }
        )
        return {"resolved": info.resolved, "run_smoke_checks": run_smoke_checks}

    with TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        path = root / "data.parquet"
        outdir = root / "out"
        path.touch()
        with ArgvContext(
            "df-analyze.py",
            "--df",
            f"{path}",
            "--device",
            "cpu",
            "--classifiers",
            "dummy",
            "--outdir",
            f"{outdir}",
            "--verbosity",
            "0",
        ):
            opts = get_options()
        monkeypatch.setattr(
            gpu_report,
            "backend_availability_report",
            fake_backend_availability_report,
        )

        opts.to_json()

    assert calls == [
        {
            "resolved": "cpu",
            "selected_models": ["dummy"],
            "run_smoke_checks": True,
        }
    ]


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_program_options_to_json_keeps_options_when_device_report_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        path = root / "data.parquet"
        outdir = root / "out"
        path.touch()
        with ArgvContext(
            "df-analyze.py",
            "--df",
            f"{path}",
            "--device",
            "cpu",
            "--classifiers",
            "dummy",
            "--outdir",
            f"{outdir}",
            "--verbosity",
            "0",
        ):
            opts = get_options()

        monkeypatch.setattr(
            gpu_report,
            "backend_availability_report",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        opts.to_json()
        saved_root = opts.program_dirs.root
        assert saved_root is not None
        assert (saved_root / "options.json").exists()
        assert not (saved_root / "device_report.json").exists()

    captured = capsys.readouterr()
    assert "Got error saving device report: boom" in captured.out


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_program_options_from_json_preserves_torch_cpu_for_backend_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: False)
    monkeypatch.setattr(device, "cuda_driver_available", lambda: True)
    monkeypatch.setattr(device, "mps_available", lambda: False)

    with TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        path = root / "data.parquet"
        outdir = root / "out"
        path.touch()
        with ArgvContext(
            "df-analyze.py",
            "--df",
            f"{path}",
            "--device",
            "auto",
            "--classifiers",
            "dummy",
            "--outdir",
            f"{outdir}",
            "--verbosity",
            "0",
        ):
            opts = get_options()
        assert opts.device == "cuda"
        assert opts.device_info.torch_device == "cpu"

        monkeypatch.setattr(
            gpu_report,
            "backend_availability_report",
            lambda info, **_: {"torch_device": info.torch_device},
        )
        opts.to_json()
        saved_root = opts.program_dirs.root
        assert saved_root is not None

        monkeypatch.setattr(device, "cuda_driver_available", lambda: False)
        loaded = ProgramOptions.from_json(saved_root)

    assert loaded.device == "cuda"
    assert loaded.device_info.torch_device == "cpu"
    assert loaded.device_info.lightning_accelerator == "cpu"


@pytest.mark.fast
def test_models_default_to_cpu_device_info() -> None:
    model = DummyClassifier()

    assert model.device_info.resolved == "cpu"
    assert model.device_info.accelerated is False


@pytest.mark.fast
def test_htune_result_exports_device_backend_and_timing_metadata() -> None:
    model = DummyClassifier()
    result = HtuneResult(
        selection="none",
        selected_cols=["x"],
        embed_select_model=None,
        model_cls=DummyClassifier,
        model=model,
        params={"strategy": "most_frequent"},
        metric=ClassifierScorer.Accuracy,
        score=0.5,
        preds_test=pd.Series([0, 1]),
        preds_train=pd.Series([0, 1]),
        probs_test=None,
        probs_train=None,
        device="cuda",
        backend="sklearn-dummy",
        model_version="",
        tune_seconds=1.25,
        eval_seconds=0.75,
        total_seconds=2.0,
    )

    row = result.to_row().iloc[0]
    preds = result.to_preds()
    encoded = json.loads(result.to_preds_json())

    assert row["device"] == "cuda"
    assert row["backend"] == "sklearn-dummy"
    assert row["model_version"] == ""
    assert row["n_selected_features"] == 1
    assert row["tune_seconds"] == 1.25
    assert preds.device == "cuda"
    assert preds.backend == "sklearn-dummy"
    assert preds.model_version == ""
    assert encoded["total_seconds"] == 2.0
    assert encoded["model_version"] == ""
    assert encoded["n_selected_features"] == 1


@pytest.mark.fast
def test_device_result_fields_report_actual_cpu_backend_device() -> None:
    model = DummyClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    fields = _device_result_fields(model, {})

    assert fields["device"] == "cpu"
    assert fields["requested_device"] == "auto"
    assert fields["resolved_device"] == "cuda"
    assert fields["torch_device"] == "cuda"
    assert fields["accelerated"] is False
    assert fields["backend"] == "dummy-cpu"


@pytest.mark.fast
def test_device_result_fields_report_torch_cpu_fallback_device() -> None:
    model = MLPEstimator(num_classes=2)
    info = device_info_from_resolved(
        "auto",
        "cuda",
        torch_cuda_available=False,
        cuda_driver_available=True,
    )
    model.set_device_info(info)

    fields = _device_result_fields(model, {})

    assert fields["device"] == "cpu"
    assert fields["requested_device"] == "auto"
    assert fields["resolved_device"] == "cuda"
    assert fields["torch_device"] == "cpu"
    assert fields["accelerated"] is False
    assert fields["backend"] == "skorch-cpu"


@pytest.mark.fast
def test_htune_result_equality_accepts_numpy_predictions() -> None:
    model = DummyClassifier()
    kwargs = dict(
        selection="none",
        selected_cols=["x"],
        embed_select_model=None,
        model_cls=DummyClassifier,
        model=model,
        params={},
        metric=ClassifierScorer.Accuracy,
        score=0.5,
        preds_test=pd.Series([0, 1]),
        preds_train=pd.Series([0, 1]),
        probs_test=None,
        probs_train=None,
    )
    left = HtuneResult(**kwargs)
    right = HtuneResult(
        **{
            **kwargs,
            "preds_test": pd.Series([0, 1]).to_numpy(),
            "preds_train": pd.Series([0, 1]).to_numpy(),
        }
    )

    assert left == right


@pytest.mark.fast
def test_htune_eval_internal_folds_use_positional_indices() -> None:
    class PositionalModel(DfAnalyzeModel):
        def __init__(self) -> None:
            super().__init__()
            self.is_classifier = False
            self.tuned_model = object()
            self.tuned_args = {}

        def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict]:
            return object, {}

        def optuna_args(self, trial: Any) -> dict[str, str | float | int]:
            return {}

        def refit_tuned(self, *args: Any, **kwargs: Any) -> None:
            self.tuned_model = object()

        def tuned_predict(self, X: pd.DataFrame) -> pd.Series:
            return pd.Series([0.0] * len(X), index=X.index)

    X_train = pd.DataFrame({"x": range(12)}, index=range(100, 112))
    y_train = pd.Series(range(12), index=range(100, 112), name="target")
    X_test = pd.DataFrame({"x": range(15)}, index=range(200, 215))
    y_test = pd.Series(range(15), index=range(200, 215), name="target")
    model = PositionalModel()

    df, *_ = model.htune_eval(
        X_train=X_train,
        y_train=y_train,
        g_train=None,
        X_test=X_test,
        y_test=y_test,
        g_test=None,
        seed=1,
    )

    assert "holdout" in df.columns


@pytest.mark.fast
def test_mlp_set_device_info_updates_skorch_device() -> None:
    model = MLPEstimator(num_classes=2)
    cuda_info = device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True)

    model.set_device_info(cuda_info)

    assert model.device_info.resolved == "cuda"
    assert model.fixed_args["device"] == "cuda"


@pytest.mark.fast
def test_gandalf_set_device_info_stores_lightning_device() -> None:
    model = GandalfEstimator(num_classes=2)
    mps_info = device_info_from_resolved("auto", "mps", mps_is_available=True)

    model.set_device_info(mps_info)

    assert model.device_info.resolved == "mps"
    assert model.device_info.lightning_accelerator == "mps"
    assert model.device_info.lightning_devices == 1


@pytest.mark.fast
def test_lgbm_classifier_stays_cpu_with_cuda() -> None:
    model = LightGBMClassifier()
    cuda_info = device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True)

    model.set_device_info(cuda_info)

    assert model.device_info.resolved == "cuda"
    assert "device_type" not in model.fixed_args
    assert model.fixed_args["n_jobs"] == -1
    assert model.backend_name() == "lightgbm-cpu"


@pytest.mark.fast
def test_lgbm_regressor_cpu_clears_device_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_backends, "lightgbm_best_device_type", lambda boosting_type="gbdt": ("cuda", "")
    )
    model = LightGBMRegressor()
    cuda_info = device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True)
    cpu_info = device_info_from_resolved("cpu", "cpu")

    model.set_device_info(cuda_info)
    model.set_device_info(cpu_info)

    assert model.device_info.resolved == "cpu"
    assert "device_type" not in model.fixed_args
    assert model.backend_name() == "lightgbm-cpu"


@pytest.mark.fast
def test_lgbm_rf_classifier_preserves_rf_boosting_with_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_boosting_types: list[str] = []

    def fake_best_device_type(boosting_type: str = "gbdt") -> tuple[str, str]:
        checked_boosting_types.append(boosting_type)
        return "cuda", ""

    monkeypatch.setattr(gpu_backends, "lightgbm_best_device_type", fake_best_device_type)
    model = LightGBMRFClassifier()
    cuda_info = device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True)

    model.set_device_info(cuda_info)

    assert model.fixed_args["boosting_type"] == "rf"
    assert "device_type" not in model.fixed_args
    assert model.fixed_args["n_jobs"] == -1
    assert model.backend_name() == "lightgbm-cpu"
    assert checked_boosting_types == []


@pytest.mark.fast
def test_lgbm_mps_stays_cpu() -> None:
    model = LightGBMClassifier()
    mps_info = device_info_from_resolved("auto", "mps", mps_is_available=True)

    model.set_device_info(mps_info)

    assert model.device_info.resolved == "mps"
    assert model.fixed_args["n_jobs"] == -1
    assert "device_type" not in model.fixed_args
    assert model.backend_name() == "lightgbm-cpu"


@pytest.mark.fast
def test_mlp_gpu_tuning_forces_single_job(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = capture_base_tuning_n_jobs(monkeypatch)
    X, y = tiny_tuning_data()
    model = MLPEstimator(num_classes=2)
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))
    assert model.backend_name() == "skorch-cuda"

    model.htune_optuna(
        X_train=X,
        y_train=y,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_jobs=-1,
    )

    assert calls == [1]


@pytest.mark.fast
def test_torch_models_use_cpu_when_cuda_backend_is_not_torch_cuda() -> None:
    backend_cuda = device_info_from_resolved(
        "auto",
        "cuda",
        torch_cuda_available=False,
        cuda_driver_available=True,
    )

    mlp = MLPEstimator(num_classes=2)
    mlp.set_device_info(backend_cuda)
    assert mlp.fixed_args["device"] == "cpu"
    assert mlp.backend_name() == "skorch-cpu"
    assert "iterator_train__pin_memory" not in mlp.fixed_args

    gandalf = GandalfEstimator(num_classes=2)
    gandalf.set_device_info(backend_cuda)
    assert gandalf.backend_name() == "lightning-cpu"


@pytest.mark.fast
def test_torch_cuda_models_use_cuda_tuning_trial_budgets() -> None:
    cuda_info = device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True)

    mlp = MLPEstimator(num_classes=2)
    mlp.set_device_info(cuda_info)
    assert mlp.effective_tuning_n_trials(100) == 75
    assert mlp.effective_tuning_n_trials(50) == 50

    gandalf = GandalfEstimator(num_classes=2)
    gandalf.set_device_info(cuda_info)
    assert gandalf.effective_tuning_n_trials(100) == 50
    assert gandalf.effective_tuning_n_trials(40) == 40

    mps = MLPEstimator(num_classes=2)
    mps.set_device_info(device_info_from_resolved("auto", "mps", mps_is_available=True))
    assert mps.effective_tuning_n_trials(100) == 100


@pytest.mark.fast
def test_gandalf_gpu_tuning_forces_single_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = capture_base_tuning_n_jobs(monkeypatch)
    X, y = tiny_tuning_data()
    model = GandalfEstimator(num_classes=2)
    model.set_device_info(device_info_from_resolved("auto", "mps", mps_is_available=True))
    assert model.backend_name() == "lightning-mps"

    model.htune_optuna(
        X_train=X,
        y_train=y,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_jobs=-1,
    )

    assert calls == [1]


@pytest.mark.fast
def test_gandalf_cpu_tuning_forces_single_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = capture_base_tuning_n_jobs(monkeypatch)
    X, y = tiny_tuning_data()
    model = GandalfEstimator(num_classes=2)
    model.set_device_info(device_info_from_resolved("cpu", "cpu"))

    model.htune_optuna(
        X_train=X,
        y_train=y,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_jobs=-1,
    )

    assert calls == [1]


@pytest.mark.fast
def test_gandalf_marks_memory_and_cuda_errors_fatal() -> None:
    from df_analyze.models import gandalf as gandalf_module

    try:
        raise RuntimeError("training failed") from RuntimeError("CUDA error")
    except RuntimeError as wrapped:
        assert is_fatal_accelerator_error(wrapped)

    assert gandalf_module._is_fatal_accelerator_error(MemoryError())
    assert gandalf_module._is_fatal_accelerator_error(
        RuntimeError("CUDA error: invalid device ordinal")
    )
    assert gandalf_module._is_fatal_accelerator_error(
        RuntimeError("CUDA out of memory")
    )
    assert not gandalf_module._is_fatal_accelerator_error(RuntimeError("bad data"))


@pytest.mark.fast
def test_evaluate_tuned_reraises_fatal_accelerator_errors() -> None:
    X, y = tiny_tuning_data()
    prep = SimpleNamespace(X=X, y=y, groups=None, X_cont=None, X_cat=None)
    options = SimpleNamespace(
        feat_select=(),
        htune_cls_metric=ClassifierScorer.Accuracy,
        htune_reg_metric=ClassifierScorer.Accuracy,
        htune_trials=1,
        models=[FatalAcceleratorModel],
        device_info=device_info_from_resolved(
            "auto",
            "cuda",
            torch_cuda_available=True,
            cuda_driver_available=True,
        ),
        tabpfn_version="v3",
        tabpfn_tune_folds=1,
        seed=42,
    )

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        evaluate_tuned(
            prepared=SimpleNamespace(is_classification=True, num_classes=2),
            prep_train=prep,
            prep_test=prep,
            assoc_filtered=SimpleNamespace(selected=[]),
            pred_filtered=None,
            model_selected=SimpleNamespace(embed_selected=None, wrap_selected=None),
            options=options,
        )


@pytest.mark.fast
@pytest.mark.parametrize("model_cls", [LightGBMClassifier, LightGBMRFClassifier])
def test_lgbm_cpu_tuning_uses_serial_optuna_with_all_core_estimator(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: Any,
) -> None:
    monkeypatch.setattr(
        gpu_backends, "lightgbm_best_device_type", lambda boosting_type="gbdt": ("cuda", "")
    )
    calls = capture_base_tuning_n_jobs(monkeypatch)
    X, y = tiny_tuning_data()
    model = model_cls()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))
    assert model.fixed_args["n_jobs"] == -1
    assert "device_type" not in model.fixed_args
    assert model.backend_name() == "lightgbm-cpu"

    model.htune_optuna(
        X_train=X,
        y_train=y,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_jobs=-1,
    )

    assert calls == [1]


@pytest.mark.fast
@pytest.mark.parametrize(
    ("model_cls", "helper_names"),
    [
        (LRClassifier, ("get_cuml_logistic_regression_cls",)),
        (ElasticNetRegressor, ("get_cuml_elasticnet_cls",)),
        (SGDRegressor, ("get_cuml_mbsgd_regressor_cls",)),
        (SGDClassifierSelector, ("get_cuml_mbsgd_classifier_cls",)),
        (SGDRegressorSelector, ("get_cuml_mbsgd_regressor_cls",)),
    ],
)
def test_cuml_traditional_tuning_forces_single_job(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: Any,
    helper_names: tuple[str, ...],
) -> None:
    calls = capture_base_tuning_n_jobs(monkeypatch)
    for helper_name in helper_names:
        monkeypatch.setattr(gpu_backends, helper_name, lambda: object)
    monkeypatch.setattr(gpu_backends, "cuml_logistic_regression_available", lambda: True)
    monkeypatch.setattr(gpu_backends, "cuml_elasticnet_available", lambda: True)
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_regressor_available", lambda: True)
    monkeypatch.setattr(gpu_backends, "cuml_mbsgd_classifier_available", lambda: True)
    X, y = tiny_tuning_data()
    model = model_cls()
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
@pytest.mark.parametrize(
    ("wrapper_model", "is_classification"),
    [
        (WrapperSelectionModel.LGBM, True),
        (WrapperSelectionModel.KNN, True),
    ],
)
def test_cuda_stepwise_gpu_serial_wrappers_use_single_job(
    wrapper_model: WrapperSelectionModel,
    is_classification: bool,
) -> None:
    assert stepwise_n_jobs(wrapper_model, "cuda", is_classification) == 1


@pytest.mark.fast
@pytest.mark.parametrize("is_classification", [True, False])
def test_cuda_stepwise_linear_wrapper_uses_cpu_parallel_policy(
    is_classification: bool,
) -> None:
    assert stepwise_n_jobs(WrapperSelectionModel.Linear, "cuda", is_classification) == -1


@pytest.mark.fast
def test_cuda_stepwise_linear_score_device_is_cpu() -> None:
    selector = object.__new__(StepwiseSelector)
    cuda_info = device_info_from_resolved(
        "auto",
        "cuda",
        torch_cuda_available=True,
        cuda_driver_available=True,
    )
    selector.options = SimpleNamespace(
        wrapper_model=WrapperSelectionModel.Linear,
        device_requested="auto",
        device="cuda",
        device_info=cuda_info,
    )
    selector.prepared = SimpleNamespace(is_classification=True)

    score_info = selector._score_device_info()

    assert score_info.requested == "auto"
    assert score_info.resolved == "cpu"
    assert score_info.torch_device == "cpu"
    assert score_info.torch_cuda_available is True


@pytest.mark.fast
def test_stepwise_parallel_jobs_keep_cpu_parallel() -> None:
    assert stepwise_n_jobs(WrapperSelectionModel.KNN, "cpu", True) == -1
    assert stepwise_n_jobs(WrapperSelectionModel.Linear, "cpu", True) == -1


@pytest.mark.fast
def test_stepwise_score_passes_device_info_to_model() -> None:
    class RecordingModel(DfAnalyzeModel):
        seen_device = None

        def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict]:
            return object, {}

        def optuna_args(self, trial: Any) -> dict[str, str | float | int]:
            return {}

        def cv_score(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            groups: pd.Series | None,
            metric: ClassifierScorer,
            test: bool = False,
        ) -> float:
            RecordingModel.seen_device = self.device_info
            return 1.0

    X, y = tiny_tuning_data()
    cuda_info = device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True)

    score = get_dfanalyze_score(
        model_cls=RecordingModel,
        X=X,
        y=y,
        g=None,
        metric=ClassifierScorer.Accuracy,
        selected={"a"},
        candidate="b",
        is_forward=True,
        test=False,
        device_info=cuda_info,
    )

    assert score == 1.0
    assert RecordingModel.seen_device == cuda_info


@pytest.mark.fast
def test_cv_score_cleans_accelerator_after_each_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from df_analyze.models import base as base_module

    calls: list[str] = []

    class ConstantEstimator:
        def fit(self, X: pd.DataFrame, y: pd.Series) -> ConstantEstimator:
            return self

        def predict(self, X: pd.DataFrame) -> list[int]:
            return [0] * len(X)

    class CleanupModel(DfAnalyzeModel):
        def __init__(self) -> None:
            super().__init__()
            self.is_classifier = True

        def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict]:
            return ConstantEstimator, {}

        def optuna_args(self, trial: Any) -> dict[str, str | float | int]:
            return {}

    def fake_cleanup(info: Any) -> None:
        calls.append(info.resolved)

    monkeypatch.setattr(base_module, "cleanup_accelerator_memory", fake_cleanup)
    X = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    y = pd.Series([0, 1] * 10, name="target")
    model = CleanupModel()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    score = model.cv_score(
        X=X,
        y=y,
        groups=None,
        metric=ClassifierScorer.Accuracy,
    )

    assert isinstance(score, float)
    assert calls == ["cuda"] * 5
    assert model.model is None


@pytest.mark.fast
def test_cleanup_accelerator_memory_clears_mps_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_empty_cache() -> None:
        calls.append("mps")

    monkeypatch.setattr(
        device.torch,
        "mps",
        SimpleNamespace(empty_cache=fake_empty_cache),
        raising=False,
    )
    info = device_info_from_resolved("mps", "mps", mps_is_available=True)

    device.cleanup_accelerator_memory(info)

    assert calls == ["mps"]
