from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from tempfile import TemporaryDirectory

import pytest
from cli_test_helpers import ArgvContext

import df_analyze.device as device
from df_analyze.cli.cli import get_options
from df_analyze.enumerables import DfAnalyzeClassifier, DfAnalyzeRegressor
from df_analyze.models import cuml_cuda
from df_analyze.models import lightgbm_cuda


def _get_options_for(*args: str):
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "data.parquet"
        path.touch()
        with ArgvContext(
            "df-analyze.py",
            "--df",
            f"{path}",
            "--verbosity",
            "0",
            *args,
        ):
            return get_options()


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_tabpfn_version_cli_defaults_to_auto_and_accepts_v25() -> None:
    default_opts = _get_options_for("--classifiers", "tabpfn")
    explicit_opts = _get_options_for(
        "--classifiers",
        "tabpfn",
        "--tabpfn-version",
        "v2_5",
    )

    assert default_opts.tabpfn_version == "auto"
    assert explicit_opts.tabpfn_version == "v2_5"
    assert default_opts.tabpfn_tune_folds == 1


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_tabpfn_tune_folds_cli_defaults_to_one_and_accepts_five() -> None:
    explicit_opts = _get_options_for(
        "--classifiers",
        "tabpfn",
        "--tabpfn-tune-folds",
        "5",
    )

    assert explicit_opts.tabpfn_tune_folds == 5


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_classifier_svm_disabled(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
) -> None:
    if device_name == "mps":
        monkeypatch.setattr(device, "mps_available", lambda: True)
    if device_name == "cuda":
        monkeypatch.setattr(device, "cuda_available", lambda: True)

    with pytest.warns(UserWarning, match="SVMs are disabled"):
        opts = _get_options_for(
            "--device",
            device_name,
            "--classifiers",
            "svm",
        )

    assert opts.device == device_name
    assert DfAnalyzeClassifier.SVM not in opts.classifiers


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_regressor_svm_disabled(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
) -> None:
    if device_name == "mps":
        monkeypatch.setattr(device, "mps_available", lambda: True)
    if device_name == "cuda":
        monkeypatch.setattr(device, "cuda_available", lambda: True)

    with pytest.warns(UserWarning, match="SVMs are disabled"):
        opts = _get_options_for(
            "--device",
            device_name,
            "--mode",
            "regress",
            "--regressors",
            "svm",
        )

    assert opts.device == device_name
    assert DfAnalyzeRegressor.SVM not in opts.regressors


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_lightgbm_cuda_check_does_not_run_for_explicit_cuda_lgbm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(
        lightgbm_cuda,
        "check_lightgbm_cuda_backend",
        lambda device_info, **kwargs: pytest.fail("check should not run"),
    )

    _get_options_for(
        "--device",
        "cuda",
        "--classifiers",
        "lgbm",
    )


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_cuml_cuda_check_runs_for_auto_resolved_cuda_knn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...], dict[str, object]]] = []
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(
        cuml_cuda,
        "check_cuml_cuda_backends",
        lambda device_info, backends, **kwargs: calls.append(
            (device_info.requested, tuple(backends), kwargs)
        ),
    )

    _get_options_for(
        "--classifiers",
        "knn",
    )

    assert calls == [
        (
            "auto",
            ("knn-classifier",),
            {"enabled": True, "auto_install": True},
        )
    ]


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_cuml_cuda_check_runs_for_explicit_cuda_regressor_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...], dict[str, object]]] = []
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(
        cuml_cuda,
        "check_cuml_cuda_backends",
        lambda device_info, backends, **kwargs: calls.append(
            (device_info.requested, tuple(backends), kwargs)
        ),
    )

    _get_options_for(
        "--device",
        "cuda",
        "--mode",
        "regress",
        "--regressors",
        "knn",
        "elastic",
        "sgd",
    )

    assert calls == [
        (
            "cuda",
            ("knn-regressor", "elasticnet", "mbsgd-regressor"),
            {"enabled": True, "auto_install": True},
        )
    ]


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_cuml_cuda_check_does_not_run_for_main_sgd_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(
        cuml_cuda,
        "check_cuml_cuda_backends",
        lambda device_info, backends, **kwargs: pytest.fail("check should not run"),
    )

    _get_options_for(
        "--device",
        "cuda",
        "--classifiers",
        "sgd",
        "--feat-select",
        "none",
    )


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_cuml_cuda_check_runs_for_embedded_linear_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...], dict[str, object]]] = []
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(
        cuml_cuda,
        "check_cuml_cuda_backends",
        lambda device_info, backends, **kwargs: calls.append(
            (device_info.requested, tuple(backends), kwargs)
        ),
    )

    _get_options_for(
        "--device",
        "cuda",
        "--classifiers",
        "dummy",
        "--feat-select",
        "embed",
        "--embed-select",
        "linear",
    )

    assert calls == [
        (
            "cuda",
            ("mbsgd-classifier",),
            {"enabled": True, "auto_install": True},
        )
    ]


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_cuml_cuda_check_does_not_run_for_wrapper_linear_regressor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...], dict[str, object]]] = []
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(
        cuml_cuda,
        "check_cuml_cuda_backends",
        lambda device_info, backends, **kwargs: calls.append(
            (device_info.requested, tuple(backends), kwargs)
        ),
    )

    _get_options_for(
        "--device",
        "cuda",
        "--mode",
        "regress",
        "--regressors",
        "dummy",
        "--feat-select",
        "wrap",
        "--wrapper-select",
        "step-up",
        "--wrapper-model",
        "linear",
    )

    assert calls == []


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
@pytest.mark.parametrize(
    "args",
    [
        ("--classifiers", "lgbm"),
        (
            "--device",
            "cuda",
            "--classifiers",
            "dummy",
            "--feat-select",
            "embed",
            "--embed-select",
            "lgbm",
        ),
        ("--device", "cuda", "--classifiers", "rf"),
        ("--device", "cuda", "--mode", "regress", "--regressors", "rf"),
        ("--device", "cuda", "--classifiers", "lgbm", "rf"),
    ],
)
def test_lightgbm_cuda_check_does_not_run_for_lgbm_or_rf(
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(
        lightgbm_cuda,
        "check_lightgbm_cuda_backend",
        lambda device_info, **kwargs: pytest.fail("check should not run"),
    )

    _get_options_for(*args)


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_lightgbm_cuda_check_does_not_run_when_lightgbm_not_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    monkeypatch.setattr(
        lightgbm_cuda,
        "check_lightgbm_cuda_backend",
        lambda device_info, **kwargs: pytest.fail("check should not run"),
    )

    _get_options_for(
        "--device",
        "cuda",
        "--classifiers",
        "dummy",
    )
