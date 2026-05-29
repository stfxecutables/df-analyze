from __future__ import annotations

import pytest

import df_analyze.gpu_setup as gpu_setup
from df_analyze.models import gpu as gpu_backends
from df_analyze.gpu_setup import (
    CUDA13_GROUP,
    GpuPreflightResult,
    choose_gpu_setup_plan,
    cuda_major_from_text,
    expand_preflight_models,
    gpu_preflight_results,
    run_gpu_preflight,
)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CUDA Version: 13.2", 13),
        ("Cuda compilation tools, release 12.8, V12.8.93", 12),
        ("no cuda here", None),
    ],
)
def test_cuda_major_from_text(text: str, expected: int | None) -> None:
    assert cuda_major_from_text(text) == expected


@pytest.mark.fast
def test_choose_gpu_setup_plan_uses_locked_cuda13_group() -> None:
    plan = choose_gpu_setup_plan(
        sys_platform="linux",
        machine="x86_64",
        cuda_major=13,
        uv="uv",
        python="python",
    )

    assert plan.supported is True
    assert plan.locked is True
    assert plan.group == CUDA13_GROUP
    assert plan.commands == (("uv", "sync", "--group", CUDA13_GROUP),)
    assert plan.run_args == ("uv", "run", "--group", CUDA13_GROUP)


@pytest.mark.fast
def test_choose_gpu_setup_plan_rejects_native_windows() -> None:
    plan = choose_gpu_setup_plan(
        sys_platform="win32",
        machine="AMD64",
        cuda_major=13,
        uv="uv",
        python="python",
    )

    assert plan.supported is False
    assert "Windows" in plan.reason
    assert "WSL2" in plan.reason


@pytest.mark.fast
def test_choose_gpu_setup_plan_rejects_cuda12_by_default() -> None:
    plan = choose_gpu_setup_plan(
        sys_platform="linux",
        machine="x86_64",
        cuda_major=12,
        uv="uv",
        python="python",
    )

    assert plan.supported is False
    assert "CUDA 12" in plan.reason
    assert "--allow-inexact-cu12" in plan.reason


@pytest.mark.fast
def test_choose_gpu_setup_plan_allows_inexact_cuda12() -> None:
    plan = choose_gpu_setup_plan(
        sys_platform="linux",
        machine="x86_64",
        cuda_major=12,
        uv="uv",
        python="/venv/bin/python",
        allow_inexact_cu12=True,
    )

    assert plan.supported is True
    assert plan.locked is False
    assert plan.commands[0] == ("uv", "sync", "--inexact")
    assert plan.commands[1][:6] == (
        "uv",
        "pip",
        "install",
        "--python",
        "/venv/bin/python",
        "--index",
    )
    assert "cuml-cu12==26.4.*" in plan.commands[1]
    assert plan.run_args == ("uv", "run", "--no-sync")


@pytest.mark.fast
def test_expand_preflight_models_defaults_to_full_cuda_scope() -> None:
    models = expand_preflight_models(())

    assert models == (
        "torch",
        "cuml-knn",
        "cuml-lr",
        "cuml-elastic",
        "cuml-sgd",
        "xgboost",
        "catboost",
        "mlp",
        "gandalf",
        "tabpfn",
        "lightgbm-cpu",
    )


@pytest.mark.fast
def test_expand_preflight_models_accepts_groups_and_aliases() -> None:
    models = expand_preflight_models(("core", "xgb", "lr", "rf", "dummy"))

    assert models == (
        "torch",
        "cuml-knn",
        "cuml-lr",
        "xgboost",
        "lightgbm-cpu",
        "dummy-cpu",
    )


@pytest.mark.fast
def test_expand_preflight_models_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown preflight model"):
        expand_preflight_models(("not-a-model",))


@pytest.mark.fast
def test_gpu_preflight_results_runs_selected_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def result(name: str) -> GpuPreflightResult:
        calls.append(name)
        return GpuPreflightResult(name=name, status="pass", detail="")

    monkeypatch.setitem(gpu_setup.PREFLIGHT_CHECKS, "torch", lambda: result("torch"))
    monkeypatch.setitem(
        gpu_setup.PREFLIGHT_CHECKS, "xgboost", lambda: result("xgboost")
    )

    results = gpu_preflight_results(("torch", "xgboost"))

    assert calls == ["torch", "xgboost"]
    assert [item.name for item in results] == ["torch", "xgboost"]


@pytest.mark.fast
def test_tabpfn_preflight_does_not_require_v25_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_probe() -> tuple[bool, str]:
        raise AssertionError("unexpected v2.5 probe")

    monkeypatch.setattr(gpu_backends, "tabpfn_api_status", lambda: (True, ""))
    monkeypatch.setattr(gpu_backends, "tabpfn_v25_api_status", fail_probe)

    result = gpu_setup.PREFLIGHT_CHECKS["tabpfn"]()

    assert result.status == "pass"


@pytest.mark.fast
def test_run_gpu_preflight_returns_failure_for_failed_backend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(
        gpu_setup.PREFLIGHT_CHECKS,
        "torch",
        lambda: GpuPreflightResult(
            name="torch",
            status="fail",
            detail="CUDA unavailable",
        ),
    )

    code = run_gpu_preflight(("torch",))

    captured = capsys.readouterr()
    assert code == 1
    assert "FAIL torch - CUDA unavailable" in captured.out
