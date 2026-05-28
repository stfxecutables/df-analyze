from __future__ import annotations

import pytest

from df_analyze.device import device_info_from_resolved
from df_analyze.models import gpu as gpu_backends
from df_analyze.models import lightgbm_cuda


@pytest.mark.fast
def test_lightgbm_cuda_check_disabled_is_noop() -> None:
    result = lightgbm_cuda.check_lightgbm_cuda_backend(
        device_info_from_resolved("cpu", "cpu"),
        enabled=False,
    )

    assert result.checked is False
    assert result.available is False
    assert result.reason == "disabled"


@pytest.mark.fast
def test_lightgbm_cuda_check_passes_when_backend_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_backends, "lightgbm_cuda_availability", lambda: (True, "")
    )
    monkeypatch.setattr(
        gpu_backends, "lightgbm_cuda_rf_availability", lambda: (True, "")
    )

    result = lightgbm_cuda.check_lightgbm_cuda_backend(
        device_info_from_resolved("cuda", "cuda", torch_cuda_available=True, cuda_driver_available=True),
        enabled=True,
    )

    assert result.checked is True
    assert result.available is True
    assert result.reason == "available"


@pytest.mark.fast
def test_lightgbm_cuda_check_only_checks_selected_gbdt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_backends, "lightgbm_cuda_availability", lambda: (True, "")
    )
    monkeypatch.setattr(
        gpu_backends,
        "lightgbm_cuda_rf_availability",
        lambda: pytest.fail("RF LightGBM CUDA should not be checked"),
    )

    result = lightgbm_cuda.check_lightgbm_cuda_backend(
        device_info_from_resolved("cuda", "cuda", torch_cuda_available=True, cuda_driver_available=True),
        enabled=True,
        boosting_types=("gbdt",),
    )

    assert result.checked is True
    assert result.available is True
    assert result.reason == "available"


@pytest.mark.fast
def test_lightgbm_cuda_check_only_checks_selected_rf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_backends,
        "lightgbm_cuda_availability",
        lambda: pytest.fail("ordinary LightGBM CUDA should not be checked"),
    )
    monkeypatch.setattr(
        gpu_backends, "lightgbm_cuda_rf_availability", lambda: (True, "")
    )

    result = lightgbm_cuda.check_lightgbm_cuda_backend(
        device_info_from_resolved("cuda", "cuda", torch_cuda_available=True, cuda_driver_available=True),
        enabled=True,
        boosting_types=("rf",),
    )

    assert result.checked is True
    assert result.available is True
    assert result.reason == "available"


@pytest.mark.fast
def test_lightgbm_cuda_check_raises_when_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_backends, "lightgbm_cuda_availability", lambda: (False, "missing cuda")
    )
    monkeypatch.setattr(
        gpu_backends, "lightgbm_cuda_rf_availability", lambda: (False, "missing cuda")
    )

    with pytest.raises(RuntimeError, match="Install a CUDA-enabled LightGBM build"):
        lightgbm_cuda.check_lightgbm_cuda_backend(
            device_info_from_resolved("cuda", "cuda", torch_cuda_available=True, cuda_driver_available=True),
            enabled=True,
        )


@pytest.mark.fast
def test_lightgbm_cuda_check_warns_when_backend_unavailable_but_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_backends, "lightgbm_cuda_availability", lambda: (False, "missing cuda")
    )
    monkeypatch.setattr(
        gpu_backends, "lightgbm_cuda_rf_availability", lambda: (False, "missing cuda")
    )

    with pytest.warns(RuntimeWarning, match="CPU fallback"):
        result = lightgbm_cuda.check_lightgbm_cuda_backend(
            device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True),
            enabled=True,
            required=False,
        )

    assert result.checked is True
    assert result.available is False
    assert "missing cuda" in result.reason


@pytest.mark.fast
def test_lightgbm_cuda_check_warns_and_skips_non_cuda_device() -> None:
    with pytest.warns(RuntimeWarning, match="resolved device is 'cpu'"):
        result = lightgbm_cuda.check_lightgbm_cuda_backend(
            device_info_from_resolved("cpu", "cpu"),
            enabled=True,
        )

    assert result.checked is False
    assert result.available is False
    assert result.reason


@pytest.mark.fast
def test_lightgbm_cuda_legacy_ensure_alias_checks_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_backends, "lightgbm_cuda_availability", lambda: (True, "")
    )
    monkeypatch.setattr(
        gpu_backends, "lightgbm_cuda_rf_availability", lambda: (True, "")
    )

    result = lightgbm_cuda.ensure_lightgbm_cuda_backend(
        device_info_from_resolved("cuda", "cuda", torch_cuda_available=True, cuda_driver_available=True),
        enabled=True,
    )

    assert result.checked is True
    assert result.available is True


@pytest.mark.fast
def test_lightgbm_cuda_result_report() -> None:
    result = lightgbm_cuda.LightGBMCudaCheckResult(
        checked=True,
        available=True,
        reason="available",
    )

    assert lightgbm_cuda.result_as_report(result) == {
        "checked": True,
        "available": True,
        "reason": "available",
    }
