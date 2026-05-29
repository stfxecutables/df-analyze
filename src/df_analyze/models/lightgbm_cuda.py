from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
from warnings import warn

from df_analyze.device import DeviceInfo
from df_analyze.models import gpu as gpu_backends

_LIGHTGBM_BOOSTING_TYPES = ("gbdt", "rf")


@dataclass(frozen=True)
class LightGBMCudaCheckResult:
    checked: bool
    available: bool
    reason: str = ""


def clear_lightgbm_backend_cache() -> None:
    cache_names = (
        "_lightgbm_device_type_availability",
        "lightgbm_cuda_available",
        "lightgbm_cuda_rf_available",
    )
    for name in cache_names:
        cache_clear = getattr(getattr(gpu_backends, name, None), "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


def _normalize_lightgbm_boosting_types(
    boosting_types: Iterable[str] | None,
) -> tuple[str, ...]:
    requested = (
        _LIGHTGBM_BOOSTING_TYPES if boosting_types is None else tuple(boosting_types)
    )
    normalized: list[str] = []
    for boosting_type in requested:
        mode = str(boosting_type).strip().lower()
        if mode == "lgbm":
            mode = "gbdt"
        if mode not in _LIGHTGBM_BOOSTING_TYPES:
            expected = ", ".join(_LIGHTGBM_BOOSTING_TYPES)
            raise ValueError(
                f"Invalid LightGBM boosting type {boosting_type!r}. "
                f"Expected one of: {expected}."
            )
        if mode not in normalized:
            normalized.append(mode)
    return tuple(normalized)


def _lightgbm_cuda_available_for_boosting_types(
    boosting_types: Iterable[str] | None = None,
) -> tuple[bool, str]:
    checks = {
        "gbdt": ("lgbm", gpu_backends.lightgbm_cuda_availability),
        "rf": ("rf", gpu_backends.lightgbm_cuda_rf_availability),
    }
    reasons: list[str] = []
    for boosting_type in _normalize_lightgbm_boosting_types(boosting_types):
        label, check = checks[boosting_type]
        available, reason = check()
        if not available and reason:
            reasons.append(f"{label}: {reason}")
        elif not available:
            reasons.append(f"{label}: unavailable")
    if not reasons:
        return True, ""
    return False, "; ".join(reasons)


def _lightgbm_cuda_available_for_lgbm_and_rf() -> tuple[bool, str]:
    return _lightgbm_cuda_available_for_boosting_types(_LIGHTGBM_BOOSTING_TYPES)


def check_lightgbm_cuda_backend(
    device_info: DeviceInfo,
    *,
    enabled: bool = False,
    required: bool = True,
    boosting_types: Iterable[str] | None = None,
) -> LightGBMCudaCheckResult:
    if not enabled:
        return LightGBMCudaCheckResult(
            checked=False,
            available=False,
            reason="disabled",
        )

    if not device_info.cuda_backends_enabled:
        reason = (
            "LightGBM CUDA was requested, but the resolved device is "
            f"{device_info.resolved!r} with CUDA backends disabled; leaving "
            "LightGBM on its normal CPU path."
        )
        warn(reason, RuntimeWarning, stacklevel=2)
        return LightGBMCudaCheckResult(
            checked=False,
            available=False,
            reason=reason,
        )

    clear_lightgbm_backend_cache()
    available, reason = _lightgbm_cuda_available_for_boosting_types(boosting_types)
    if available:
        return LightGBMCudaCheckResult(
            checked=True,
            available=True,
            reason="available",
        )

    detail = reason or "LightGBM CUDA smoke checks failed."
    if not required:
        warn(
            "LightGBM CUDA is not available in the current environment; "
            "LightGBM models will use their normal CPU fallback. "
            f"Backend reason: {detail}",
            RuntimeWarning,
            stacklevel=2,
        )
        return LightGBMCudaCheckResult(
            checked=True,
            available=False,
            reason=detail,
        )

    raise RuntimeError(
        "LightGBM CUDA was requested, but the installed LightGBM package does "
        "not pass df-analyze's CUDA smoke checks. Install a CUDA-enabled "
        "LightGBM build in this environment before running df-analyze with "
        f"LightGBM on CUDA. Backend reason: {detail}"
    )


def ensure_lightgbm_cuda_backend(
    device_info: DeviceInfo,
    *,
    enabled: bool = False,
    required: bool = True,
    boosting_types: Iterable[str] | None = None,
) -> LightGBMCudaCheckResult:
    return check_lightgbm_cuda_backend(
        device_info,
        enabled=enabled,
        required=required,
        boosting_types=boosting_types,
    )


def result_as_report(result: LightGBMCudaCheckResult) -> Mapping[str, object]:
    return {
        "checked": result.checked,
        "available": result.available,
        "reason": result.reason,
    }
