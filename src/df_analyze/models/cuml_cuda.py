from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Sequence
from warnings import warn

from df_analyze.device import DeviceInfo
from df_analyze.models import gpu as gpu_backends

AUTO_INSTALL_ENV_VAR = "DF_ANALYZE_AUTO_INSTALL_CUML"
AUTO_INSTALL_ATTEMPTED_ENV_VAR = "DF_ANALYZE_CUML_AUTO_INSTALL_ATTEMPTED"
CUDA_SUFFIX_ENV_VAR = "DF_ANALYZE_CUML_CUDA_SUFFIX"
NVIDIA_PYPI_INDEX = "https://pypi.nvidia.com"

_FALSE_ENV_VALUES = {"0", "false", "no", "n", "off"}
_TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class CumlBackendStatus:
    name: str
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class CumlCudaCheckResult:
    checked: bool
    available: bool
    reason: str = ""
    installed: bool = False


@dataclass(frozen=True)
class CumlCudaInstallResult:
    attempted: bool
    succeeded: bool
    package: str = ""
    command: tuple[str, ...] = ()
    reason: str = ""


def env_flag_disabled(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in _FALSE_ENV_VALUES


def env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in _TRUE_ENV_VALUES


def auto_install_cuml_enabled() -> bool:
    return not env_flag_disabled(AUTO_INSTALL_ENV_VAR)


def _run_text_command(command: Sequence[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            [*command],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return f"{result.stdout}\n{result.stderr}"


def _cuda_major_from_nvcc() -> int | None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    text = _run_text_command([nvcc, "--version"])
    match = re.search(r"release\s+(\d+)(?:\.\d+)?", text)
    if match is None:
        return None
    return int(match.group(1))


def _cuda_major_from_nvidia_smi() -> int | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    text = _run_text_command([nvidia_smi])
    match = re.search(r"CUDA Version:\s*(\d+)(?:\.\d+)?", text)
    if match is None:
        return None
    return int(match.group(1))


def cuml_cuda_suffix() -> str | None:
    override = os.environ.get(CUDA_SUFFIX_ENV_VAR, "").strip().lower()
    if override:
        if override in {"12", "cuda12"}:
            return "cu12"
        if override in {"13", "cuda13"}:
            return "cu13"
        if override in {"cu12", "cu13"}:
            return override
        return None

    detected = [
        major
        for major in (_cuda_major_from_nvcc(), _cuda_major_from_nvidia_smi())
        if major is not None
    ]
    major = max(detected) if detected else None
    if major == 12:
        return "cu12"
    if major is not None and major >= 13:
        return "cu13"
    return None


def cuml_package_spec() -> str | None:
    suffix = cuml_cuda_suffix()
    if suffix is None:
        return None
    return f"cuml-{suffix}"


def cuml_cuda_install_command() -> tuple[str, ...] | None:
    package = cuml_package_spec()
    if package is None:
        return None
    uv = shutil.which("uv")
    if uv is not None:
        return (
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--index",
            NVIDIA_PYPI_INDEX,
            package,
        )
    return (
        sys.executable,
        "-m",
        "pip",
        "install",
        f"--extra-index-url={NVIDIA_PYPI_INDEX}",
        package,
    )


def install_cuml_cuda_package() -> CumlCudaInstallResult:
    if not auto_install_cuml_enabled():
        return CumlCudaInstallResult(
            attempted=False,
            succeeded=False,
            reason=f"automatic cuML install is disabled by {AUTO_INSTALL_ENV_VAR}",
        )
    if os.environ.get(AUTO_INSTALL_ATTEMPTED_ENV_VAR) == "1":
        return CumlCudaInstallResult(
            attempted=False,
            succeeded=False,
            reason="automatic cuML install was already attempted",
        )
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return CumlCudaInstallResult(
            attempted=False,
            succeeded=False,
            reason="RAPIDS cuML pip packages require Linux or WSL2.",
        )

    package = cuml_package_spec()
    command = cuml_cuda_install_command()
    if package is None or command is None:
        return CumlCudaInstallResult(
            attempted=False,
            succeeded=False,
            reason=(
                "could not determine RAPIDS CUDA wheel suffix; set "
                f"{CUDA_SUFFIX_ENV_VAR}=cu12 or cu13"
            ),
        )

    warn(
        f"CUDA was selected for a cuML-capable model, but {package} is not "
        "available. Attempting to install RAPIDS cuML in the current Python "
        "environment.",
        RuntimeWarning,
        stacklevel=2,
    )
    env = dict(os.environ)
    env[AUTO_INSTALL_ATTEMPTED_ENV_VAR] = "1"
    result = subprocess.run(command, check=False, env=env)
    return CumlCudaInstallResult(
        attempted=True,
        succeeded=result.returncode == 0,
        package=package,
        command=command,
        reason=(
            "install command completed"
            if result.returncode == 0
            else f"install command exited with status {result.returncode}"
        ),
    )


def reexec_current_process() -> None:
    env = dict(os.environ)
    env[AUTO_INSTALL_ATTEMPTED_ENV_VAR] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _status_from_probe(
    name: str,
    probe: Callable[[], tuple[bool, str]],
) -> CumlBackendStatus:
    available, reason = probe()
    return CumlBackendStatus(name=name, available=available, reason=reason)


def cuml_backend_statuses(backends: Sequence[str]) -> list[CumlBackendStatus]:
    statuses: list[CumlBackendStatus] = []
    for backend in backends:
        if backend == "knn-classifier":
            statuses.append(
                _status_from_probe(
                    backend, lambda: gpu_backends.cuml_knn_availability(True)
                )
            )
        elif backend == "knn-regressor":
            statuses.append(
                _status_from_probe(
                    backend, lambda: gpu_backends.cuml_knn_availability(False)
                )
            )
        elif backend == "logistic-regression":
            statuses.append(
                _status_from_probe(
                    backend, gpu_backends.cuml_logistic_regression_availability
                )
            )
        elif backend == "elasticnet":
            statuses.append(
                _status_from_probe(backend, gpu_backends.cuml_elasticnet_availability)
            )
        elif backend == "mbsgd-regressor":
            statuses.append(
                _status_from_probe(
                    backend, gpu_backends.cuml_mbsgd_regressor_availability
                )
            )
        elif backend == "mbsgd-classifier":
            statuses.append(
                _status_from_probe(
                    backend, gpu_backends.cuml_mbsgd_classifier_availability
                )
            )
    return statuses


def _unavailable_reason(statuses: Sequence[CumlBackendStatus]) -> str:
    reasons = [
        f"{status.name}: {status.reason or 'unavailable'}"
        for status in statuses
        if not status.available
    ]
    return "; ".join(reasons)


def check_cuml_cuda_backends(
    device_info: DeviceInfo,
    backends: Sequence[str],
    *,
    enabled: bool = True,
    auto_install: bool = True,
) -> CumlCudaCheckResult:
    selected = tuple(dict.fromkeys(backends))
    if not enabled or not selected:
        return CumlCudaCheckResult(checked=False, available=False, reason="disabled")
    if not device_info.cuda_backends_enabled:
        return CumlCudaCheckResult(
            checked=False,
            available=False,
            reason="CUDA backends are disabled",
        )

    statuses = cuml_backend_statuses(selected)
    if statuses and all(status.available for status in statuses):
        return CumlCudaCheckResult(checked=True, available=True, reason="available")

    detail = _unavailable_reason(statuses) or "cuML backend is unavailable"
    if auto_install:
        install_result = install_cuml_cuda_package()
        if install_result.succeeded:
            warn(
                "RAPIDS cuML was installed. Restarting df-analyze with the "
                "same command so the cuML package is loaded.",
                RuntimeWarning,
                stacklevel=2,
            )
            reexec_current_process()
        if install_result.reason:
            detail = f"{detail} Automatic install: {install_result.reason}."

    warn(
        "CUDA was selected for at least one cuML-capable model, but RAPIDS "
        f"cuML is unavailable. Falling back to sklearn where needed. {detail}",
        RuntimeWarning,
        stacklevel=2,
    )
    return CumlCudaCheckResult(
        checked=True,
        available=False,
        reason=detail,
        installed=False,
    )
