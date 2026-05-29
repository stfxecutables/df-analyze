from __future__ import annotations

import pytest

from df_analyze.device import device_info_from_resolved
from df_analyze.models import cuml_cuda
from df_analyze.models import gpu as gpu_backends


@pytest.mark.fast
def test_cuml_cuda_suffix_uses_highest_detected_major(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNvccProcess:
        returncode = 0
        stdout = "Cuda compilation tools, release 12.4, V12.4.131\n"
        stderr = ""

    class FakeNvidiaSmiProcess:
        returncode = 0
        stdout = "NVIDIA-SMI 595.71.01  CUDA Version: 13.2\n"
        stderr = ""

    def fake_run(command, *args, **kwargs):
        return FakeNvccProcess() if "nvcc" in command[0] else FakeNvidiaSmiProcess()

    monkeypatch.delenv(cuml_cuda.CUDA_SUFFIX_ENV_VAR, raising=False)
    monkeypatch.setattr(
        cuml_cuda.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"nvcc", "nvidia-smi"} else None,
    )
    monkeypatch.setattr(cuml_cuda.subprocess, "run", fake_run)

    assert cuml_cuda.cuml_cuda_suffix() == "cu13"
    assert cuml_cuda.cuml_package_spec() == "cuml-cu13"


@pytest.mark.fast
def test_cuml_cuda_install_command_uses_nvidia_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cuml_cuda.CUDA_SUFFIX_ENV_VAR, "cu13")
    monkeypatch.setattr(
        cuml_cuda.shutil,
        "which",
        lambda name: "/usr/bin/uv" if name == "uv" else None,
    )

    command = cuml_cuda.cuml_cuda_install_command()

    assert command is not None
    assert "cuml-cu13" in command
    assert cuml_cuda.NVIDIA_PYPI_INDEX in command


@pytest.mark.fast
def test_cuml_cuda_check_auto_installs_and_reexecs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReexecRequested(RuntimeError):
        pass

    monkeypatch.setattr(
        gpu_backends,
        "cuml_knn_availability",
        lambda is_classifier=True: (False, "cuml missing"),
    )
    monkeypatch.setattr(
        cuml_cuda,
        "install_cuml_cuda_package",
        lambda: cuml_cuda.CumlCudaInstallResult(
            attempted=True,
            succeeded=True,
            package="cuml-cu12",
            command=("uv", "pip", "install", "cuml-cu12"),
            reason="install command completed",
        ),
    )
    monkeypatch.setattr(
        cuml_cuda,
        "reexec_current_process",
        lambda: (_ for _ in ()).throw(ReexecRequested()),
    )

    with pytest.warns(RuntimeWarning, match="Restarting"):
        with pytest.raises(ReexecRequested):
            cuml_cuda.check_cuml_cuda_backends(
                device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True),
                ["knn-classifier"],
                enabled=True,
                auto_install=True,
            )


@pytest.mark.fast
def test_cuml_cuda_check_warns_and_falls_back_when_install_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_backends,
        "cuml_knn_availability",
        lambda is_classifier=True: (False, "cuml missing"),
    )
    monkeypatch.setattr(
        cuml_cuda,
        "install_cuml_cuda_package",
        lambda: cuml_cuda.CumlCudaInstallResult(
            attempted=True,
            succeeded=False,
            package="cuml-cu12",
            command=("uv", "pip", "install", "cuml-cu12"),
            reason="install command exited with status 1",
        ),
    )

    with pytest.warns(RuntimeWarning, match="Falling back to sklearn"):
        result = cuml_cuda.check_cuml_cuda_backends(
            device_info_from_resolved("cuda", "cuda", torch_cuda_available=True, cuda_driver_available=True),
            ["knn-classifier"],
            enabled=True,
            auto_install=True,
        )

    assert result.checked is True
    assert result.available is False
    assert "install command exited with status 1" in result.reason


@pytest.mark.fast
def test_cuml_cuda_check_passes_when_selected_backends_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_backends,
        "cuml_knn_availability",
        lambda is_classifier=True: (True, ""),
    )

    result = cuml_cuda.check_cuml_cuda_backends(
        device_info_from_resolved("cuda", "cuda", torch_cuda_available=True, cuda_driver_available=True),
        ["knn-classifier"],
        enabled=True,
    )

    assert result.checked is True
    assert result.available is True
