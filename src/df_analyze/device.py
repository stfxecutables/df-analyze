from __future__ import annotations

import gc
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Literal, Union

import torch

DeviceRequest = Literal["auto", "cpu", "cuda", "mps"]
ResolvedDevice = Literal["cpu", "cuda", "mps"]

DEVICE_CHOICES: tuple[DeviceRequest, ...] = ("auto", "cpu", "cuda", "mps")
RESOLVED_DEVICE_CHOICES: tuple[ResolvedDevice, ...] = ("cpu", "cuda", "mps")
RANDOM_DEVICE_CHOICES: tuple[DeviceRequest, ...] = ("auto", "cpu")
FATAL_ACCELERATOR_ERROR_MARKERS = (
    "out of memory",
    "cuda error",
)


@dataclass(frozen=True)
class DeviceInfo:
    requested: DeviceRequest
    resolved: ResolvedDevice
    torch_device: ResolvedDevice
    skorch_device: ResolvedDevice
    lightning_accelerator: str
    lightning_devices: Union[int, str]
    accelerated: bool
    torch_cuda_available: bool
    cuda_driver_available: bool
    mps_available: bool

    @property
    def torch_uses_cuda(self) -> bool:
        return self.torch_device == "cuda"

    @property
    def torch_uses_accelerator(self) -> bool:
        return self.torch_device in {"cuda", "mps"}

    @property
    def torch_accelerated(self) -> bool:
        return self.torch_uses_accelerator

    @property
    def cuda_backends_enabled(self) -> bool:
        return self.resolved == "cuda" and (
            self.torch_cuda_available or self.cuda_driver_available
        )


def normalize_device_request(value: str) -> DeviceRequest:
    device = value.strip().lower()
    if device not in DEVICE_CHOICES:
        choices = ", ".join(DEVICE_CHOICES)
        raise ValueError(f"Invalid device '{value}'. Expected one of: {choices}.")
    return device  # type: ignore[return-value]


def normalize_resolved_device(value: str) -> ResolvedDevice:
    device = value.strip().lower()
    if device not in RESOLVED_DEVICE_CHOICES:
        choices = ", ".join(RESOLVED_DEVICE_CHOICES)
        raise ValueError(
            f"Invalid resolved device '{value}'. Expected one of: {choices}."
        )
    return device  # type: ignore[return-value]


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def cuda_visible_devices_allows_gpu() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return True
    return visible.strip().lower() not in {"", "-1", "none", "nodevfiles"}


def cuda_driver_available() -> bool:
    if not cuda_visible_devices_allows_gpu():
        return False
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def cuda_backend_available() -> bool:
    return cuda_available() or cuda_driver_available()


def configure_torch_cuda_performance(device_info: DeviceInfo) -> None:
    if device_info.torch_device != "cuda":
        return
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def cleanup_accelerator_memory(device_info: DeviceInfo) -> None:
    if not device_info.cuda_backends_enabled and device_info.torch_device != "mps":
        return
    gc.collect()
    if device_info.torch_device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    if device_info.torch_device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    try:
        import cupy  # type: ignore[import-not-found]

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def is_fatal_accelerator_error(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, MemoryError):
            return True
        message = str(current).lower()
        if any(marker in message for marker in FATAL_ACCELERATOR_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def mps_available() -> bool:
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is None:
        return False
    is_built = getattr(mps_backend, "is_built", None)
    if is_built is not None and not bool(is_built()):
        return False
    return bool(mps_backend.is_available())


def device_info_from_resolved(
    requested: str,
    resolved: str,
    *,
    torch_cuda_available: bool | None = None,
    cuda_driver_available: bool | None = None,
    mps_is_available: bool | None = None,
) -> DeviceInfo:
    request = normalize_device_request(requested)
    device = normalize_resolved_device(resolved)
    torch_has_cuda = (
        bool(globals()["cuda_available"]())
        if torch_cuda_available is None
        else torch_cuda_available
    )
    driver_has_cuda = (
        bool(globals()["cuda_driver_available"]())
        if cuda_driver_available is None
        else cuda_driver_available
    )
    has_mps = (
        bool(globals()["mps_available"]())
        if mps_is_available is None
        else mps_is_available
    )

    if device == "cpu":
        return DeviceInfo(
            requested=request,
            resolved=device,
            torch_device="cpu",
            skorch_device="cpu",
            lightning_accelerator="cpu",
            lightning_devices="auto",
            accelerated=False,
            torch_cuda_available=torch_has_cuda,
            cuda_driver_available=driver_has_cuda,
            mps_available=has_mps,
        )
    if device == "cuda":
        torch_device: ResolvedDevice = "cuda" if torch_has_cuda else "cpu"
        return DeviceInfo(
            requested=request,
            resolved=device,
            torch_device=torch_device,
            skorch_device=torch_device,
            lightning_accelerator="gpu" if torch_has_cuda else "cpu",
            lightning_devices=1 if torch_has_cuda else "auto",
            accelerated=torch_has_cuda,
            torch_cuda_available=torch_has_cuda,
            cuda_driver_available=driver_has_cuda,
            mps_available=has_mps,
        )
    torch_device = "mps" if has_mps else "cpu"
    return DeviceInfo(
        requested=request,
        resolved=device,
        torch_device=torch_device,
        skorch_device=torch_device,
        lightning_accelerator="mps" if has_mps else "cpu",
        lightning_devices=1 if has_mps else "auto",
        accelerated=has_mps,
        torch_cuda_available=torch_has_cuda,
        cuda_driver_available=driver_has_cuda,
        mps_available=has_mps,
    )


def device_info_from_json(value: Any) -> DeviceInfo:
    if isinstance(value, DeviceInfo):
        if all(
            hasattr(value, attr)
            for attr in (
                "torch_cuda_available",
                "cuda_driver_available",
                "mps_available",
            )
        ):
            return value
        value = value.__dict__
    if not isinstance(value, dict):
        raise TypeError(f"Cannot restore DeviceInfo from {type(value).__name__}.")

    requested = value.get("requested", "auto")
    resolved = value.get("resolved", value.get("device", "cpu"))
    torch_device = value.get("torch_device")
    torch_has_cuda = value.get("torch_cuda_available")
    if torch_has_cuda is None and torch_device is not None:
        torch_has_cuda = str(torch_device).lower() == "cuda"

    driver_has_cuda = value.get("cuda_driver_available")
    if (
        driver_has_cuda is None
        and str(resolved).lower() == "cuda"
        and torch_device is not None
    ):
        driver_has_cuda = True

    has_mps = value.get("mps_available")
    if has_mps is None and torch_device is not None:
        has_mps = str(torch_device).lower() == "mps"
    return device_info_from_resolved(
        str(requested),
        str(resolved),
        torch_cuda_available=None if torch_has_cuda is None else bool(torch_has_cuda),
        cuda_driver_available=None if driver_has_cuda is None else bool(driver_has_cuda),
        mps_is_available=None if has_mps is None else bool(has_mps),
    )


def resolve_device(requested: str = "auto") -> DeviceInfo:
    request = normalize_device_request(requested)
    torch_has_cuda = cuda_available()
    driver_has_cuda = cuda_driver_available()
    any_cuda_backend = torch_has_cuda or driver_has_cuda
    has_mps = mps_available()

    if request == "auto":
        if any_cuda_backend:
            return device_info_from_resolved(
                request,
                "cuda",
                torch_cuda_available=torch_has_cuda,
                cuda_driver_available=driver_has_cuda,
                mps_is_available=has_mps,
            )
        if has_mps:
            return device_info_from_resolved(
                request,
                "mps",
                torch_cuda_available=torch_has_cuda,
                cuda_driver_available=driver_has_cuda,
                mps_is_available=has_mps,
            )
        return device_info_from_resolved(
            request,
            "cpu",
            torch_cuda_available=torch_has_cuda,
            cuda_driver_available=driver_has_cuda,
            mps_is_available=has_mps,
        )

    if request == "cpu":
        return device_info_from_resolved(
            request,
            "cpu",
            torch_cuda_available=torch_has_cuda,
            cuda_driver_available=driver_has_cuda,
            mps_is_available=has_mps,
        )

    if request == "cuda":
        if not any_cuda_backend:
            raise RuntimeError(
                "CUDA was requested with --device cuda, but CUDA is not available."
            )
        return device_info_from_resolved(
            request,
            "cuda",
            torch_cuda_available=torch_has_cuda,
            cuda_driver_available=driver_has_cuda,
            mps_is_available=has_mps,
        )

    if not has_mps:
        raise RuntimeError(
            "MPS was requested with --device mps, but MPS is not available."
        )
    return device_info_from_resolved(
        request,
        "mps",
        torch_cuda_available=torch_has_cuda,
        cuda_driver_available=driver_has_cuda,
        mps_is_available=has_mps,
    )
