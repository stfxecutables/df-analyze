from __future__ import annotations

import argparse
import importlib
import platform
import re
import shlex
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

NVIDIA_PYPI_INDEX = "https://pypi.nvidia.com"
CUDA13_GROUP = "cuda-cu13"
CUDA12_INEXACT_PACKAGES = ("cudf-cu12==26.4.*", "cuml-cu12==26.4.*")
SUPPORTED_LINUX_MACHINES = {"x86_64", "amd64"}
PREFLIGHT_MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "none": (),
    "core": ("torch", "cuml-knn", "cuml-lr"),
    "classification": (
        "torch",
        "cuml-knn",
        "cuml-lr",
        "cuml-sgd",
        "xgboost",
        "catboost",
        "mlp",
        "gandalf",
        "tabpfn",
    ),
    "regression": (
        "torch",
        "cuml-knn",
        "cuml-elastic",
        "cuml-sgd",
        "xgboost",
        "catboost",
        "mlp",
        "gandalf",
        "tabpfn",
    ),
    "all": (
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
    ),
}
PREFLIGHT_MODEL_ALIASES: dict[str, str] = {
    "torch-cuda": "torch",
    "cuda": "torch",
    "knn": "cuml-knn",
    "cuml": "cuml-knn",
    "lr": "cuml-lr",
    "logistic": "cuml-lr",
    "logistic-regression": "cuml-lr",
    "elastic": "cuml-elastic",
    "elasticnet": "cuml-elastic",
    "sgd": "cuml-sgd",
    "mbsgd": "cuml-sgd",
    "xgb": "xgboost",
    "cat": "catboost",
    "lightgbm": "lightgbm-cpu",
    "lgbm": "lightgbm-cpu",
    "rf": "lightgbm-cpu",
    "dummy": "dummy-cpu",
}
PREFLIGHT_MODEL_HELP = ", ".join(
    [
        "all",
        "classification",
        "regression",
        "core",
        "none",
        "knn",
        "lr",
        "elastic",
        "sgd",
        "xgboost",
        "catboost",
        "mlp",
        "gandalf",
        "tabpfn",
    ]
)


@dataclass(frozen=True)
class GpuSetupPlan:
    supported: bool
    reason: str
    commands: tuple[tuple[str, ...], ...] = ()
    run_args: tuple[str, ...] = ()
    locked: bool = False
    group: str | None = None


@dataclass(frozen=True)
class GpuPreflightResult:
    name: str
    status: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    @property
    def passed(self) -> bool:
        return self.status in {"pass", "skip"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_uv_executable() -> str:
    uv = shutil.which("uv")
    if uv is not None:
        return uv
    for candidate in (Path.home() / ".local" / "bin" / "uv",):
        if candidate.exists():
            return str(candidate)
    return "uv"


def cuda_major_from_text(text: str) -> int | None:
    match = re.search(r"CUDA Version:\s*(\d+)(?:\.\d+)?", text)
    if match is not None:
        return int(match.group(1))
    match = re.search(r"release\s+(\d+)(?:\.\d+)?", text)
    if match is not None:
        return int(match.group(1))
    return None


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


def detect_cuda_major() -> int | None:
    detected: list[int] = []
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        major = cuda_major_from_text(_run_text_command([nvcc, "--version"]))
        if major is not None:
            detected.append(major)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        major = cuda_major_from_text(_run_text_command([nvidia_smi]))
        if major is not None:
            detected.append(major)
    return max(detected) if detected else None


def choose_gpu_setup_plan(
    *,
    sys_platform: str,
    machine: str,
    cuda_major: int | None,
    uv: str,
    python: str,
    allow_inexact_cu12: bool = False,
) -> GpuSetupPlan:
    normalized_machine = machine.strip().lower()
    if sys_platform.startswith("win"):
        return GpuSetupPlan(
            supported=False,
            reason=(
                "Native Windows is not a supported RAPIDS/cuML pip target. "
                "Use WSL2/Linux for df-analyze CUDA cuML runs."
            ),
        )
    if sys_platform == "darwin":
        return GpuSetupPlan(
            supported=False,
            reason=(
                "macOS can use MPS for PyTorch models, but RAPIDS/cuML CUDA "
                "packages require Linux/WSL2 with an NVIDIA GPU."
            ),
        )
    if sys_platform != "linux":
        return GpuSetupPlan(
            supported=False,
            reason=f"Unsupported platform '{sys_platform}'. Use Linux or WSL2.",
        )
    if normalized_machine not in SUPPORTED_LINUX_MACHINES:
        return GpuSetupPlan(
            supported=False,
            reason=(
                f"Unsupported Linux architecture '{machine}'. The locked RAPIDS "
                "wheels are configured for x86_64 Linux."
            ),
        )
    if cuda_major is None:
        return GpuSetupPlan(
            supported=False,
            reason=(
                "Could not detect an NVIDIA CUDA driver with nvcc or nvidia-smi. "
                "Install/enable the NVIDIA driver first, then rerun this command."
            ),
        )
    if cuda_major >= 13:
        return GpuSetupPlan(
            supported=True,
            reason=(
                f"Detected CUDA {cuda_major}; using locked uv group "
                f"'{CUDA13_GROUP}'."
            ),
            commands=((uv, "sync", "--group", CUDA13_GROUP),),
            run_args=(uv, "run", "--group", CUDA13_GROUP),
            locked=True,
            group=CUDA13_GROUP,
        )
    if cuda_major == 12 and allow_inexact_cu12:
        return GpuSetupPlan(
            supported=True,
            reason=(
                "Detected CUDA 12; using inexact cu12 RAPIDS installation because "
                "the current Torch/CUDA 12.8 stack is not lock-compatible with "
                "RAPIDS 26.4 cu12."
            ),
            commands=(
                (uv, "sync", "--inexact"),
                (
                    uv,
                    "pip",
                    "install",
                    "--python",
                    python,
                    "--index",
                    NVIDIA_PYPI_INDEX,
                    *CUDA12_INEXACT_PACKAGES,
                ),
            ),
            run_args=(uv, "run", "--no-sync"),
            locked=False,
        )
    if cuda_major == 12:
        return GpuSetupPlan(
            supported=False,
            reason=(
                "Detected CUDA 12. df-analyze's locked GPU dependency group "
                "currently targets CUDA 13 because RAPIDS 26.4 cu12 conflicts "
                "with the project's locked Torch CUDA 12.8 runtime dependencies. "
                "Use a CUDA 13-capable driver, or rerun with --allow-inexact-cu12 "
                "to install cu12 RAPIDS outside the exact uv lock."
            ),
        )
    return GpuSetupPlan(
        supported=False,
        reason=(
            f"Detected CUDA {cuda_major}. df-analyze GPU setup supports CUDA 13 "
            "through the locked cuda-cu13 group."
        ),
    )


def _format_command(command: Sequence[str]) -> str:
    return shlex.join([*command])


def _run_command(command: Sequence[str], cwd: Path) -> int:
    print(f"+ {_format_command(command)}", flush=True)
    result = subprocess.run([*command], cwd=cwd, check=False)
    return result.returncode


def _preflight_command(
    run_args: Sequence[str],
    models: Sequence[str],
    *,
    tabpfn_fit_smoke: bool = False,
) -> tuple[str, ...]:
    command = [*run_args, "python", "-m", "df_analyze.gpu_setup", "--preflight-only"]
    if models:
        command.extend(["--models", *models])
    if tabpfn_fit_smoke:
        command.append("--tabpfn-fit-smoke")
    return tuple(command)


def _run_hint(run_args: Sequence[str]) -> str:
    return f"Run df-analyze with: {_format_command(run_args)} python df-analyze.py ..."


def _short_reason(error: BaseException | str) -> str:
    if isinstance(error, str):
        message = error
        name = ""
    else:
        message = str(error)
        name = type(error).__name__
    message = message.strip().splitlines()[0] if message.strip() else "unknown error"
    reason = f"{name}: {message}" if name else message
    return reason[:500]


def _result(
    name: str,
    available: bool,
    detail: str = "",
    *,
    skip: bool = False,
) -> GpuPreflightResult:
    status = "skip" if skip else "pass" if available else "fail"
    return GpuPreflightResult(name=name, status=status, detail=detail)


def _import_status(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as e:
        return False, _short_reason(e)


def _torch_cuda_preflight() -> GpuPreflightResult:
    try:
        import torch

        if not torch.cuda.is_available():
            return _result(
                "torch",
                False,
                (
                    f"torch {torch.__version__} reports cuda unavailable "
                    f"(torch.version.cuda={torch.version.cuda!r})."
                ),
            )
        device_name = torch.cuda.get_device_name(0)
        value = torch.ones(1, device="cuda").cpu().item()
        if value != 1.0:
            return _result("torch", False, "CUDA tensor smoke check returned bad data.")
        return _result(
            "torch",
            True,
            (
                f"torch {torch.__version__}, runtime CUDA {torch.version.cuda}, "
                f"device 0: {device_name}"
            ),
        )
    except Exception as e:
        return _result("torch", False, _short_reason(e))


def _cuml_knn_preflight() -> GpuPreflightResult:
    from df_analyze.models import gpu

    cls_ok, cls_reason = gpu.cuml_knn_availability(True)
    reg_ok, reg_reason = gpu.cuml_knn_availability(False)
    detail = "; ".join(
        reason
        for ok, reason in ((cls_ok, cls_reason), (reg_ok, reg_reason))
        if not ok and reason
    )
    return _result("cuml-knn", cls_ok and reg_ok, detail)


def _cuml_lr_preflight() -> GpuPreflightResult:
    from df_analyze.models import gpu

    available, reason = gpu.cuml_logistic_regression_availability()
    return _result("cuml-lr", available, reason)


def _cuml_elastic_preflight() -> GpuPreflightResult:
    from df_analyze.models import gpu

    available, reason = gpu.cuml_elasticnet_availability()
    return _result("cuml-elastic", available, reason)


def _cuml_sgd_preflight() -> GpuPreflightResult:
    from df_analyze.models import gpu

    cls_ok, cls_reason = gpu.cuml_mbsgd_classifier_availability()
    reg_ok, reg_reason = gpu.cuml_mbsgd_regressor_availability()
    detail = "; ".join(
        reason
        for ok, reason in ((cls_ok, cls_reason), (reg_ok, reg_reason))
        if not ok and reason
    )
    return _result("cuml-sgd", cls_ok and reg_ok, detail)


def _xgboost_preflight() -> GpuPreflightResult:
    from df_analyze.models import gpu

    available, reason = gpu.xgboost_cuda_availability()
    return _result("xgboost", available, reason)


def _catboost_preflight() -> GpuPreflightResult:
    from df_analyze.models import gpu

    available, reason = gpu.catboost_cuda_availability()
    return _result("catboost", available, reason)


def _mlp_preflight() -> GpuPreflightResult:
    torch_result = _torch_cuda_preflight()
    if torch_result.failed:
        return GpuPreflightResult(
            name="mlp",
            status="fail",
            detail=f"PyTorch CUDA unavailable: {torch_result.detail}",
        )
    for module in ("skorch", "df_analyze.models.mlp"):
        available, reason = _import_status(module)
        if not available:
            return _result("mlp", False, f"{module}: {reason}")
    return _result("mlp", True, "skorch and PyTorch CUDA are available.")


def _gandalf_preflight() -> GpuPreflightResult:
    torch_result = _torch_cuda_preflight()
    if torch_result.failed:
        return GpuPreflightResult(
            name="gandalf",
            status="fail",
            detail=f"PyTorch CUDA unavailable: {torch_result.detail}",
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*isinstance\(treespec, LeafSpec\).*deprecated.*",
        )
        for module in ("lightning", "df_analyze.models.gandalf"):
            available, reason = _import_status(module)
            if not available:
                return _result("gandalf", False, f"{module}: {reason}")
    return _result("gandalf", True, "Lightning and PyTorch CUDA are available.")


def _tabpfn_preflight() -> GpuPreflightResult:
    from df_analyze.models import gpu

    api_ok, api_reason = gpu.tabpfn_api_status()
    v25_ok, v25_reason = gpu.tabpfn_v25_api_status()
    if api_ok and v25_ok:
        return _result(
            "tabpfn",
            True,
            "TabPFN API is importable; fit smoke is opt-in.",
        )
    detail = "; ".join(
        reason
        for ok, reason in ((api_ok, api_reason), (v25_ok, v25_reason))
        if not ok and reason
    )
    return _result("tabpfn", False, detail)


def _tabpfn_fit_preflight() -> GpuPreflightResult:
    try:
        import pandas as pd

        from df_analyze.device import device_info_from_resolved
        from df_analyze.models.tabpfn import TabPFNClassifier

        X = pd.DataFrame(
            {
                "x0": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
                "x1": [1.0, 0.9, 0.8, 0.2, 0.1, 0.0],
            }
        )
        y = pd.Series([0, 0, 0, 1, 1, 1], name="target")
        model = TabPFNClassifier()
        model.set_device_info(
            device_info_from_resolved(
                "cuda",
                "cuda",
                torch_cuda_available=True,
                cuda_driver_available=True,
            )
        )
        model.fit(X, y)
        preds = model.predict(X)
        if len(preds) != len(X):
            return _result("tabpfn-fit", False, "prediction length mismatch.")
        return _result("tabpfn-fit", True, "tiny CUDA fit succeeded.")
    except Exception as e:
        message = str(e).lower()
        if any(
            term in message
            for term in (
                "download",
                "weight",
                "checkpoint",
                "cache",
                "license",
                "token",
            )
        ):
            return _result(
                "tabpfn-fit",
                False,
                (
                    "TabPFN fit reached the model-weight/license/cache flow. "
                    f"{_short_reason(e)}"
                ),
            )
        return _result("tabpfn-fit", False, _short_reason(e))


def _lightgbm_cpu_preflight() -> GpuPreflightResult:
    available, reason = _import_status("lightgbm")
    if not available:
        return _result("lightgbm-cpu", False, reason)
    return _result(
        "lightgbm-cpu",
        True,
        "df-analyze pins lgbm/rf to CPU; no CUDA LightGBM install is required.",
        skip=True,
    )


def _dummy_cpu_preflight() -> GpuPreflightResult:
    return _result(
        "dummy-cpu",
        True,
        "dummy is a CPU baseline and does not need CUDA.",
        skip=True,
    )


PREFLIGHT_CHECKS: dict[str, Callable[[], GpuPreflightResult]] = {
    "torch": _torch_cuda_preflight,
    "cuml-knn": _cuml_knn_preflight,
    "cuml-lr": _cuml_lr_preflight,
    "cuml-elastic": _cuml_elastic_preflight,
    "cuml-sgd": _cuml_sgd_preflight,
    "xgboost": _xgboost_preflight,
    "catboost": _catboost_preflight,
    "mlp": _mlp_preflight,
    "gandalf": _gandalf_preflight,
    "tabpfn": _tabpfn_preflight,
    "tabpfn-fit": _tabpfn_fit_preflight,
    "lightgbm-cpu": _lightgbm_cpu_preflight,
    "dummy-cpu": _dummy_cpu_preflight,
}


def expand_preflight_models(models: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(models) if models else ("all",)
    expanded: list[str] = []
    for raw in requested:
        key = raw.strip().lower()
        if not key:
            continue
        if key in PREFLIGHT_MODEL_GROUPS:
            candidates = PREFLIGHT_MODEL_GROUPS[key]
        else:
            canonical = PREFLIGHT_MODEL_ALIASES.get(key, key)
            if canonical not in PREFLIGHT_CHECKS:
                valid = sorted(
                    {
                        *PREFLIGHT_MODEL_GROUPS.keys(),
                        *PREFLIGHT_MODEL_ALIASES.keys(),
                        *PREFLIGHT_CHECKS.keys(),
                    }
                )
                raise ValueError(
                    f"Unknown preflight model '{raw}'. Valid values include: "
                    f"{', '.join(valid)}"
                )
            candidates = (canonical,)
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    return tuple(expanded)


def gpu_preflight_results(
    models: Sequence[str],
    *,
    tabpfn_fit_smoke: bool = False,
) -> list[GpuPreflightResult]:
    expanded = list(expand_preflight_models(models))
    if tabpfn_fit_smoke and "tabpfn" in expanded and "tabpfn-fit" not in expanded:
        expanded.append("tabpfn-fit")

    results: list[GpuPreflightResult] = []
    for model in expanded:
        check = PREFLIGHT_CHECKS[model]
        results.append(check())
    return results


def print_preflight_results(results: Sequence[GpuPreflightResult]) -> None:
    print("CUDA GPU preflight checks:", flush=True)
    if not results:
        print("  SKIP no model backends selected for preflight.", flush=True)
        return
    for result in results:
        status = result.status.upper()
        detail = f" - {result.detail}" if result.detail else ""
        print(f"  {status:<4} {result.name}{detail}", flush=True)


def run_gpu_preflight(
    models: Sequence[str],
    *,
    tabpfn_fit_smoke: bool = False,
) -> int:
    try:
        results = gpu_preflight_results(
            models,
            tabpfn_fit_smoke=tabpfn_fit_smoke,
        )
    except ValueError as e:
        print(_short_reason(e), file=sys.stderr, flush=True)
        return 2

    print_preflight_results(results)
    failures = [result for result in results if result.failed]
    if failures:
        print(
            "One or more CUDA GPU preflight checks failed. If you do not use a "
            "failed model, rerun setup with --models limited to the models in "
            "your df-analyze command.",
            flush=True,
        )
        return 1
    print("All selected CUDA GPU preflight checks passed.", flush=True)
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="df-analyze-gpu-setup",
        description="Detect and sync df-analyze CUDA/RAPIDS dependencies.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=project_root(),
        help="Project directory containing pyproject.toml and uv.lock.",
    )
    parser.add_argument(
        "--uv",
        default=default_uv_executable(),
        help="uv executable to run.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for inexact cu12 pip installs.",
    )
    parser.add_argument(
        "--cuda-major",
        type=int,
        default=None,
        help="Override detected CUDA major version.",
    )
    parser.add_argument(
        "--allow-inexact-cu12",
        action="store_true",
        help="Allow best-effort CUDA 12 RAPIDS install outside the exact uv lock.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected commands without running them.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip post-sync CUDA GPU backend preflight checks.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=("all",),
        help=(
            "CUDA GPU backend preflight scope. Use groups or model names such "
            f"as: {PREFLIGHT_MODEL_HELP}. Default: all."
        ),
    )
    parser.add_argument(
        "--tabpfn-fit-smoke",
        action="store_true",
        help=(
            "After the TabPFN API check, run a tiny CUDA TabPFN fit. This can "
            "download model weights or trigger license/token setup."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.preflight_only:
        return run_gpu_preflight(
            args.models,
            tabpfn_fit_smoke=args.tabpfn_fit_smoke,
        )

    cuda_major = args.cuda_major if args.cuda_major is not None else detect_cuda_major()
    project = args.project.resolve()
    plan = choose_gpu_setup_plan(
        sys_platform=sys.platform,
        machine=platform.machine(),
        cuda_major=cuda_major,
        uv=args.uv,
        python=args.python,
        allow_inexact_cu12=args.allow_inexact_cu12,
    )

    print(plan.reason, flush=True)
    if not plan.supported:
        return 2

    commands = [*plan.commands]
    if plan.run_args and not args.skip_smoke:
        commands.append(
            _preflight_command(
                plan.run_args,
                args.models,
                tabpfn_fit_smoke=args.tabpfn_fit_smoke,
            )
        )

    if args.dry_run:
        for command in commands:
            print(_format_command(command))
        if plan.run_args:
            print(_run_hint(plan.run_args))
        return 0

    for command in commands:
        code = _run_command(command, project)
        if code != 0:
            return code

    if plan.run_args:
        print(_run_hint(plan.run_args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
