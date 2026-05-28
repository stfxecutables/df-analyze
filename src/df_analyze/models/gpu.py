from __future__ import annotations

import os
import json
import warnings
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from importlib import import_module
from io import StringIO
from inspect import Parameter, signature
from typing import Any, Iterable

import numpy as np

from df_analyze.device import (
    DeviceInfo,
    cuda_available,  # noqa: F401 - retained for compatibility with tests/callers
    cuda_driver_available,  # noqa: F401 - retained for compatibility with tests/callers
    mps_available,  # noqa: F401 - retained for compatibility with tests/callers
)


_WARNED_MISSING_CUML_BACKENDS: set[str] = set()
_WARNED_UNAVAILABLE_GPU_BACKENDS: set[str] = set()
_CUML_KNN_METRIC_ALIASES = {
    "l2": "euclidean",
    "euclidean": "euclidean",
}


def _selected_model_names(selected_models: Iterable[Any] | None) -> set[str] | None:
    if selected_models is None:
        return None
    names: set[str] = set()
    for model in selected_models:
        value = getattr(model, "value", model)
        names.add(str(value).lower())
    return names


def _model_selected(selected_models: set[str] | None, *names: str) -> bool:
    return selected_models is None or any(name in selected_models for name in names)


def _get_optional_cls(module_name: str, cls_name: str) -> type | None:
    try:
        module = import_module(module_name)
    except Exception:
        return None
    return getattr(module, cls_name, None)


def _short_reason(error: BaseException | str) -> str:
    if isinstance(error, str):
        message = error
        name = ""
    else:
        message = str(error)
        name = type(error).__name__
    message = message.strip().splitlines()[0] if message.strip() else "unknown error"
    reason = f"{name}: {message}" if name else message
    return reason[:300]


def get_cuml_knn_cls(is_classifier: bool) -> type | None:
    name = "KNeighborsClassifier" if is_classifier else "KNeighborsRegressor"
    return _get_optional_cls("cuml.neighbors", name)


def get_cuml_logistic_regression_cls() -> type | None:
    return _get_optional_cls("cuml.linear_model", "LogisticRegression")


def get_cuml_elasticnet_cls() -> type | None:
    return _get_optional_cls("cuml.linear_model", "ElasticNet")


def get_cuml_mbsgd_classifier_cls() -> type | None:
    return _get_optional_cls("cuml.linear_model", "MBSGDClassifier")


def get_cuml_mbsgd_regressor_cls() -> type | None:
    return _get_optional_cls("cuml.linear_model", "MBSGDRegressor")


def has_cuml_knn(is_classifier: bool) -> bool:
    return cuml_knn_available(is_classifier)


def normalize_cuml_knn_metric(metric: str) -> str | None:
    return _CUML_KNN_METRIC_ALIASES.get(metric)


def cuml_knn_metric_available(metric: str, is_classifier: bool = True) -> bool:
    return normalize_cuml_knn_metric(metric) is not None


def cuml_knn_metric_choices(metrics: list[str]) -> list[str]:
    return [metric for metric in metrics if normalize_cuml_knn_metric(metric) is not None]


def _toy_classification_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.2],
            [0.8, 0.8],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
    return X, y


def _toy_regression_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.2],
            [0.8, 0.8],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.5, 0.75, 1.25, 0.1, 1.1], dtype=np.float32)
    return X, y


def _lightgbm_boosting_type(value: str | None = None) -> str:
    return "rf" if str(value or "").lower() == "rf" else "gbdt"


def _lightgbm_smoke_args(device_type: str, boosting_type: str) -> dict[str, Any]:
    args: dict[str, Any] = {
        "device_type": device_type,
        "n_estimators": 1,
        "num_leaves": 2,
        "min_child_samples": 1,
        "min_data_in_bin": 1,
        "verbosity": -1,
    }
    if _lightgbm_boosting_type(boosting_type) == "rf":
        args.update(
            boosting_type="rf",
            bagging_freq=1,
            bagging_fraction=0.75,
            feature_fraction=0.8,
        )
    return args


def _smoke_fit_predict(
    model_cls: type | None,
    args: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    require_proba: bool = False,
) -> bool:
    available, _ = _smoke_fit_predict_status(
        model_cls,
        args,
        X,
        y,
        require_proba=require_proba,
    )
    return available


def _smoke_fit_predict_status(
    model_cls: type | None,
    args: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    require_proba: bool = False,
    model_name: str = "GPU backend class",
) -> tuple[bool, str]:
    if model_cls is None:
        return False, f"{model_name} could not be imported."

    try:
        clean_args = with_numpy_output(model_cls, args)
        clean_args = filter_model_args(model_cls, clean_args)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator = model_cls(**clean_args)
            estimator.fit(X, y)
            as_numpy(estimator.predict(X[:2]))
            if require_proba:
                predict_proba = getattr(estimator, "predict_proba", None)
                if predict_proba is None:
                    return False, f"{model_name} does not expose predict_proba."
                as_numpy(predict_proba(X[:2]))
        return True, ""
    except Exception as e:
        return False, _short_reason(e)


def cuml_knn_availability(is_classifier: bool = True) -> tuple[bool, str]:
    X, y = (
        _toy_classification_data()
        if is_classifier
        else _toy_regression_data()
    )
    name = "KNeighborsClassifier" if is_classifier else "KNeighborsRegressor"
    return _smoke_fit_predict_status(
        get_cuml_knn_cls(is_classifier),
        {"n_neighbors": 1, "weights": "uniform", "metric": "euclidean"},
        X,
        y,
        require_proba=is_classifier,
        model_name=f"cuml.neighbors.{name}",
    )


@lru_cache(maxsize=None)
def cuml_knn_available(is_classifier: bool = True) -> bool:
    available, _ = cuml_knn_availability(is_classifier)
    return available


def cuml_logistic_regression_availability() -> tuple[bool, str]:
    X, y = _toy_classification_data()
    model_cls = get_cuml_logistic_regression_cls()
    args = {
        "max_iter": 20,
        "penalty": "elasticnet",
        "solver": "saga",
        "l1_ratio": 0.5,
        "C": 1.0,
    }
    if model_cls is not None:
        args = cuml_logistic_regression_args(model_cls, args)
    return _smoke_fit_predict_status(
        model_cls,
        args,
        X,
        y,
        require_proba=True,
        model_name="cuml.linear_model.LogisticRegression",
    )


@lru_cache(maxsize=None)
def cuml_logistic_regression_available() -> bool:
    available, _ = cuml_logistic_regression_availability()
    return available


def cuml_elasticnet_availability() -> tuple[bool, str]:
    X, y = _toy_regression_data()
    return _smoke_fit_predict_status(
        get_cuml_elasticnet_cls(),
        {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 20},
        X,
        y,
        model_name="cuml.linear_model.ElasticNet",
    )


@lru_cache(maxsize=None)
def cuml_elasticnet_available() -> bool:
    available, _ = cuml_elasticnet_availability()
    return available


def cuml_mbsgd_regressor_availability() -> tuple[bool, str]:
    X, y = _toy_regression_data()
    return _smoke_fit_predict_status(
        get_cuml_mbsgd_regressor_cls(),
        {"loss": "squared_loss", "penalty": "l2", "eta0": 0.01, "epochs": 3},
        X,
        y,
        model_name="cuml.linear_model.MBSGDRegressor",
    )


@lru_cache(maxsize=None)
def cuml_mbsgd_regressor_available() -> bool:
    available, _ = cuml_mbsgd_regressor_availability()
    return available


def cuml_mbsgd_classifier_availability() -> tuple[bool, str]:
    X, y = _toy_classification_data()
    return _smoke_fit_predict_status(
        get_cuml_mbsgd_classifier_cls(),
        {"loss": "hinge", "penalty": "l2", "eta0": 0.01, "epochs": 3},
        X,
        y,
        model_name="cuml.linear_model.MBSGDClassifier",
    )


@lru_cache(maxsize=None)
def cuml_mbsgd_classifier_available() -> bool:
    available, _ = cuml_mbsgd_classifier_availability()
    return available


@lru_cache(maxsize=None)
def _lightgbm_device_type_availability(
    device_type: str, boosting_type: str = "gbdt"
) -> tuple[bool, str]:
    try:
        from lightgbm import LGBMClassifier

        X, y = _toy_classification_data()
        model = LGBMClassifier(
            **_lightgbm_smoke_args(device_type, boosting_type)
        )
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            model.fit(X, y)
        return True, ""
    except Exception as e:
        return False, _short_reason(e)


def lightgbm_cuda_availability() -> tuple[bool, str]:
    return _lightgbm_device_type_availability("cuda", "gbdt")


def lightgbm_cuda_rf_availability() -> tuple[bool, str]:
    return _lightgbm_device_type_availability("cuda", "rf")


@lru_cache(maxsize=None)
def lightgbm_cuda_available() -> bool:
    available, _ = lightgbm_cuda_availability()
    return available


@lru_cache(maxsize=None)
def lightgbm_cuda_rf_available() -> bool:
    available, _ = lightgbm_cuda_rf_availability()
    return available


def lightgbm_best_device_type(boosting_type: str = "gbdt") -> tuple[str | None, str]:
    boosting = _lightgbm_boosting_type(boosting_type)
    check = (
        lightgbm_cuda_rf_availability
        if boosting == "rf"
        else lightgbm_cuda_availability
    )
    available, reason = check()
    if available:
        return "cuda", ""
    return None, reason


def catboost_cuda_availability() -> tuple[bool, str]:
    try:
        from catboost import CatBoostClassifier

        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 0, 1, 1])
        model = CatBoostClassifier(
            task_type="GPU",
            devices=os.environ.get("DF_ANALYZE_CATBOOST_DEVICES", "0"),
            iterations=1,
            depth=1,
            learning_rate=1.0,
            verbose=False,
            allow_writing_files=False,
            random_seed=0,
        )
        model.fit(X, y)
        return True, ""
    except Exception as e:
        return False, _short_reason(e)


@lru_cache(maxsize=None)
def catboost_cuda_available() -> bool:
    available, _ = catboost_cuda_availability()
    return available


def xgboost_cuda_args() -> dict[str, Any]:
    return {
        "device": os.environ.get("DF_ANALYZE_XGBOOST_DEVICE", "cuda"),
        "tree_method": os.environ.get("DF_ANALYZE_XGBOOST_TREE_METHOD", "hist"),
    }


def _config_values(value: Any, keys: set[str]) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys:
                values.append(str(item))
            values.extend(_config_values(item, keys))
    elif isinstance(value, list):
        for item in value:
            values.extend(_config_values(item, keys))
    return values


def _xgboost_booster_uses_cuda(model: Any) -> bool:
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    values = _config_values(config, {"device", "tree_method", "predictor"})
    return any(
        value.startswith("cuda") or value in {"gpu_hist", "gpu_predictor"}
        for value in values
    )


def xgboost_cuda_availability() -> tuple[bool, str]:
    try:
        from xgboost import XGBClassifier

        X, y = _toy_classification_data()
        model = XGBClassifier(
            **xgboost_cuda_args(),
            n_estimators=1,
            max_depth=1,
            learning_rate=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
            random_state=0,
        )
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            model.fit(X, y)
        if not _xgboost_booster_uses_cuda(model):
            return (
                False,
                "XGBoost fit succeeded but booster config did not report CUDA/GPU use.",
            )
        return True, ""
    except Exception as e:
        return False, _short_reason(e)


@lru_cache(maxsize=None)
def xgboost_cuda_available() -> bool:
    available, _ = xgboost_cuda_availability()
    return available


def tabpfn_v25_api_status() -> tuple[bool, str]:
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
    except Exception as e:
        return False, _short_reason(e)

    if not hasattr(ModelVersion, "V2_5"):
        return False, "tabpfn.constants.ModelVersion does not expose V2_5."

    missing = [
        cls.__name__
        for cls in (TabPFNClassifier, TabPFNRegressor)
        if not callable(getattr(cls, "create_default_for_version", None))
    ]
    if len(missing) > 0:
        joined = ", ".join(missing)
        return False, f"{joined} missing create_default_for_version."

    return True, ""


def tabpfn_v25_api_available() -> bool:
    available, _ = tabpfn_v25_api_status()
    return available


def tabpfn_api_status() -> tuple[bool, str]:
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except Exception as e:
        return False, _short_reason(e)

    missing = [
        cls.__name__
        for cls in (TabPFNClassifier, TabPFNRegressor)
        if not callable(cls)
    ]
    if len(missing) > 0:
        joined = ", ".join(missing)
        return False, f"{joined} is not callable."

    return True, ""


def tabpfn_api_available() -> bool:
    available, _ = tabpfn_api_status()
    return available


def _signature_params(model_cls: type) -> dict[str, Parameter] | None:
    try:
        return dict(signature(model_cls).parameters)
    except (TypeError, ValueError):
        return None


def accepts_var_kwargs(model_cls: type) -> bool:
    params = _signature_params(model_cls)
    if params is None:
        return False
    return any(param.kind == Parameter.VAR_KEYWORD for param in params.values())


def accepts_arg(model_cls: type, arg: str) -> bool:
    params = _signature_params(model_cls)
    if params is None:
        return True
    return arg in params or accepts_var_kwargs(model_cls)


def filter_model_args(model_cls: type, args: dict[str, Any]) -> dict[str, Any]:
    if _signature_params(model_cls) is None or accepts_var_kwargs(model_cls):
        return dict(args)
    return {name: value for name, value in args.items() if accepts_arg(model_cls, name)}


def with_numpy_output(model_cls: type, args: dict[str, Any]) -> dict[str, Any]:
    clean_args = dict(args)
    if accepts_arg(model_cls, "output_type"):
        clean_args["output_type"] = "numpy"
    return clean_args


def cuml_logistic_regression_args(
    model_cls: type, args: dict[str, Any]
) -> dict[str, Any]:
    clean_args = dict(args)
    clean_args["solver"] = "qn"
    clean_args["penalty"] = "l2"
    clean_args["tol"] = clean_args.get("tol", 1e-2)
    clean_args.pop("l1_ratio", None)
    clean_args = with_numpy_output(model_cls, clean_args)
    return filter_model_args(model_cls, clean_args)


def warn_missing_cuml_backend(model_name: str) -> None:
    if model_name in _WARNED_MISSING_CUML_BACKENDS:
        return
    warnings.warn(
        f"CUDA was selected for {model_name}, but the optional RAPIDS cuML "
        "backend is unavailable. Falling back to sklearn.",
        RuntimeWarning,
        stacklevel=3,
    )
    _WARNED_MISSING_CUML_BACKENDS.add(model_name)


def warn_unavailable_gpu_backend(name: str, reason: str | None = None) -> None:
    if name in _WARNED_UNAVAILABLE_GPU_BACKENDS:
        return
    extra = "" if reason is None else f" {reason}"
    warnings.warn(
        f"CUDA was selected for {name}, but its GPU backend is unavailable. "
        f"Falling back to CPU.{extra}",
        RuntimeWarning,
        stacklevel=3,
    )
    _WARNED_UNAVAILABLE_GPU_BACKENDS.add(name)


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, "get"):
        return np.asarray(value.get())

    if hasattr(value, "to_numpy"):
        array = value.to_numpy()
        if hasattr(array, "get"):
            array = array.get()
        return np.asarray(array)

    values = getattr(value, "values", None)
    if values is not None:
        if hasattr(values, "get"):
            values = values.get()
        elif hasattr(values, "to_numpy"):
            values = values.to_numpy()
        return np.asarray(values)

    return np.asarray(value)


def _checked_backend(
    *,
    model: str,
    backend: str,
    device: str,
    available: bool,
    reason: str = "",
    fallback_backend: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "model": model,
        "backend": backend,
        "device": device,
        "checked": True,
        "available": bool(available),
    }
    if fallback_backend is not None:
        entry["fallback_backend"] = fallback_backend
    if reason:
        entry["reason"] = reason
    return entry


def _static_backend(
    *,
    model: str,
    backend: str,
    device: str,
    note: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "model": model,
        "backend": backend,
        "device": device,
        "checked": False,
        "available": True,
    }
    if note:
        entry["note"] = note
    return entry


def _unchecked_backend(
    *,
    model: str,
    backend: str,
    device: str,
    note: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "backend": backend,
        "device": device,
        "checked": False,
        "note": note,
    }


def _torch_model_backend(
    *,
    model: str,
    cuda_backend: str,
    cpu_backend: str,
    device_info: DeviceInfo,
) -> dict[str, Any]:
    uses_cuda = device_info.torch_device == "cuda"
    return _static_backend(
        model=model,
        backend=cuda_backend if uses_cuda else cpu_backend,
        device="cuda" if uses_cuda else "cpu",
    )


def _report_selected_models(selected: set[str] | None) -> list[str] | str:
    if selected is None:
        return "all"
    return sorted(selected)


def _selected_or_all(selected: set[str] | None) -> set[str]:
    if selected is not None:
        return set(selected)
    return {
        "catboost",
        "dummy",
        "gandalf",
        "knn-classifier",
        "knn-regressor",
        "lgbm",
        "lr",
        "mlp",
        "rf",
        "sgd-classifier",
        "sgd-regressor",
        "sgd-selector-classifier",
        "sgd-selector-regressor",
        "tabpfn",
        "xgboost",
    }


def backend_availability_report(
    device_info: DeviceInfo,
    selected_models: Iterable[Any] | None = None,
    *,
    run_smoke_checks: bool = True,
) -> dict[str, Any]:
    selected = _selected_model_names(selected_models)
    selected_for_output = _selected_or_all(selected)
    check_tabpfn = run_smoke_checks and _model_selected(selected, "tabpfn")
    check_cuda_backends = run_smoke_checks and device_info.cuda_backends_enabled
    check_knn_classifier = check_cuda_backends and _model_selected(
        selected,
        "knn",
        "knn-classifier",
    )
    check_knn_regressor = check_cuda_backends and _model_selected(
        selected,
        "knn",
        "knn-regressor",
    )
    check_lr = check_cuda_backends and _model_selected(selected, "lr")
    check_elastic = check_cuda_backends and _model_selected(
        selected, "elastic"
    )
    check_sgd_classifier = check_cuda_backends and _model_selected(
        selected,
        "sgd",
        "sgd-select",
        "sgd-selector",
        "sgd-selector-classifier",
        "mbsgd-classifier",
    )
    check_sgd_regressor = check_cuda_backends and _model_selected(
        selected,
        "sgd",
        "sgd-regressor",
        "sgd-selector-regressor",
        "mbsgd-regressor",
    )
    check_catboost = check_cuda_backends and _model_selected(
        selected, "catboost"
    )
    check_xgboost = check_cuda_backends and _model_selected(
        selected, "xgboost"
    )
    run_cuda_backend_checks = any(
        [
            check_knn_classifier,
            check_knn_regressor,
            check_lr,
            check_elastic,
            check_sgd_classifier,
            check_sgd_regressor,
            check_catboost,
            check_xgboost,
        ]
    )
    torch_cuda_available = device_info.torch_cuda_available
    driver_cuda_available = device_info.cuda_driver_available
    report: dict[str, Any] = {
        "device": {
            "requested": device_info.requested,
            "resolved": device_info.resolved,
            "torch_device": device_info.torch_device,
            "cuda_ready": device_info.cuda_backends_enabled,
            "torch_cuda_ready": torch_cuda_available,
            "nvidia_driver_ready": driver_cuda_available,
            "mps_ready": device_info.mps_available,
        },
        "checks": {
            "scope": "all" if selected is None else "selected",
            "smoke_checks": bool(run_smoke_checks),
            "cuda_backend_checks": bool(run_cuda_backend_checks),
        },
        "selected_models": _report_selected_models(selected),
        "models": [],
    }
    models: list[dict[str, Any]] = report["models"]

    if "dummy" in selected_for_output:
        models.append(
            _static_backend(
                model="dummy",
                backend="dummy-cpu",
                device="cpu",
                note="CPU baseline model.",
            )
        )

    if "lgbm" in selected_for_output:
        models.append(
            _static_backend(
                model="lgbm",
                backend="lightgbm-cpu",
                device="cpu",
                note="df-analyze pins lgbm to CPU.",
            )
        )

    if "rf" in selected_for_output:
        models.append(
            _static_backend(
                model="rf",
                backend="lightgbm-rf-cpu",
                device="cpu",
                note="df-analyze pins rf to CPU.",
            )
        )

    if "sgd-classifier" in selected_for_output:
        models.append(
            _static_backend(
                model="sgd",
                backend="sklearn-sgd-classifier",
                device="cpu",
                note="classifier SGD stays on CPU because probability outputs may be required.",
            )
        )

    if "mlp" in selected_for_output:
        models.append(
            _torch_model_backend(
                model="mlp",
                cuda_backend="skorch-cuda",
                cpu_backend="skorch-cpu",
                device_info=device_info,
            )
        )

    if "gandalf" in selected_for_output:
        models.append(
            _torch_model_backend(
                model="gandalf",
                cuda_backend="lightning-cuda",
                cpu_backend=f"lightning-{device_info.lightning_accelerator}",
                device_info=device_info,
            )
        )

    if "tabpfn" in selected_for_output:
        if check_tabpfn:
            tabpfn_available, tabpfn_reason = tabpfn_api_status()
            tabpfn_v25_available, tabpfn_v25_reason = tabpfn_v25_api_status()
            details = []
            if not tabpfn_available and tabpfn_reason:
                details.append(str(tabpfn_reason))
            if not tabpfn_v25_available and tabpfn_v25_reason:
                details.append(str(tabpfn_v25_reason))
            entry = _checked_backend(
                model="tabpfn",
                backend=(
                    "tabpfn-cuda"
                    if device_info.torch_device == "cuda"
                    else "tabpfn-cpu"
                ),
                device="cuda" if device_info.torch_device == "cuda" else "cpu",
                available=tabpfn_available and tabpfn_v25_available,
                reason="; ".join(details),
            )
            entry["fit_smoke"] = False
            models.append(entry)
        else:
            models.append(
                _static_backend(
                    model="tabpfn",
                    backend=(
                        "tabpfn-cuda"
                        if device_info.torch_device == "cuda"
                        else "tabpfn-cpu"
                    ),
                    device="cuda" if device_info.torch_device == "cuda" else "cpu",
                    note="API smoke check was skipped.",
                )
            )

    skipped_note = (
        "CUDA smoke check was skipped."
        if not run_smoke_checks
        else "CUDA backends are not enabled; CPU fallback will be used."
    )

    def fallback_or_unchecked(
        *,
        selected_name: str,
        model: str,
        cuda_backend: str,
        cpu_backend: str,
    ) -> None:
        if selected_name not in selected_for_output:
            return
        if device_info.cuda_backends_enabled and not run_smoke_checks:
            models.append(
                _unchecked_backend(
                    model=model,
                    backend=cuda_backend,
                    device="cuda",
                    note=skipped_note,
                )
            )
            return
        if not device_info.cuda_backends_enabled:
            models.append(
                _static_backend(
                    model=model,
                    backend=cpu_backend,
                    device="cpu",
                    note=skipped_note,
                )
            )

    if not check_knn_classifier and (
        "knn" in selected_for_output or "knn-classifier" in selected_for_output
    ):
        fallback_or_unchecked(
            selected_name=(
                "knn" if "knn" in selected_for_output else "knn-classifier"
            ),
            model="knn",
            cuda_backend="cuml-knn-classifier",
            cpu_backend="sklearn-knn-classifier",
        )

    if not check_knn_regressor and (
        "knn" in selected_for_output or "knn-regressor" in selected_for_output
    ):
        fallback_or_unchecked(
            selected_name=(
                "knn" if "knn" in selected_for_output else "knn-regressor"
            ),
            model="knn",
            cuda_backend="cuml-knn-regressor",
            cpu_backend="sklearn-knn-regressor",
        )

    if not check_lr:
        fallback_or_unchecked(
            selected_name="lr",
            model="lr",
            cuda_backend="cuml-logistic-regression",
            cpu_backend="sklearn-logistic-regression",
        )

    if not check_elastic:
        fallback_or_unchecked(
            selected_name="elastic",
            model="elastic",
            cuda_backend="cuml-elasticnet",
            cpu_backend="sklearn-elasticnet",
        )

    if not check_sgd_regressor:
        fallback_or_unchecked(
            selected_name="sgd-regressor",
            model="sgd",
            cuda_backend="cuml-mbsgd-regressor",
            cpu_backend="sklearn-sgd-regressor",
        )
        fallback_or_unchecked(
            selected_name="sgd-selector-regressor",
            model="sgd-selector",
            cuda_backend="cuml-mbsgd-regressor",
            cpu_backend="sklearn-sgd-regressor",
        )

    if not check_sgd_classifier:
        fallback_or_unchecked(
            selected_name="sgd-selector-classifier",
            model="sgd-selector",
            cuda_backend="cuml-mbsgd-classifier",
            cpu_backend="sklearn-sgd-classifier",
        )

    if not check_catboost:
        fallback_or_unchecked(
            selected_name="catboost",
            model="catboost",
            cuda_backend="catboost-gpu",
            cpu_backend="catboost-cpu",
        )

    if not check_xgboost:
        fallback_or_unchecked(
            selected_name="xgboost",
            model="xgboost",
            cuda_backend="xgboost-cuda",
            cpu_backend="xgboost-cpu",
        )

    if not run_smoke_checks:
        return report

    if check_knn_classifier:
        available, reason = cuml_knn_availability(True)
        models.append(
            _checked_backend(
                model="knn",
                backend="cuml-knn-classifier",
                device="cuda",
                available=available,
                reason=reason,
                fallback_backend="sklearn-knn-classifier",
            )
        )

    if check_knn_regressor:
        available, reason = cuml_knn_availability(False)
        models.append(
            _checked_backend(
                model="knn",
                backend="cuml-knn-regressor",
                device="cuda",
                available=available,
                reason=reason,
                fallback_backend="sklearn-knn-regressor",
            )
        )

    if check_lr:
        lr_available, lr_reason = cuml_logistic_regression_availability()
        models.append(
            _checked_backend(
                model="lr",
                backend="cuml-logistic-regression",
                device="cuda",
                available=lr_available,
                reason=lr_reason,
                fallback_backend="sklearn-logistic-regression",
            )
        )

    if check_elastic:
        elastic_available, elastic_reason = cuml_elasticnet_availability()
        models.append(
            _checked_backend(
                model="elastic",
                backend="cuml-elasticnet",
                device="cuda",
                available=elastic_available,
                reason=elastic_reason,
                fallback_backend="sklearn-elasticnet",
            )
        )

    if check_sgd_classifier:
        mbsgd_cls_available, mbsgd_cls_reason = cuml_mbsgd_classifier_availability()
        models.append(
            _checked_backend(
                model="sgd-selector",
                backend="cuml-mbsgd-classifier",
                device="cuda",
                available=mbsgd_cls_available,
                reason=mbsgd_cls_reason,
                fallback_backend="sklearn-sgd-classifier",
            )
        )

    if check_sgd_regressor:
        mbsgd_reg_available, mbsgd_reg_reason = cuml_mbsgd_regressor_availability()
        for selected_name, model_name in (
            ("sgd-regressor", "sgd"),
            ("sgd-selector-regressor", "sgd-selector"),
        ):
            if selected_name not in selected_for_output:
                continue
            models.append(
                _checked_backend(
                    model=model_name,
                    backend="cuml-mbsgd-regressor",
                    device="cuda",
                    available=mbsgd_reg_available,
                    reason=mbsgd_reg_reason,
                    fallback_backend="sklearn-sgd-regressor",
                )
            )

    if check_catboost:
        catboost_available, catboost_reason = catboost_cuda_availability()
        models.append(
            _checked_backend(
                model="catboost",
                backend="catboost-gpu",
                device="cuda",
                available=catboost_available,
                reason=catboost_reason,
                fallback_backend="catboost-cpu",
            )
        )

    if check_xgboost:
        xgboost_available, xgboost_reason = xgboost_cuda_availability()
        models.append(
            _checked_backend(
                model="xgboost",
                backend="xgboost-cuda",
                device="cuda",
                available=xgboost_available,
                reason=xgboost_reason,
                fallback_backend="xgboost-cpu",
            )
        )
    return report
