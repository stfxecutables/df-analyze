from __future__ import annotations

# fmt: off
import os
import sys  # isort: skip
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Type

import optuna
from optuna import Study, Trial, create_study
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from numpy import ndarray
from pandas import DataFrame, Series

from df_analyze._constants import SEED
from df_analyze.device import DeviceInfo, cleanup_accelerator_memory
from df_analyze.enumerables import Scorer
from df_analyze.models import gpu as gpu_backends
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.splitting import OmniKFold

_TABPFN_CAT_ARG_NAMES = ("categorical_features_indices", "categorical_feature_indices")
_TABPFN_BACKEND_ENV_VAR = "DF_ANALYZE_TABPFN_BACKEND"
_TABPFN_VERSION_ENV_VAR = "DF_ANALYZE_TABPFN_VERSION"
TABPFN_DEFAULT_VERSION = "auto"
TABPFN_VERSION_CHOICES = ("auto", "v3", "v2_6", "v2_5")
TABPFN_DEFAULT_TUNE_FOLDS = 1
TABPFN_TUNE_FOLD_CHOICES = (1, 3, 5)
_TABPFN_VERSION_ATTRS = {
    "v3": "V3",
    "v2_6": "V2_6",
    "v2_5": "V2_5",
}
_TABPFN_CLIENT_TOKEN_ENV_VARS = (
    "DF_ANALYZE_TABPFN_API_TOKEN",
    "TABPFN_CLIENT_ACCESS_TOKEN",
    "TABPFN_CLIENT_TOKEN",
    "TABPFN_API_TOKEN",
)
_TABPFN_CLIENT_FALLBACK_TOKEN_ENV_VARS = ("TABPFN_TOKEN",)
_TABPFN_AUTO_BACKENDS = {"", "auto"}
_TABPFN_CLIENT_BACKENDS = {"api", "client", "cloud"}
_TABPFN_LOCAL_BACKENDS = {"local", "package", "pytorch"}
_TABPFN_VERSION_ALIASES = {
    "": TABPFN_DEFAULT_VERSION,
    "auto": "auto",
    "default": "auto",
    "latest": "auto",
    "3": "v3",
    "v3": "v3",
    "v_3": "v3",
    "tabpfn3": "v3",
    "tabpfn_3": "v3",
    "2_6": "v2_6",
    "26": "v2_6",
    "v2_6": "v2_6",
    "v26": "v2_6",
    "tabpfn2_6": "v2_6",
    "tabpfn_2_6": "v2_6",
    "2_5": "v2_5",
    "25": "v2_5",
    "v2_5": "v2_5",
    "v25": "v2_5",
    "tabpfn2_5": "v2_5",
    "tabpfn_2_5": "v2_5",
}
_TABPFN_LOCAL_WEIGHT_ERROR_MESSAGE = (
    "Local TabPFN weights are not available. For local CUDA runs, "
    "accept the Prior Labs license at "
    "https://ux.priorlabs.ai/account/licenses and set TABPFN_TOKEN, "
    "or set DF_ANALYZE_TABPFN_BACKEND=local together with "
    "DF_ANALYZE_TABPFN_API_TOKEN in the environment/.env so df-analyze "
    "can pass the token through to the local tabpfn package. In WSL or "
    "headless shells, set TABPFN_NO_BROWSER=1 to skip browser launch. "
    "Alternatively, place the required checkpoint in ~/.cache/tabpfn. "
    "Use DF_ANALYZE_TABPFN_BACKEND=client only when you intentionally "
    "want the remote API backend instead of local GPU execution."
)


class TabPFNLocalWeightsError(RuntimeError):
    """Raised when local TabPFN weights cannot be downloaded or verified."""


def normalize_tabpfn_version(value: str | None) -> str:
    raw = os.environ.get(_TABPFN_VERSION_ENV_VAR, TABPFN_DEFAULT_VERSION)
    requested = raw if value is None else value
    clean = requested.strip().lower().replace("-", "_").replace(".", "_")
    try:
        return _TABPFN_VERSION_ALIASES[clean]
    except KeyError as e:
        choices = ", ".join(TABPFN_VERSION_CHOICES)
        raise ValueError(
            f"Unsupported TabPFN version {requested!r}. Expected one of: {choices}."
        ) from e


def normalize_tabpfn_tune_folds(value: int | str | None) -> int:
    if value is None:
        return TABPFN_DEFAULT_TUNE_FOLDS
    try:
        folds = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Unsupported TabPFN tuning fold count {value!r}. "
            "Expected a positive integer."
        ) from e
    if folds < 1:
        raise ValueError(
            f"Unsupported TabPFN tuning fold count {value!r}. "
            "Expected a positive integer."
        )
    return folds


def _tabpfn_token_env_vars(include_tabpfn_token: bool = False) -> tuple[str, ...]:
    if include_tabpfn_token:
        return (*_TABPFN_CLIENT_TOKEN_ENV_VARS, *_TABPFN_CLIENT_FALLBACK_TOKEN_ENV_VARS)
    return _TABPFN_CLIENT_TOKEN_ENV_VARS


def _tabpfn_backend_request() -> str:
    raw = os.environ.get(_TABPFN_BACKEND_ENV_VAR, "auto")
    requested = raw.strip().lower()
    if requested in _TABPFN_AUTO_BACKENDS:
        return "auto"
    if requested in _TABPFN_CLIENT_BACKENDS:
        return "client"
    if requested in _TABPFN_LOCAL_BACKENDS:
        return "local"
    raise ValueError(
        f"Unsupported {_TABPFN_BACKEND_ENV_VAR}={raw!r}. Expected one of "
        "`auto`, `client`, or `local`."
    )


def _tabpfn_dotenv_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in (Path.cwd() / ".env", ROOT.parent / ".env"):
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def _tabpfn_client_token_from_dotenv(env_vars: Sequence[str]) -> str | None:
    try:
        from dotenv import dotenv_values
    except Exception:
        return None

    for dotenv_path in _tabpfn_dotenv_paths():
        if not dotenv_path.is_file():
            continue
        values = dotenv_values(dotenv_path)
        for env_var in env_vars:
            token = values.get(env_var)
            if token is not None and token.strip() != "":
                return token.strip()
    return None


def _tabpfn_client_token(include_tabpfn_token: bool = False) -> str | None:
    env_vars = _tabpfn_token_env_vars(include_tabpfn_token=include_tabpfn_token)
    for env_var in env_vars:
        token = os.environ.get(env_var)
        if token is not None and token.strip() != "":
            return token.strip()
    return _tabpfn_client_token_from_dotenv(env_vars)


def _use_tabpfn_client_api() -> bool:
    requested = _tabpfn_backend_request()
    if requested == "client":
        return True
    return False


def _configure_tabpfn_client_access_token() -> None:
    include_tabpfn_token = _tabpfn_backend_request() == "client"
    token = _tabpfn_client_token(include_tabpfn_token=include_tabpfn_token)
    if token is None:
        return
    try:
        import tabpfn_client
    except ImportError as e:
        raise ImportError(
            "TabPFN API backend requires a compatible optional `tabpfn-client` "
            "package. Install it or unset the TabPFN API token environment "
            "variable to use the local `tabpfn` package."
        ) from e

    set_access_token = getattr(tabpfn_client, "set_access_token", None)
    if not callable(set_access_token):
        raise RuntimeError(
            "The installed `tabpfn_client` package does not expose "
            "`set_access_token`, so df-analyze cannot configure API access."
        )
    set_access_token(token)


def _configure_tabpfn_local_access_token() -> None:
    if os.environ.get("TABPFN_TOKEN") is not None:
        return
    token = _tabpfn_client_token(include_tabpfn_token=True)
    if token is not None:
        os.environ["TABPFN_TOKEN"] = token


class TabPFNEstimator(DfAnalyzeModel):
    shortname = "tabpfn"
    longname = "TabPFN Estimator"
    timeout_s = 3600

    def __init__(
        self,
        model_args: Optional[Mapping] = None,
        tabpfn_version: str | None = None,
    ) -> None:
        model_args = dict(model_args or {})
        if tabpfn_version is None:
            tabpfn_version = model_args.pop("tabpfn_version", None)
        super().__init__(model_args)
        self.is_classifier = False
        self.fixed_args = dict(device="cpu")
        self._categorical_feature_indices: list[int] = []
        self._tabpfn_version = normalize_tabpfn_version(tabpfn_version)
        self._tune_folds = TABPFN_DEFAULT_TUNE_FOLDS
        self._resolved_tabpfn_version: str | None = None

    def set_device_info(self, device_info: DeviceInfo) -> None:
        super().set_device_info(device_info)
        self.fixed_args["device"] = (
            "cuda" if device_info.torch_device == "cuda" else "cpu"
        )

    def backend_name(self, tuned_args: Optional[Mapping[str, Any]] = None) -> str:
        version_label = self.model_version()
        if _use_tabpfn_client_api():
            return f"tabpfn-{version_label}-client"
        device = "cuda" if self.fixed_args.get("device") == "cuda" else "cpu"
        return f"tabpfn-{version_label}-{device}"

    def uses_accelerated_backend(
        self, tuned_args: Optional[Mapping[str, Any]] = None
    ) -> bool:
        if _use_tabpfn_client_api():
            return False
        return self.fixed_args.get("device") == "cuda"

    def cuda_tuning_trial_budget(self) -> Optional[int]:
        if not self.uses_accelerated_backend():
            return None
        return 24 if self.is_classifier else 12

    def model_version(self) -> str:
        if self._resolved_tabpfn_version is not None:
            return self._resolved_tabpfn_version
        if self._tabpfn_version == "auto":
            return "v3"
        return self._tabpfn_version

    def set_tabpfn_version(self, version: str | None) -> None:
        self._tabpfn_version = normalize_tabpfn_version(version)
        self._resolved_tabpfn_version = None

    def set_tune_folds(self, n_folds: int | str | None) -> None:
        self._tune_folds = normalize_tabpfn_tune_folds(n_folds)

    def set_categorical_feature_indices(self, indices: Sequence[int]) -> None:
        self._categorical_feature_indices = [int(idx) for idx in indices]

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[Type[Any], dict[str, Any]]:
        return self._tabpfn_cls(), full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        args: dict[str, str | float | int] = {
            "n_estimators": trial.suggest_categorical("n_estimators", [6, 8]),
            "average_before_softmax": trial.suggest_categorical(
                "average_before_softmax", [False, True]
            ),
        }
        if self.is_classifier:
            args["softmax_temperature"] = trial.suggest_float(
                "softmax_temperature", 0.7, 1.1
            )
            args["balance_probabilities"] = trial.suggest_categorical(
                "balance_probabilities", [False, True]
            )
        return args

    @staticmethod
    def _raise_local_weight_error(e: Exception) -> None:
        message = str(e).lower()
        exception_name = e.__class__.__name__.lower()
        markers = (
            "failed to download model",
            "tabpfnlicenseerror",
            "license not yet accepted",
            "license acceptance",
            "one-time license acceptance",
            "accept-license",
            "model weights",
        )
        if not any(
            marker in message or marker in exception_name for marker in markers
        ):
            return
        raise TabPFNLocalWeightsError(_TABPFN_LOCAL_WEIGHT_ERROR_MESSAGE) from None

    def _tabpfn_cls(self) -> Type[Any]:
        if _use_tabpfn_client_api():
            _configure_tabpfn_client_access_token()
            try:
                from tabpfn_client import TabPFNClassifier as ClientTabPFNClassifier
                from tabpfn_client import TabPFNRegressor as ClientTabPFNRegressor
            except ImportError as e:
                raise ImportError(
                    "TabPFN API backend requires a compatible optional "
                    "`tabpfn-client` package. Install it before selecting the "
                    "client backend."
                ) from e
            return ClientTabPFNClassifier if self.is_classifier else ClientTabPFNRegressor

        _configure_tabpfn_local_access_token()
        try:
            from tabpfn import TabPFNClassifier as PriorTabPFNClassifier
            from tabpfn import TabPFNRegressor as PriorTabPFNRegressor
        except ImportError as e:
            raise ImportError(
                "TabPFN support requires the optional `tabpfn` package. Install the "
                "project dependencies or add `tabpfn` before selecting model `tabpfn`."
            ) from e
        return PriorTabPFNClassifier if self.is_classifier else PriorTabPFNRegressor

    def _tabpfn_model_version(self, requested_version: str) -> Any:
        if _use_tabpfn_client_api():
            try:
                from tabpfn_client.constants import ModelVersion
            except ImportError as e:
                raise ImportError(
                    "TabPFN API backend requires "
                    "`tabpfn_client.constants.ModelVersion`."
                ) from e
        else:
            try:
                from tabpfn.constants import ModelVersion
            except ImportError as e:
                raise ImportError(
                    "Explicit TabPFN version selection requires "
                    "`tabpfn.constants.ModelVersion`."
                ) from e
        attr = _TABPFN_VERSION_ATTRS[requested_version]
        try:
            return getattr(ModelVersion, attr)
        except AttributeError as e:
            package = "tabpfn-client" if _use_tabpfn_client_api() else "tabpfn"
            raise RuntimeError(
                f"The installed `{package}` package does not expose "
                f"`ModelVersion.{attr}`, so df-analyze cannot guarantee that "
                f"TabPFN {requested_version} will be used. Install a TabPFN "
                "release that supports the requested model version."
            ) from e

    @staticmethod
    def _tabpfn_package_major(package: str) -> int | None:
        try:
            raw = version(package)
        except (PackageNotFoundError, ValueError):
            return None
        try:
            return int(raw.split(".", maxsplit=1)[0])
        except (TypeError, ValueError):
            return None

    def _allow_default_v3_constructor(self, use_client_api: bool) -> bool:
        package = "tabpfn-client" if use_client_api else "tabpfn"
        major = self._tabpfn_package_major(package)
        return major is not None and major >= 8

    @classmethod
    def _version_factory(cls, model_cls: Type[Any]) -> Callable[..., Any]:
        factory = getattr(model_cls, "create_default_for_version", None)
        if factory is None:
            factory = getattr(model_cls, "get_default_for_version", None)
        if factory is None:
            raise AttributeError(
                f"{model_cls.__name__} does not expose a TabPFN version factory. "
                "df-analyze refuses to instantiate an explicit TabPFN version from "
                "package defaults."
            )
        return factory

    @staticmethod
    def _signature_parameters(factory: Callable[..., Any]) -> dict[str, Any] | None:
        from inspect import signature

        try:
            return dict(signature(factory).parameters)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _accepts_var_kwargs(cls, factory: Callable[..., Any]) -> bool:
        from inspect import Parameter

        parameters = cls._signature_parameters(factory)
        if parameters is None:
            return False
        if any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values()):
            return True
        return False

    @classmethod
    def _factory_kwargs(
        cls, factory: Callable[..., Any], args: Mapping[str, Any]
    ) -> dict[str, Any]:
        parameters = cls._signature_parameters(factory)
        if parameters is None or cls._accepts_var_kwargs(factory):
            return dict(args)
        return {name: value for name, value in args.items() if name in parameters}

    @classmethod
    def _callable_arg_name(cls, factory: Callable[..., Any]) -> str | None:
        parameters = cls._signature_parameters(factory)
        names = set(parameters or {})
        for name in _TABPFN_CAT_ARG_NAMES:
            if name in names:
                return name
        if cls._accepts_var_kwargs(factory):
            return _TABPFN_CAT_ARG_NAMES[0]
        return None

    @staticmethod
    def _estimator_param_names(estimator: Any) -> set[str] | None:
        get_params = getattr(estimator, "get_params", None)
        if get_params is None:
            return None
        try:
            return set(get_params(deep=True))
        except Exception:
            return None

    @classmethod
    def _estimator_arg_name(cls, estimator: Any) -> str | None:
        params = cls._estimator_param_names(estimator)
        if params is not None:
            for name in _TABPFN_CAT_ARG_NAMES:
                if name in params:
                    return name
        set_params = getattr(estimator, "set_params", None)
        if set_params is None:
            return None
        return cls._callable_arg_name(set_params)

    @classmethod
    def _set_estimator_params(
        cls, estimator: Any, model_cls: Type[Any], args: Mapping[str, Any]
    ) -> None:
        if len(args) == 0:
            return
        set_params = getattr(estimator, "set_params", None)
        if set_params is None:
            names = ", ".join(sorted(args))
            raise TypeError(
                f"{model_cls.__name__} does not support setting args after "
                f"factory creation: {names}"
            )

        param_names = cls._estimator_param_names(estimator)
        if param_names is not None:
            args = {name: value for name, value in args.items() if name in param_names}
        elif not cls._accepts_var_kwargs(set_params):
            parameters = cls._signature_parameters(set_params) or {}
            args = {name: value for name, value in args.items() if name in parameters}
        if len(args) > 0:
            set_params(**args)

    @classmethod
    def _set_estimator_categorical_indices(
        cls, estimator: Any, indices: Sequence[int]
    ) -> None:
        if len(indices) == 0:
            return
        set_params = getattr(estimator, "set_params", None)
        if set_params is None:
            raise TypeError(
                "TabPFN estimator does not support categorical feature indices."
            )

        preferred = cls._estimator_arg_name(estimator)
        candidates = [preferred] if preferred is not None else []
        candidates.extend(
            name for name in _TABPFN_CAT_ARG_NAMES if name not in candidates
        )
        errors: list[Exception] = []
        for name in candidates:
            try:
                set_params(**{name: list(indices)})
                return
            except (TypeError, ValueError) as e:
                errors.append(e)
        raise TypeError(
            "TabPFN estimator rejected both categorical feature index parameter "
            "spellings."
        ) from errors[-1]

    def _new_default_estimator(
        self,
        model_cls: Type[Any],
        args: Mapping[str, Any],
        use_client_api: bool,
    ) -> Any:
        args = dict(args)
        categorical_arg = None
        if len(self._categorical_feature_indices) > 0:
            categorical_arg = self._callable_arg_name(model_cls)
            if categorical_arg is not None:
                args[categorical_arg] = list(self._categorical_feature_indices)

        init_args = self._factory_kwargs(model_cls, args)
        estimator = model_cls(**init_args)
        remaining = {k: v for k, v in args.items() if k not in init_args}
        self._set_estimator_params(estimator, model_cls, remaining)
        if categorical_arg is None and not use_client_api:
            self._set_estimator_categorical_indices(
                estimator, self._categorical_feature_indices
            )
        return estimator

    def _new_estimator(self, extra_args: Optional[Mapping[str, Any]] = None) -> Any:
        use_client_api = _use_tabpfn_client_api()
        args = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **(extra_args or {}),
        }
        if use_client_api:
            args.pop("device", None)
        else:
            args["device"] = self.fixed_args["device"]
        model_cls = self._tabpfn_cls()
        requested_version = (
            "v3" if self._tabpfn_version == "auto" else self._tabpfn_version
        )
        self._resolved_tabpfn_version = requested_version

        try:
            version = self._tabpfn_model_version(requested_version)
        except (ImportError, RuntimeError):
            if requested_version != "v3" or not self._allow_default_v3_constructor(
                use_client_api
            ):
                raise
            return self._new_default_estimator(model_cls, args, use_client_api)

        # TabPFN weights may have upstream license restrictions; keep explicit
        # version selection for publication and distribution reviews.
        try:
            factory = self._version_factory(model_cls)
        except AttributeError:
            if requested_version != "v3" or not self._allow_default_v3_constructor(
                use_client_api
            ):
                raise
            return self._new_default_estimator(model_cls, args, use_client_api)

        categorical_arg = None
        if len(self._categorical_feature_indices) > 0:
            categorical_arg = self._callable_arg_name(factory)
            if categorical_arg is not None:
                args[categorical_arg] = list(self._categorical_feature_indices)

        factory_args = self._factory_kwargs(factory, args)
        estimator = factory(version, **factory_args)
        remaining = {k: v for k, v in args.items() if k not in factory_args}
        self._set_estimator_params(estimator, model_cls, remaining)
        if categorical_arg is None and not use_client_api:
            self._set_estimator_categorical_indices(
                estimator, self._categorical_feature_indices
            )
        return estimator

    def fit(self, X_train: DataFrame, y_train: Series) -> None:
        if self.model is None:
            self.model = self._new_estimator()
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            self._raise_local_weight_error(e)
            raise

    def refit_tuned(
        self,
        X: DataFrame,
        y: Series,
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> None:
        self._reset_tuned_model_for_refit()
        self.tuned_model = self._new_estimator(tuned_args)
        try:
            self.tuned_model.fit(X, y)
        except Exception as e:
            self._raise_local_weight_error(e)
            raise

    def predict(self, X: DataFrame) -> ndarray:
        if self.model is None:
            raise RuntimeError("Need to call `model.fit()` before calling `.predict()`")
        return gpu_backends.as_numpy(self.model.predict(X))

    def tuned_predict(self, X: DataFrame) -> ndarray:
        if self.tuned_model is None:
            raise RuntimeError(
                "Need to call `model.tune()` before calling `.tuned_predict()`"
            )
        return gpu_backends.as_numpy(self.tuned_model.predict(X))

    def predict_proba_untuned(self, X: DataFrame) -> ndarray:
        if not self.is_classifier:
            raise ValueError("Cannot get probabilities for a regression model.")
        if self.model is None:
            raise RuntimeError(
                "Need to call `model.fit()` before calling `.predict_proba_untuned()`"
            )
        return gpu_backends.as_numpy(self.model.predict_proba(X))

    def predict_proba(self, X: DataFrame) -> ndarray:
        if not self.is_classifier:
            raise ValueError("Cannot get probabilities for a regression model.")
        if self.tuned_model is None:
            raise RuntimeError("Need to tune estimator before calling `.predict_proba()`")
        return gpu_backends.as_numpy(self.tuned_model.predict_proba(X))

    def optuna_objective(
        self,
        X_train: DataFrame,
        y_train: Series,
        g_train: Optional[Series],
        metric: Scorer,
        n_folds: int | None = None,
    ) -> Callable[[Trial], float]:
        tune_folds = (
            self._tune_folds
            if n_folds is None
            else normalize_tabpfn_tune_folds(n_folds)
        )
        split_count = max(3, tune_folds)

        def objective(trial: Trial) -> float:
            kf = OmniKFold(
                n_splits=split_count,
                is_classification=self.is_classifier,
                grouped=g_train is not None,
                labels=None,
                warn_on_fallback=False,
                df_analyze_phase="TabPFN tuning split",
            )
            splits = kf.split(X_train=X_train, y_train=y_train, g_train=g_train)[0]
            scores = []
            for step, (idx_train, idx_test) in enumerate(splits[:tune_folds]):
                X_tr = X_train.iloc[idx_train]
                y_tr = y_train.iloc[idx_train]
                X_test = X_train.iloc[idx_test]
                y_test = y_train.iloc[idx_test]

                try:
                    estimator = self._new_estimator(self.optuna_args(trial))
                    estimator.fit(X_tr, y_tr)
                except Exception as e:
                    self._raise_local_weight_error(e)
                    raise
                preds = gpu_backends.as_numpy(estimator.predict(X_test))
                score = metric.tuning_score(y_test, preds)
                scores.append(score)
                trial.report(float(sum(scores) / len(scores)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(sum(scores) / len(scores))

        return objective

    def _preflight_sample(
        self,
        X_train: DataFrame,
        y_train: Series,
        max_rows: int = 16,
    ) -> tuple[DataFrame, Series]:
        if len(X_train) <= max_rows:
            return X_train, y_train
        if not self.is_classifier:
            return X_train.iloc[:max_rows], y_train.iloc[:max_rows]

        classes = y_train.dropna().drop_duplicates()
        if len(classes) <= 1:
            return X_train.iloc[:max_rows], y_train.iloc[:max_rows]

        selected: list[int] = []
        y_values = y_train.to_numpy()
        for target in classes.iloc[:max_rows]:
            for idx, value in enumerate(y_values):
                if value == target:
                    selected.append(idx)
                    break

        if len(selected) < max_rows:
            selected_set = set(selected)
            selected.extend(
                idx
                for idx in range(len(X_train))
                if idx not in selected_set
            )

        selected = selected[:max_rows]
        return X_train.iloc[selected], y_train.iloc[selected]

    def _preflight_local_weights(self, X_train: DataFrame, y_train: Series) -> None:
        if _use_tabpfn_client_api():
            return

        was_set = "TABPFN_NO_BROWSER" in os.environ
        previous = os.environ.get("TABPFN_NO_BROWSER")
        os.environ["TABPFN_NO_BROWSER"] = "1"
        estimator = None
        try:
            estimator = self._new_estimator()
            X_preflight, y_preflight = self._preflight_sample(X_train, y_train)
            estimator.fit(X_preflight, y_preflight)
        except Exception as e:
            self._raise_local_weight_error(e)
            raise
        finally:
            if was_set and previous is not None:
                os.environ["TABPFN_NO_BROWSER"] = previous
            else:
                os.environ.pop("TABPFN_NO_BROWSER", None)
            estimator = None
            cleanup_accelerator_memory(self.device_info)

    def htune_optuna(
        self,
        X_train: DataFrame,
        y_train: Series,
        g_train: Optional[Series],
        metric: Scorer,
        n_trials: int = 100,
        n_jobs: int = -1,
        verbosity: int = optuna.logging.ERROR,
    ) -> Study:
        if self.tuned_args is not None:
            raise RuntimeError(
                f"Model {self.__class__.__name__} has already been tuned with Optuna"
            )

        self._preflight_local_weights(X_train, y_train)
        direction = "maximize" if metric.higher_is_better() else "minimize"
        study = create_study(direction=direction, sampler=TPESampler(seed=SEED))
        optuna.logging.set_verbosity(verbosity)
        n_jobs = 1
        study.optimize(
            self.optuna_objective(
                X_train=X_train,
                y_train=y_train,
                g_train=g_train,
                metric=metric,
            ),
            n_trials=self.effective_tuning_n_trials(n_trials),
            n_jobs=n_jobs,
            callbacks=[
                lambda _study, _trial: cleanup_accelerator_memory(self.device_info)
            ],
            timeout=self.__class__.timeout_s,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        cleanup_accelerator_memory(self.device_info)
        self.tuned_args = study.best_params
        self.refit_tuned(X=X_train, y=y_train, tuned_args=self.tuned_args)
        return study


class TabPFNClassifier(TabPFNEstimator):
    shortname = "tabpfn"
    longname = "TabPFN Classifier"

    def __init__(
        self,
        model_args: Optional[Mapping] = None,
        tabpfn_version: str | None = None,
    ) -> None:
        super().__init__(model_args, tabpfn_version=tabpfn_version)
        self.is_classifier = True


class TabPFNRegressor(TabPFNEstimator):
    shortname = "tabpfn"
    longname = "TabPFN Regressor"

    def __init__(
        self,
        model_args: Optional[Mapping] = None,
        tabpfn_version: str | None = None,
    ) -> None:
        super().__init__(model_args, tabpfn_version=tabpfn_version)
        self.is_classifier = False
