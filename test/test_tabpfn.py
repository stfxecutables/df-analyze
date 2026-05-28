from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from df_analyze.device import device_info_from_resolved
from df_analyze.enumerables import (
    ClassifierScorer,
    DfAnalyzeClassifier,
    DfAnalyzeRegressor,
)
from df_analyze.hypertune import (
    TABPFN_UNKNOWN_CATEGORY,
    _get_tabpfn_splits,
    _tabpfn_feature_matrices,
    _tabpfn_selected_cols,
)
from df_analyze.models import gpu as gpu_backends
from df_analyze.models.tabpfn import (
    TabPFNClassifier,
    TabPFNEstimator,
    TabPFNLocalWeightsError,
    TabPFNRegressor,
)
from df_analyze.preprocessing.prepare import PreparedData

TABPFN_CLIENT_ENV_VARS = (
    "DF_ANALYZE_TABPFN_BACKEND",
    "DF_ANALYZE_TABPFN_API_TOKEN",
    "TABPFN_CLIENT_ACCESS_TOKEN",
    "TABPFN_CLIENT_TOKEN",
    "TABPFN_API_TOKEN",
    "TABPFN_TOKEN",
    "TABPFN_NO_BROWSER",
)


class FakeModelVersion:
    V3 = "v3"
    V2_6 = "v2.6"
    V2_5 = "v2.5"


class FakeModelVersionWithoutV25:
    V3 = "v3"
    V2_6 = "v2.6"


class FakeModelVersionWithoutV3:
    V2_6 = "v2.6"
    V2_5 = "v2.5"


class FakeTabPFNBase:
    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.version: str | None = FakeModelVersion.V3
        self.fit_shape: tuple[int, int] | None = None

    @classmethod
    def create_default_for_version(cls, version: str) -> FakeTabPFNBase:
        estimator = cls()
        estimator.version = version
        return estimator

    def set_params(self, **params: Any) -> FakeTabPFNBase:
        self.params.update(params)
        return self

    def fit(self, X: DataFrame, y: Series) -> FakeTabPFNBase:
        self.fit_shape = X.shape
        return self


class FakeArrayWithGet:
    def __init__(self, value: Any) -> None:
        self.value = value

    def get(self) -> Any:
        return self.value


class FakeTabPFNClassifier(FakeTabPFNBase):
    def predict(self, X: DataFrame) -> FakeArrayWithGet:
        return FakeArrayWithGet(np.arange(len(X)) % 2)

    def predict_proba(self, X: DataFrame) -> FakeArrayWithGet:
        probs = np.zeros((len(X), 2), dtype=float)
        probs[:, 0] = 0.75
        probs[:, 1] = 0.25
        return FakeArrayWithGet(probs)


class FakeTabPFNDefaultOnlyClassifier(FakeTabPFNClassifier):
    create_default_for_version = None


class FakeTabPFNCountingClassifier(FakeTabPFNClassifier):
    fit_count = 0

    def fit(self, X: DataFrame, y: Series) -> FakeTabPFNCountingClassifier:
        type(self).fit_count += 1
        return super().fit(X, y)


class FakeTabPFNRegressor(FakeTabPFNBase):
    def predict(self, X: DataFrame) -> FakeArrayWithGet:
        return FakeArrayWithGet(np.full(len(X), 1.5))


class FakeTabPFNWithoutFactory:
    pass


class FakeTabPFNSingularCategorical(FakeTabPFNClassifier):
    @classmethod
    def create_default_for_version(
        cls, version: str, categorical_feature_indices: list[int] | None = None
    ) -> FakeTabPFNSingularCategorical:
        estimator = cls()
        estimator.version = version
        estimator.params["categorical_feature_indices"] = categorical_feature_indices
        return estimator


class FakeTabPFNPluralCategorical(FakeTabPFNClassifier):
    @classmethod
    def create_default_for_version(
        cls, version: str, categorical_features_indices: list[int] | None = None
    ) -> FakeTabPFNPluralCategorical:
        estimator = cls()
        estimator.version = version
        estimator.params["categorical_features_indices"] = categorical_features_indices
        return estimator


class FakeTabPFNVarKeywordFactory(FakeTabPFNClassifier):
    @classmethod
    def create_default_for_version(
        cls, version: str, **overrides: Any
    ) -> FakeTabPFNVarKeywordFactory:
        if "categorical_feature_indices" in overrides:
            raise TypeError("unexpected singular categorical arg")
        estimator = cls(**overrides)
        estimator.version = version
        return estimator


class FakeTabPFNMissingWeights(FakeTabPFNClassifier):
    def fit(self, X: DataFrame, y: Series) -> FakeTabPFNMissingWeights:
        raise RuntimeError(
            "Failed to download model to /tmp/tabpfn.ckpt!\n\n"
            "For offline usage, please download the model manually."
        )


class FakeTabPFNBadPreflight(FakeTabPFNClassifier):
    def fit(self, X: DataFrame, y: Series) -> FakeTabPFNBadPreflight:
        raise ValueError("bad TabPFN input shape")


class FakeTabPFNRequiresTwoClasses(FakeTabPFNClassifier):
    y_seen: list[Any] = []

    def fit(self, X: DataFrame, y: Series) -> FakeTabPFNRequiresTwoClasses:
        type(self).y_seen = y.to_list()
        if y.nunique() < 2:
            raise ValueError("preflight sample only has one class")
        self.fit_shape = X.shape
        return self


class TabPFNLicenseError(Exception):
    pass


class FakeTabPFNLicenseNotAccepted(FakeTabPFNClassifier):
    no_browser_seen: str | None = None

    def fit(self, X: DataFrame, y: Series) -> FakeTabPFNLicenseNotAccepted:
        type(self).no_browser_seen = os.environ.get("TABPFN_NO_BROWSER")
        raise TabPFNLicenseError(
            "License not yet accepted. Please complete the acceptance form at\n"
            "https://ux.priorlabs.ai/accept-license?hf_repo_id=tabpfn_3 "
            "and try again."
        )


def install_fake_tabpfn(
    monkeypatch: pytest.MonkeyPatch,
    classifier_cls: type = FakeTabPFNClassifier,
    regressor_cls: type = FakeTabPFNRegressor,
    model_version_cls: type = FakeModelVersion,
) -> None:
    tabpfn = ModuleType("tabpfn")
    tabpfn.TabPFNClassifier = classifier_cls  # type: ignore[attr-defined]
    tabpfn.TabPFNRegressor = regressor_cls  # type: ignore[attr-defined]

    constants = ModuleType("tabpfn.constants")
    constants.ModelVersion = model_version_cls  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "tabpfn", tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)


def install_fake_tabpfn_client(
    monkeypatch: pytest.MonkeyPatch,
    classifier_cls: type = FakeTabPFNClassifier,
    regressor_cls: type = FakeTabPFNRegressor,
    model_version_cls: type = FakeModelVersion,
) -> list[str]:
    tokens: list[str] = []
    tabpfn_client = ModuleType("tabpfn_client")
    tabpfn_client.__path__ = []  # type: ignore[attr-defined]
    tabpfn_client.TabPFNClassifier = classifier_cls  # type: ignore[attr-defined]
    tabpfn_client.TabPFNRegressor = regressor_cls  # type: ignore[attr-defined]

    def set_access_token(token: str) -> None:
        tokens.append(token)

    tabpfn_client.set_access_token = set_access_token  # type: ignore[attr-defined]

    constants = ModuleType("tabpfn_client.constants")
    constants.ModelVersion = model_version_cls  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "tabpfn_client", tabpfn_client)
    monkeypatch.setitem(sys.modules, "tabpfn_client.constants", constants)
    return tokens


@pytest.fixture(autouse=True)
def clear_tabpfn_client_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in TABPFN_CLIENT_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("DF_ANALYZE_TABPFN_BACKEND", "local")


def tabular_data() -> tuple[DataFrame, Series, Series]:
    X = pd.DataFrame(
        {
            "a": np.linspace(0.0, 1.0, 8),
            "b": np.linspace(1.0, 2.0, 8),
        }
    )
    y_cls = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="target")
    y_reg = pd.Series(np.linspace(0.0, 1.0, 8), name="target")
    return X, y_cls, y_reg


def prepared_for_tabpfn_matrix_tests() -> tuple[PreparedData, PreparedData]:
    y_train = pd.Series([0, 1, 0, 1, 0, 1], name="target")
    y_test = pd.Series([0, 1, 0, 1], name="target")

    train = PreparedData(
        X=pd.DataFrame(
            {
                "age": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "age_NAN": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                "city_Boston": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "city_Halifax": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                "city_Toronto": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            }
        ),
        X_cont=pd.DataFrame({"age": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}),
        X_cat=pd.DataFrame(
            {"city": ["Boston", "Halifax", "Toronto", "Boston", "Halifax", "Toronto"]}
        ),
        y=y_train,
        groups=None,
        is_classification=True,
        validate=False,
    )
    test = PreparedData(
        X=pd.DataFrame(
            {
                "age": [20.0, 21.0, 22.0, 23.0],
                "age_NAN": [1.0, 0.0, 0.0, 1.0],
                "city_Boston": [0.0, 1.0, 0.0, 0.0],
                "city_Halifax": [1.0, 0.0, 0.0, 1.0],
                "city_Toronto": [0.0, 0.0, 1.0, 0.0],
            }
        ),
        X_cont=pd.DataFrame({"age": [0.0, 0.33, 0.66, 1.0]}),
        X_cat=pd.DataFrame({"city": ["Halifax", "Boston", "Toronto", "Halifax"]}),
        y=y_test,
        groups=None,
        is_classification=True,
        validate=False,
    )
    return train, test


def prepared_with_overlapping_feature_names() -> tuple[PreparedData, PreparedData]:
    y_train = pd.Series([0, 1, 0, 1], name="target")
    y_test = pd.Series([0, 1], name="target")

    train = PreparedData(
        X=pd.DataFrame(
            {
                "age": [10.0, 11.0, 12.0, 13.0],
                "age_group": [1.0, 1.0, 2.0, 2.0],
                "city_Boston": [1.0, 0.0, 1.0, 0.0],
                "city_Halifax": [0.0, 1.0, 0.0, 1.0],
            }
        ),
        X_cont=pd.DataFrame(
            {
                "age": [0.0, 0.33, 0.66, 1.0],
                "age_group": [0.0, 0.0, 1.0, 1.0],
            }
        ),
        X_cat=pd.DataFrame({"city": ["Boston", "Halifax", "Boston", "Halifax"]}),
        y=y_train,
        groups=None,
        is_classification=True,
        validate=False,
    )
    test = PreparedData(
        X=pd.DataFrame(
            {
                "age": [20.0, 21.0],
                "age_group": [3.0, 3.0],
                "city_Boston": [0.0, 1.0],
                "city_Halifax": [1.0, 0.0],
            }
        ),
        X_cont=pd.DataFrame(
            {
                "age": [0.0, 1.0],
                "age_group": [1.0, 1.0],
            }
        ),
        X_cat=pd.DataFrame({"city": ["Montreal", "Boston"]}),
        y=y_test,
        groups=None,
        is_classification=True,
        validate=False,
    )
    return train, test


@pytest.mark.fast
def test_tabpfn_registered_but_not_default() -> None:
    assert DfAnalyzeClassifier.TabPFN.get_model() is TabPFNClassifier
    assert DfAnalyzeRegressor.TabPFN.get_model() is TabPFNRegressor
    assert DfAnalyzeClassifier.TabPFN.value in DfAnalyzeClassifier.choices()
    assert DfAnalyzeRegressor.TabPFN.value in DfAnalyzeRegressor.choices()
    assert DfAnalyzeClassifier.TabPFN.value not in DfAnalyzeClassifier.defaults()
    assert DfAnalyzeRegressor.TabPFN.value not in DfAnalyzeRegressor.defaults()


@pytest.mark.fast
def test_tabpfn_classifier_uses_v3_by_default_and_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model.fit(X, y_cls)
    preds = model.predict(X)
    probs = model.predict_proba_untuned(X)

    assert isinstance(model.model, FakeTabPFNClassifier)
    assert model.model.version == FakeModelVersion.V3
    assert model.model.params["device"] == "cuda"
    assert model.model_version() == "v3"
    assert model.backend_name() == "tabpfn-v3-cuda"
    assert model.model.fit_shape == X.shape
    assert preds.shape == (len(X),)
    assert probs.shape == (len(X), 2)


@pytest.mark.fast
def test_tabpfn_explicit_v25_is_still_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier(tabpfn_version="v2_5")
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model.fit(X, y_cls)

    assert isinstance(model.model, FakeTabPFNClassifier)
    assert model.model.version == FakeModelVersion.V2_5
    assert model.model.params["device"] == "cuda"
    assert model.model_version() == "v2_5"
    assert model.backend_name() == "tabpfn-v2_5-cuda"


@pytest.mark.fast
def test_tabpfn_v3_can_use_package_default_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch, model_version_cls=FakeModelVersionWithoutV3)
    monkeypatch.setattr(
        TabPFNEstimator,
        "_tabpfn_package_major",
        staticmethod(lambda package: 8),
    )
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model.fit(X, y_cls)

    assert isinstance(model.model, FakeTabPFNClassifier)
    assert model.model.version == FakeModelVersion.V3
    assert model.model.params["device"] == "cuda"
    assert model.model_version() == "v3"
    assert model.backend_name() == "tabpfn-v3-cuda"


@pytest.mark.fast
def test_tabpfn_v3_missing_factory_uses_default_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch, classifier_cls=FakeTabPFNDefaultOnlyClassifier)
    monkeypatch.setattr(
        TabPFNEstimator,
        "_tabpfn_package_major",
        staticmethod(lambda package: 8),
    )
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model.fit(X, y_cls)

    assert isinstance(model.model, FakeTabPFNDefaultOnlyClassifier)
    assert model.model.version == FakeModelVersion.V3
    assert model.model.params["device"] == "cuda"
    assert model.model_version() == "v3"


@pytest.mark.fast
def test_tabpfn_regressor_falls_back_to_cpu_for_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch)
    X, _, y_reg = tabular_data()
    model = TabPFNRegressor()
    model.set_device_info(device_info_from_resolved("auto", "mps", mps_is_available=True))

    model.fit(X, y_reg)
    preds = model.predict(X)

    assert isinstance(model.model, FakeTabPFNRegressor)
    assert model.model.version == FakeModelVersion.V3
    assert model.model.params["device"] == "cpu"
    assert model.backend_name() == "tabpfn-v3-cpu"
    assert preds.shape == (len(X),)


@pytest.mark.fast
def test_tabpfn_uses_cpu_when_cuda_backend_is_not_torch_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()
    model.set_device_info(
        device_info_from_resolved(
            "auto",
            "cuda",
            torch_cuda_available=False,
        )
    )

    model.fit(X, y_cls)

    assert isinstance(model.model, FakeTabPFNClassifier)
    assert model.model.params["device"] == "cpu"
    assert model.backend_name() == "tabpfn-v3-cpu"


@pytest.mark.fast
def test_tabpfn_client_backend_sets_access_token_and_omits_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = install_fake_tabpfn_client(monkeypatch)
    monkeypatch.setenv("DF_ANALYZE_TABPFN_BACKEND", "client")
    monkeypatch.setenv("DF_ANALYZE_TABPFN_API_TOKEN", "test-token")
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    model.fit(X, y_cls)

    assert tokens == ["test-token"]
    assert isinstance(model.model, FakeTabPFNClassifier)
    assert model.model.version == FakeModelVersion.V3
    assert "device" not in model.model.params
    assert model.backend_name() == "tabpfn-v3-client"


@pytest.mark.fast
def test_tabpfn_auto_backend_stays_local_with_client_token_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch)
    tokens = install_fake_tabpfn_client(monkeypatch)
    monkeypatch.delenv("DF_ANALYZE_TABPFN_BACKEND", raising=False)
    monkeypatch.setenv("TABPFN_CLIENT_TOKEN", "auto-token")
    X, _, y_reg = tabular_data()
    model = TabPFNRegressor()

    model.fit(X, y_reg)

    assert tokens == []
    assert isinstance(model.model, FakeTabPFNRegressor)
    assert model.model.version == FakeModelVersion.V3
    assert os.environ["TABPFN_TOKEN"] == "auto-token"
    assert model.backend_name() == "tabpfn-v3-cpu"


@pytest.mark.fast
def test_tabpfn_auto_backend_stays_local_with_dotenv_client_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    pytest.importorskip("dotenv")
    install_fake_tabpfn(monkeypatch)
    tokens = install_fake_tabpfn_client(monkeypatch)
    monkeypatch.delenv("DF_ANALYZE_TABPFN_BACKEND", raising=False)
    (tmp_path / ".env").write_text(
        "DF_ANALYZE_TABPFN_API_TOKEN=dotenv-token\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()

    model.fit(X, y_cls)

    assert tokens == []
    assert isinstance(model.model, FakeTabPFNClassifier)
    assert os.environ["TABPFN_TOKEN"] == "dotenv-token"
    assert model.backend_name() == "tabpfn-v3-cpu"


@pytest.mark.fast
def test_tabpfn_local_backend_passes_dotenv_token_to_local_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    pytest.importorskip("dotenv")
    install_fake_tabpfn(monkeypatch)
    (tmp_path / ".env").write_text(
        "DF_ANALYZE_TABPFN_API_TOKEN=dotenv-token\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()

    model.fit(X, y_cls)

    assert os.environ["TABPFN_TOKEN"] == "dotenv-token"
    assert isinstance(model.model, FakeTabPFNClassifier)


@pytest.mark.fast
def test_tabpfn_refuses_installed_package_without_v25_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch, model_version_cls=FakeModelVersionWithoutV25)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier(tabpfn_version="v2_5")

    with pytest.raises(RuntimeError, match="TabPFN v2_5"):
        model.fit(X, y_cls)


@pytest.mark.fast
@pytest.mark.parametrize(
    "classifier_cls", [FakeTabPFNMissingWeights, FakeTabPFNLicenseNotAccepted]
)
def test_tabpfn_fit_rewrites_local_weight_errors(
    monkeypatch: pytest.MonkeyPatch,
    classifier_cls: type,
) -> None:
    install_fake_tabpfn(monkeypatch, classifier_cls=classifier_cls)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()

    with pytest.raises(TabPFNLocalWeightsError, match="Local TabPFN weights"):
        model.fit(X, y_cls)


@pytest.mark.fast
@pytest.mark.parametrize(
    "classifier_cls", [FakeTabPFNMissingWeights, FakeTabPFNLicenseNotAccepted]
)
def test_tabpfn_tuning_rewrites_local_weight_errors(
    monkeypatch: pytest.MonkeyPatch,
    classifier_cls: type,
) -> None:
    import optuna

    install_fake_tabpfn(monkeypatch, classifier_cls=classifier_cls)
    X = pd.DataFrame(
        {
            "a": np.linspace(0.0, 1.0, 40),
            "b": np.linspace(1.0, 2.0, 40),
        }
    )
    y_cls = pd.Series([0, 1] * 20, name="target")
    model = TabPFNClassifier()
    objective = model.optuna_objective(
        X_train=X,
        y_train=y_cls,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_folds=2,
    )
    trial = optuna.trial.FixedTrial(
        {
            "n_estimators": 6,
            "average_before_softmax": False,
            "softmax_temperature": 0.9,
            "balance_probabilities": False,
        }
    )

    with pytest.raises(TabPFNLocalWeightsError, match="Local TabPFN weights"):
        objective(trial)


@pytest.mark.fast
def test_tabpfn_tuning_preflights_local_weights_before_study(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from df_analyze.models import tabpfn as tabpfn_module

    FakeTabPFNLicenseNotAccepted.no_browser_seen = None
    install_fake_tabpfn(monkeypatch, classifier_cls=FakeTabPFNLicenseNotAccepted)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()

    def fake_create_study(direction: str) -> Any:
        assert direction == "maximize"
        raise AssertionError("TabPFN preflight should fail before creating a study.")

    monkeypatch.setattr(tabpfn_module, "create_study", fake_create_study)

    with pytest.raises(TabPFNLocalWeightsError, match="Local TabPFN weights"):
        model.htune_optuna(
            X_train=X,
            y_train=y_cls,
            g_train=None,
            metric=ClassifierScorer.Accuracy,
            n_trials=99,
            n_jobs=-1,
        )

    assert FakeTabPFNLicenseNotAccepted.no_browser_seen == "1"
    assert "TABPFN_NO_BROWSER" not in os.environ


@pytest.mark.fast
def test_tabpfn_preflight_reraises_non_weight_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch, classifier_cls=FakeTabPFNBadPreflight)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()

    with pytest.raises(ValueError, match="bad TabPFN input shape"):
        model.htune_optuna(
            X_train=X,
            y_train=y_cls,
            g_train=None,
            metric=ClassifierScorer.Accuracy,
            n_trials=1,
            n_jobs=-1,
        )

    assert "TABPFN_NO_BROWSER" not in os.environ


@pytest.mark.fast
def test_tabpfn_preflight_keeps_multiple_classes_for_sorted_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeTabPFNRequiresTwoClasses.y_seen = []
    install_fake_tabpfn(monkeypatch, classifier_cls=FakeTabPFNRequiresTwoClasses)
    X = DataFrame({"x": np.arange(40)})
    y = Series([0] * 20 + [1] * 20)
    model = TabPFNClassifier()

    model._preflight_local_weights(X, y)

    assert set(FakeTabPFNRequiresTwoClasses.y_seen) == {0, 1}
    assert len(FakeTabPFNRequiresTwoClasses.y_seen) == 16
    assert "TABPFN_NO_BROWSER" not in os.environ


@pytest.mark.fast
def test_tabpfn_v25_api_status_checks_lightweight_factory_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch)

    available, reason = gpu_backends.tabpfn_v25_api_status()

    assert available is True
    assert reason == ""
    assert gpu_backends.tabpfn_v25_api_available() is True


@pytest.mark.fast
def test_tabpfn_v25_api_status_requires_classifier_and_regressor_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch, regressor_cls=FakeTabPFNWithoutFactory)

    available, reason = gpu_backends.tabpfn_v25_api_status()

    assert available is False
    assert "FakeTabPFNWithoutFactory" in reason
    assert "create_default_for_version" in reason


@pytest.mark.fast
def test_tabpfn_tuning_uses_requested_trials_and_single_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from df_analyze.models import tabpfn as tabpfn_module

    class FakeStudy:
        best_params: dict[str, Any] = {}
        best_value = 1.0

        def __init__(self) -> None:
            self.optimize_kwargs: dict[str, Any] = {}

        def optimize(self, objective: Any, **kwargs: Any) -> None:
            self.optimize_kwargs = kwargs

    study = FakeStudy()

    def fake_create_study(direction: str) -> FakeStudy:
        assert direction == "maximize"
        return study

    refit_calls: list[dict[str, Any]] = []

    def fake_refit_tuned(**kwargs: Any) -> None:
        refit_calls.append(kwargs)

    monkeypatch.setattr(tabpfn_module, "create_study", fake_create_study)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()
    monkeypatch.setattr(model, "_preflight_local_weights", lambda *args: None)
    monkeypatch.setattr(model, "refit_tuned", fake_refit_tuned)

    returned = model.htune_optuna(
        X_train=X,
        y_train=y_cls,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_trials=99,
        n_jobs=-1,
    )

    assert returned is study
    assert study.optimize_kwargs["n_trials"] == 99
    assert study.optimize_kwargs["n_jobs"] == 1
    assert model.tuned_args == {}
    assert len(refit_calls) == 1


@pytest.mark.fast
def test_tabpfn_cuda_tuning_forces_single_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from df_analyze.models import tabpfn as tabpfn_module

    seen: dict[str, Any] = {}

    class FakeStudy:
        best_params: dict[str, Any] = {}
        best_value = 1.0

        def __init__(self) -> None:
            self.optimize_kwargs: dict[str, Any] = {}

        def optimize(self, objective: Any, **kwargs: Any) -> None:
            self.optimize_kwargs = kwargs

    def fake_tpe_sampler(*, seed: int) -> str:
        seen["seed"] = seed
        return "seeded-sampler"

    def fake_create_study(**kwargs: Any) -> FakeStudy:
        seen.update(kwargs)
        return study

    study = FakeStudy()
    monkeypatch.setattr(tabpfn_module, "TPESampler", fake_tpe_sampler)
    monkeypatch.setattr(tabpfn_module, "create_study", fake_create_study)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()
    model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

    def fake_preflight(*args: Any) -> None:
        seen["preflight_calls"] = int(seen.get("preflight_calls", 0)) + 1

    monkeypatch.setattr(model, "_preflight_local_weights", fake_preflight)
    monkeypatch.setattr(model, "refit_tuned", lambda **kwargs: None)

    returned = model.htune_optuna(
        X_train=X,
        y_train=y_cls,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_trials=99,
        n_jobs=-1,
    )

    assert returned is study
    assert seen["seed"] == tabpfn_module.SEED
    assert seen["sampler"] == "seeded-sampler"
    assert seen["preflight_calls"] == 1
    assert study.optimize_kwargs["n_trials"] == 24
    assert study.optimize_kwargs["n_jobs"] == 1


@pytest.mark.fast
def test_tabpfn_cuda_tuning_trial_budget_is_task_specific() -> None:
    cuda_info = device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True)

    classifier = TabPFNClassifier()
    classifier.set_device_info(cuda_info)
    assert classifier.effective_tuning_n_trials(100) == 24
    assert classifier.effective_tuning_n_trials(16) == 16

    regressor = TabPFNRegressor()
    regressor.set_device_info(cuda_info)
    assert regressor.effective_tuning_n_trials(100) == 12
    assert regressor.effective_tuning_n_trials(8) == 8

    cpu_classifier = TabPFNClassifier()
    cpu_classifier.set_device_info(device_info_from_resolved("cpu", "cpu"))
    assert cpu_classifier.effective_tuning_n_trials(100) == 100


@pytest.mark.fast
def test_tabpfn_objective_uses_configured_tuning_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTrial:
        def __init__(self) -> None:
            self.reports: list[int] = []

        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
            return choices[0]

        def suggest_float(self, name: str, low: float, high: float) -> float:
            return low

        def report(self, value: float, step: int) -> None:
            self.reports.append(step)

        def should_prune(self) -> bool:
            return False

    FakeTabPFNCountingClassifier.fit_count = 0
    install_fake_tabpfn(monkeypatch, classifier_cls=FakeTabPFNCountingClassifier)
    X = DataFrame({"x": np.arange(30)})
    y = Series([0, 1] * 15)
    trial = FakeTrial()
    model = TabPFNClassifier()
    model.set_tune_folds(3)

    objective = model.optuna_objective(
        X_train=X,
        y_train=y,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
    )
    objective(trial)  # type: ignore[arg-type]

    assert FakeTabPFNCountingClassifier.fit_count == 3
    assert trial.reports == [0, 1, 2]


@pytest.mark.fast
def test_tabpfn_splits_use_raw_continuous_and_raw_categoricals() -> None:
    train, test = prepared_for_tabpfn_matrix_tests()

    X_train, X_test, cat_indices = _get_tabpfn_splits(
        train, test, selection="none", cols=slice(None)
    )

    assert X_train.columns.to_list() == ["age", "age_NAN", "city"]
    assert X_test.columns.to_list() == ["age", "age_NAN", "city"]
    assert X_train["age"].to_list() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    assert X_test["age"].to_list() == [20.0, 21.0, 22.0, 23.0]
    assert X_train["age_NAN"].to_list() == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert X_test["age_NAN"].to_list() == [1.0, 0.0, 0.0, 1.0]
    assert str(X_train["city"].dtype) == "category"
    assert str(X_test["city"].dtype) == "category"
    assert cat_indices == [2]


@pytest.mark.fast
def test_tabpfn_feature_matrices_do_not_use_normalized_continuous() -> None:
    train, test = prepared_for_tabpfn_matrix_tests()

    X_train, X_test, cat_cols = _tabpfn_feature_matrices(train, test)

    assert X_train["age"].to_list() == train.X["age"].to_list()
    assert X_test["age"].to_list() == test.X["age"].to_list()
    assert X_train["age"].to_list() != train.X_cont["age"].to_list()
    assert X_test["age"].to_list() != test.X_cont["age"].to_list()
    assert X_train["age_NAN"].to_list() == train.X["age_NAN"].to_list()
    assert X_test["age_NAN"].to_list() == test.X["age_NAN"].to_list()
    assert cat_cols == ["city"]


@pytest.mark.fast
def test_tabpfn_categorical_levels_are_train_only() -> None:
    train, test = prepared_with_overlapping_feature_names()

    X_train, X_test, cat_cols = _tabpfn_feature_matrices(train, test)

    assert cat_cols == ["city"]
    assert X_train["city"].cat.categories.to_list() == [
        "Boston",
        "Halifax",
        TABPFN_UNKNOWN_CATEGORY,
    ]
    assert X_test["city"].iloc[0] == TABPFN_UNKNOWN_CATEGORY
    assert X_test["city"].iloc[1] == "Boston"
    assert not X_test["city"].isna().any()


@pytest.mark.fast
def test_tabpfn_splits_map_one_hot_selection_to_base_categorical() -> None:
    train, test = prepared_for_tabpfn_matrix_tests()

    X_train, X_test, cat_indices = _get_tabpfn_splits(
        train, test, selection="embed_lgbm", cols=["city_Halifax"]
    )

    assert X_train.columns.to_list() == ["city"]
    assert X_test.columns.to_list() == ["city"]
    assert X_train["city"].to_list() == ["Boston", "Halifax", "Toronto"] * 2
    assert cat_indices == [0]


@pytest.mark.fast
def test_tabpfn_splits_keep_mixed_raw_selection() -> None:
    train, test = prepared_for_tabpfn_matrix_tests()

    X_train, X_test, cat_indices = _get_tabpfn_splits(
        train, test, selection="embed_lgbm", cols=["age", "city_Halifax"]
    )

    assert X_train.columns.to_list() == ["age", "city"]
    assert X_test.columns.to_list() == ["age", "city"]
    assert X_train["age"].to_list() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    assert X_train["city"].to_list() == ["Boston", "Halifax", "Toronto"] * 2
    assert cat_indices == [1]


@pytest.mark.fast
def test_tabpfn_splits_keep_selected_missing_indicator() -> None:
    train, test = prepared_for_tabpfn_matrix_tests()

    X_train, X_test, cat_indices = _get_tabpfn_splits(
        train, test, selection="embed_lgbm", cols=["age_NAN"]
    )

    assert X_train.columns.to_list() == ["age_NAN"]
    assert X_test.columns.to_list() == ["age_NAN"]
    assert X_train["age_NAN"].to_list() == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert X_test["age_NAN"].to_list() == [1.0, 0.0, 0.0, 1.0]
    assert cat_indices == []


@pytest.mark.fast
def test_tabpfn_splits_keep_selected_missing_indicator_with_categorical() -> None:
    train, test = prepared_for_tabpfn_matrix_tests()

    X_train, X_test, cat_indices = _get_tabpfn_splits(
        train, test, selection="embed_lgbm", cols=["age_NAN", "city_Halifax"]
    )

    assert X_train.columns.to_list() == ["age_NAN", "city"]
    assert X_test.columns.to_list() == ["age_NAN", "city"]
    assert X_train["age_NAN"].to_list() == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert X_train["city"].to_list() == ["Boston", "Halifax", "Toronto"] * 2
    assert cat_indices == [1]


@pytest.mark.fast
def test_tabpfn_selected_cols_preserve_original_matrix_order() -> None:
    selected = _tabpfn_selected_cols(
        all_cols=["age_group", "age", "age_NAN", "city"],
        cat_cols=["city"],
        cols=["city_Halifax", "age_NAN", "age"],
    )

    assert selected == ["age", "age_NAN", "city"]


@pytest.mark.fast
def test_tabpfn_exact_selection_does_not_match_column_prefixes() -> None:
    train, test = prepared_with_overlapping_feature_names()

    X_train, X_test, cat_indices = _get_tabpfn_splits(
        train, test, selection="embed_lgbm", cols=["age"]
    )

    assert X_train.columns.to_list() == ["age"]
    assert X_test.columns.to_list() == ["age"]
    assert X_train["age"].to_list() == [10.0, 11.0, 12.0, 13.0]
    assert cat_indices == []


@pytest.mark.fast
def test_tabpfn_categorical_indices_follow_final_exact_subset_order() -> None:
    train, test = prepared_with_overlapping_feature_names()

    X_train, X_test, cat_indices = _get_tabpfn_splits(
        train, test, selection="embed_lgbm", cols=["city_Halifax", "age_group"]
    )

    assert X_train.columns.to_list() == ["age_group", "city"]
    assert X_test.columns.to_list() == ["age_group", "city"]
    assert cat_indices == [1]


@pytest.mark.fast
def test_tabpfn_passes_categorical_indices_to_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()
    model.set_categorical_feature_indices([1])

    model.fit(X, y_cls)

    assert isinstance(model.model, FakeTabPFNClassifier)
    assert model.model.params["categorical_features_indices"] == [1]


@pytest.mark.fast
def test_tabpfn_var_keyword_factory_uses_real_plural_categorical_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch, classifier_cls=FakeTabPFNVarKeywordFactory)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()
    model.set_categorical_feature_indices([1])

    model.fit(X, y_cls)

    assert isinstance(model.model, FakeTabPFNVarKeywordFactory)
    assert model.model.params["categorical_features_indices"] == [1]
    assert "categorical_feature_indices" not in model.model.params


@pytest.mark.fast
def test_tabpfn_prefers_supported_singular_categorical_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch, classifier_cls=FakeTabPFNSingularCategorical)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()
    model.set_categorical_feature_indices([1])

    model.fit(X, y_cls)

    assert isinstance(model.model, FakeTabPFNSingularCategorical)
    assert model.model.params["categorical_feature_indices"] == [1]


@pytest.mark.fast
def test_tabpfn_accepts_supported_plural_categorical_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tabpfn(monkeypatch, classifier_cls=FakeTabPFNPluralCategorical)
    X, y_cls, _ = tabular_data()
    model = TabPFNClassifier()
    model.set_categorical_feature_indices([1])

    model.fit(X, y_cls)

    assert isinstance(model.model, FakeTabPFNPluralCategorical)
    assert model.model.params["categorical_features_indices"] == [1]


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("DF_ANALYZE_REAL_TABPFN_TESTS") != "1",
    reason="Set DF_ANALYZE_REAL_TABPFN_TESTS=1 to inspect the installed TabPFN API.",
)
def test_real_tabpfn_factory_api_available() -> None:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    assert ModelVersion.V2_5 is not None
    assert callable(getattr(TabPFNClassifier, "create_default_for_version", None))
    assert callable(getattr(TabPFNRegressor, "create_default_for_version", None))


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("DF_ANALYZE_REAL_TABPFN_TESTS") != "1",
    reason=(
        "Set DF_ANALYZE_REAL_TABPFN_TESTS=1 to run a real TabPFN tiny fit "
        "when model weights are already available."
    ),
)
def test_real_tabpfn_tiny_fit_smoke() -> None:
    device = os.environ.get("DF_ANALYZE_REAL_TABPFN_DEVICE", "cpu").lower()
    if device not in {"cpu", "cuda"}:
        pytest.skip(
            "Real TabPFN smoke test supports "
            "DF_ANALYZE_REAL_TABPFN_DEVICE=cpu/cuda."
        )
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            pytest.skip(
                "CUDA was requested for the real TabPFN smoke test but is unavailable."
            )

    X = DataFrame(
        {
            "x0": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
            "x1": [1.0, 0.9, 0.8, 0.2, 0.1, 0.0],
        }
    )
    y = Series([0, 0, 0, 1, 1, 1], name="target")
    model = TabPFNClassifier()
    model.set_device_info(device_info_from_resolved(device, device))

    try:
        model.fit(X, y)
    except Exception as e:
        message = str(e).lower()
        if any(term in message for term in ("download", "weight", "checkpoint", "cache")):
            pytest.skip(f"Real TabPFN weights are not available in this environment: {e}")
        raise

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert model.backend_name() == f"tabpfn-v3-{device}"
