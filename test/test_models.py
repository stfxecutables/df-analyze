from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from pandas import DataFrame, Series
from pytest import CaptureFixture
from sklearn.preprocessing import KBinsDiscretizer

from df_analyze.enumerables import ClassifierScorer, RegressorScorer, Scorer
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.models.knn import KNNClassifier, KNNRegressor
from df_analyze.models.lgbm import (
    LightGBMClassifier,
    LightGBMRegressor,
    LightGBMRFClassifier,
    LightGBMRFRegressor,
)
from df_analyze.models.linear import (
    ElasticNetRegressor,
    LRClassifier,
    SGDClassifier,
    SGDRegressor,
)
from df_analyze.models.svm import SVMClassifier, SVMRegressor


def fake_data(
    mode: Literal["classify", "regress"], noise: float = 1.0
) -> tuple[DataFrame, DataFrame, Series, Series]:
    N = 100
    C = 20

    X_cont_tr = np.random.standard_normal([N, C])
    X_cont_test = np.random.standard_normal([N, C])

    cat_sizes = np.random.randint(2, 20, C)
    cats_tr = [np.random.randint(0, c) for c in cat_sizes]
    cats_test = [np.random.randint(0, c) for c in cat_sizes]

    X_cat_tr = np.empty([N, C])
    for i, cat in enumerate(cats_tr):
        X_cat_tr[:, i] = cat

    X_cat_test = np.empty([N, C])
    for i, cat in enumerate(cats_test):
        X_cat_test[:, i] = cat

    df_cat_tr = pd.get_dummies(DataFrame(X_cat_tr))
    df_cat_test = pd.get_dummies(DataFrame(X_cat_test))

    df_cont_tr = DataFrame(X_cont_tr)
    df_cont_test = DataFrame(X_cont_test)

    df_tr = pd.concat([df_cont_tr, df_cat_tr], axis=1)
    df_test = pd.concat([df_cont_test, df_cat_test], axis=1)

    cols = [f"f{i}" for i in range(df_tr.shape[1])]
    df_tr.columns = cols
    df_test.columns = cols

    weights = np.random.uniform(0, 1, 2 * C)
    y_tr = np.dot(df_tr.values, weights) + np.random.normal(0, noise, N)
    y_test = np.dot(df_test.values, weights) + np.random.normal(0, noise, N)

    if mode == "classify":
        encoder = KBinsDiscretizer(n_bins=2, encode="ordinal")
        encoder.fit(y_tr.reshape(-1, 1))
        y_tr = encoder.transform(y_tr.reshape(-1, 1))
        y_test = encoder.transform(y_test.reshape(-1, 1))

    target_tr = Series(np.asarray(y_tr).ravel(), name="target")
    target_test = Series(np.asarray(y_test).ravel(), name="target")

    return df_tr, df_test, target_tr, target_test


def check_basics(model: DfAnalyzeModel, mode: Literal["classify", "regress"]) -> None:
    X_tr, X_test, y_tr, y_test = fake_data(mode)
    model.fit(X_train=X_tr, y_train=y_tr)
    model.predict(X_test)
    model.score(X_test, y_test)


def check_optuna_tune_metric(
    model: DfAnalyzeModel,
    mode: Literal["classify", "regress"],
    metric: Union[ClassifierScorer, RegressorScorer],
) -> tuple[float, ndarray | None]:
    X_tr, X_test, y_tr, y_test = fake_data(mode)
    # metric = ClassifierScorer.default() if is_cls else RegressorScorer.default()
    model = deepcopy(model)
    study = model.htune_optuna(
        X_train=X_tr,
        y_train=y_tr,
        metric=metric,  # type: ignore
        n_trials=8,
    )

    overrides = study.best_params
    model.refit_tuned(X_tr, y_tr, tuned_args=overrides)
    score = model.tuned_score(X_test, y_test)
    if model.is_classifier:
        probs = model.predict_proba(X_test)
        return score, probs
    return score, None


def check_optuna_tune(
    model: DfAnalyzeModel,
    mode: Literal["classify", "regress"],
) -> None:
    is_cls = model.is_classifier
    metrics = ClassifierScorer if is_cls else RegressorScorer
    # metric = ClassifierScorer.default() if is_cls else RegressorScorer.default()
    for metric in metrics:
        check_optuna_tune_metric(model=model, mode=mode, metric=metric)


@pytest.mark.fast
class TestLinear:
    def test_lin_cls(self) -> None:
        model = LRClassifier()
        check_basics(model, "classify")

    def test_lin_reg(self) -> None:
        model = ElasticNetRegressor()
        check_basics(model, "regress")

    def test_lin_cls_tune(self, capsys: CaptureFixture) -> None:
        logging.captureWarnings(capture=True)
        logger = logging.getLogger("py.warnings")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())
        try:
            model = LRClassifier()
            check_optuna_tune(model, "classify")
        except Exception as e:
            raise e
        finally:
            logging.captureWarnings(capture=False)

    def test_lin_reg_tune(self, capsys: CaptureFixture) -> None:
        model = ElasticNetRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestSGDLinear:
    def test_sgd_cls(self) -> None:
        model = SGDClassifier()
        check_basics(model, "classify")

    def test_sgd_reg(self) -> None:
        model = SGDRegressor()
        check_basics(model, "regress")

    def test_sgd_cls_tune(self, capsys: CaptureFixture) -> None:
        logging.captureWarnings(capture=True)
        logger = logging.getLogger("py.warnings")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())
        try:
            model = SGDClassifier()
            check_optuna_tune(model, "classify")
        except Exception as e:
            raise e
        finally:
            logging.captureWarnings(capture=False)

    def test_sgd_reg_tune(self, capsys: CaptureFixture) -> None:
        model = SGDRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestKNN:
    def test_knn_cls(self) -> None:
        model = KNNClassifier()
        check_basics(model, "classify")

    def test_knn_reg(self) -> None:
        model = KNNRegressor()
        check_basics(model, "regress")

    def test_knn_cls_tune(self, capsys: CaptureFixture) -> None:
        model = KNNClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_knn_reg_tune(self, capsys: CaptureFixture) -> None:
        model = KNNRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestSVM:
    def test_svm_cls(self) -> None:
        model = SVMClassifier()
        check_basics(model, "classify")

    def test_svm_reg(self) -> None:
        model = SVMRegressor()
        check_basics(model, "regress")

    def test_svm_cls_tune(self, capsys: CaptureFixture) -> None:
        model = SVMClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_svm_reg_tune(self, capsys: CaptureFixture) -> None:
        model = SVMRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestLightGBM:
    def test_lgbm_cls(self) -> None:
        model = LightGBMClassifier()
        check_basics(model, "classify")

    def test_lgbm_reg(self) -> None:
        model = LightGBMRegressor()
        check_basics(model, "regress")

    def test_lgbm_rf_cls(self) -> None:
        model = LightGBMRFClassifier()
        check_basics(model, "classify")

    def test_lgbm_rf_reg(self) -> None:
        model = LightGBMRFRegressor()
        check_basics(model, "regress")

    def test_lgbm_cls_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_lgbm_reg_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")

    def test_lgbm_rf_cls_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMRFClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_lgbm_rf_reg_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMRFRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")
