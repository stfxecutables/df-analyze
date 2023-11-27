from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing_extensions import Literal

from src.models.base import DfAnalyzeModel
from src.models.knn import KNNClassifier, KNNRegressor
from src.models.lgbm import (
    LightGBMClassifier,
    LightGBMRegressor,
    LightGBMRFClassifier,
    LightGBMRFRegressor,
)
from src.models.linear import ElasticNetRegressor, LRClassifier
from src.models.svm import SVMClassifier, SVMRegressor


def fake_data(mode: Literal["classify", "regress"]) -> tuple[DataFrame, Series]:
    N = 100
    C = 20
    y = np.random.standard_exponential(N) if mode == "regress" else np.random.randint(0, 2, N)
    target = Series(y, name="target")

    X_cont = np.random.standard_normal([N, C])
    cat_sizes = np.random.randint(2, 20, C)
    cats = [np.random.randint(0, c) for c in cat_sizes]
    X_cat = np.empty([N, C])
    for i, cat in enumerate(cats):
        X_cat[:, i] = cat
    df_cat = pd.get_dummies(DataFrame(X_cat))
    df_cont = DataFrame(X_cont)
    df = pd.concat([df_cont, df_cat], axis=1)
    cols = [f"f{i}" for i in range(df.shape[1])]
    df.columns = cols
    return df, target


def check_basics(model: DfAnalyzeModel, mode: Literal["classify", "regress"]) -> None:
    X_tr, y_tr = fake_data(mode)
    X_test, y_test = fake_data(mode)
    model.fit(X_train=X_tr, y_train=y_tr)
    model.predict(X_test)
    model.score(X_test, y_test)


def check_optuna_tune(model: DfAnalyzeModel, mode: Literal["classify", "regress"]) -> None:
    X_tr, y_tr = fake_data(mode)
    model.htune_optuna(X_train=X_tr, y_train=y_tr, n_trials=20)


class TestLinear:
    def test_lin_cls(self) -> None:
        model = LRClassifier()
        check_basics(model, "classify")

    def test_lin_reg(self) -> None:
        model = ElasticNetRegressor()
        check_basics(model, "regress")

    def test_lin_cls_tune(self) -> None:
        model = LRClassifier()
        check_optuna_tune(model, "classify")

    def test_lin_reg_tune(self) -> None:
        model = ElasticNetRegressor()
        check_optuna_tune(model, "regress")


class TestKNN:
    def test_knn_cls(self) -> None:
        model = KNNClassifier()
        check_basics(model, "classify")

    def test_knn_reg(self) -> None:
        model = KNNRegressor()
        check_basics(model, "regress")

    def test_knn_cls_tune(self) -> None:
        model = KNNClassifier()
        check_optuna_tune(model, "classify")

    def test_knn_reg_tune(self) -> None:
        model = KNNRegressor()
        check_optuna_tune(model, "regress")


class TestSVM:
    def test_svm_cls(self) -> None:
        model = SVMClassifier()
        check_basics(model, "classify")

    def test_svm_reg(self) -> None:
        model = SVMRegressor()
        check_basics(model, "regress")

    def test_svm_cls_tune(self) -> None:
        model = SVMClassifier()
        check_optuna_tune(model, "classify")

    def test_svm_reg_tune(self) -> None:
        model = SVMRegressor()
        check_optuna_tune(model, "regress")


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

    def test_lgbm_cls_tune(self) -> None:
        model = LightGBMClassifier()
        check_optuna_tune(model, "classify")

    def test_lgbm_reg_tune(self) -> None:
        model = LightGBMRegressor()
        check_optuna_tune(model, "regress")

    def test_lgbm_rf_cls_tune(self) -> None:
        model = LightGBMRFClassifier()
        check_optuna_tune(model, "classify")

    def test_lgbm_rf_reg_tune(self) -> None:
        model = LightGBMRFRegressor()
        check_optuna_tune(model, "regress")
