from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from typing import Literal
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split as tt_split
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler

from src._constants import TESTDATA
from src.enumerables import NanHandling
from src.preprocessing.cleaning import (
    clean_regression_target,
    drop_id_cols,
    encode_categoricals,
    encode_target,
    handle_continuous_nans,
    normalize,
    remove_timestamps,
)

CLASSIFICATIONS = TESTDATA / "classification"
REGRESSIONS = TESTDATA / "regression"
ALL = sorted(list(CLASSIFICATIONS.glob("*")) + list(REGRESSIONS.glob("*")))


class TestDataset:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.is_classification: bool = root.parent.name == "classification"
        self.datapath = root / f"{root.name}.parquet"
        self.types = root / "types.csv"
        df = pd.read_csv(self.types)
        df = df.loc[df["type"] == "categorical"]
        df = df.loc[df["feature_name"] != "target"]
        self.categoricals = df["feature_name"].to_list()
        self.is_multiclass = False
        if self.is_classification:
            df = pd.read_parquet(self.datapath)
            num_classes = len(np.unique(df["target"]))
            self.is_multiclass = num_classes > 2

    def load(self) -> DataFrame:
        return pd.read_parquet(self.datapath)

    def train_test_split(
        self, test_size: float = 0.2
    ) -> tuple[DataFrame, DataFrame, Series, Series, int]:
        df = self.load()
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            df, dropped = remove_timestamps(df, target="target")
            cats = list(set(self.categoricals).difference(dropped))
            df, dropped = drop_id_cols(df, "target")
            cats = list(set(cats).difference(dropped))
            df = handle_continuous_nans(df, target="target", cat_cols=cats, nans=NanHandling.Median)
            df = encode_categoricals(df, target="target", categoricals=cats)[0]
            df = normalize(df, "target")
            df = df.copy(deep=True)

            X = df.drop(columns="target")
            y = df["target"]

            if self.is_classification:
                X, y = encode_target(X, y)
            else:
                X, y = clean_regression_target(X, y)

            strat = y
            if not self.is_classification:
                yy = y.to_numpy().reshape(-1, 1)
                strat = KBinsDiscretizer(n_bins=3, encode="ordinal").fit_transform(yy)
                strat = strat.ravel()
        X_tr, X_test, y_tr, y_test = tt_split(X, y, test_size=test_size, stratify=strat)
        num_classes = len(np.unique(y)) if self.is_classification else 1
        return X_tr, X_test, y_tr, y_test, num_classes


__UNSORTED: list[tuple[str, TestDataset]] = [(p.name, TestDataset(p)) for p in ALL]

TEST_DATASETS: dict[str, TestDataset] = dict(sorted(__UNSORTED, key=lambda p: p[1].load().shape[0]))


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
