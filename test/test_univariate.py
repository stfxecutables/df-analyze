from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import logging
import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame, Series
from pytest import CaptureFixture

from src._types import EstimationMode
from src.analysis.univariate.associate import feature_target_stats
from src.analysis.univariate.predict.predict import feature_target_predictions
from src.preprocessing.cleaning import encode_target
from src.testing.datasets import TEST_DATASETS

logging.captureWarnings(capture=True)
logger = logging.getLogger("py.warnings")
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())


def test_datasets_predict() -> None:
    for dsname, ds in TEST_DATASETS.items():
        df = ds.load()
        cats = ds.categoricals
        target = df["target"]
        df = df.drop(columns="target")
        df, target = encode_target(df, target)

        mode: EstimationMode = "classify" if ds.is_classification else "regress"
        print(f"Making univariate predictions for {dsname} {df.shape}")
        df_cont, df_cat = feature_target_predictions(
            categoricals=df[cats], continuous=df.drop(columns=cats), target=target, mode=mode
        )
        print(f"Continous prediction stats for {dsname}:")
        print(df_cont)
        print(f"Categorical prediction stats for {dsname}:")
        print(df_cat)


def test_random_associate() -> None:
    cat_sizes = np.random.randint(1, 20, 30)

    y_cont = Series(np.random.uniform(0, 1, [250]), name="target")
    y_cat = Series(np.random.randint(0, 6, 250), name="target")
    X_cont = np.random.standard_normal([250, 30])
    X_cat = np.full([250, 30], fill_value=np.nan)
    for i, catsize in enumerate(cat_sizes):
        X_cat[:, i] = np.random.randint(0, catsize, X_cat.shape[0])

    cont_names = [f"r{i}" for i in range(X_cont.shape[1])]
    cat_names = [f"c{i}" for i in range(X_cont.shape[1])]
    df_cont = DataFrame(data=X_cont, columns=cont_names)
    df_cat = DataFrame(data=X_cat, columns=cat_names)

    df_cont_stats, df_cat_stats = feature_target_stats(
        continuous=df_cont, categoricals=df_cat, target=y_cat, mode="classify"
    )
    level_idx = df_cat_stats.index.to_series().apply(lambda s: "." in s)
    cat_level_stats = df_cat_stats[level_idx]
    cat_stats = df_cat_stats[~level_idx]
    print("Continuous stats:\n", df_cont_stats)
    print("Categorical target level stats:\n", cat_level_stats)
    print("Categorical full target stats:\n", cat_stats)

    df_cont_stats, df_cat_stats = feature_target_stats(
        continuous=df_cont, categoricals=df_cat, target=y_cont, mode="regress"
    )
    level_idx = df_cat_stats.index.to_series().apply(lambda s: "." in s)
    cat_level_stats = df_cat_stats[level_idx]
    cat_stats = df_cat_stats[~level_idx]
    print("Continuous stats:\n", df_cont_stats)
    print("Categorical target level stats:\n", cat_level_stats)
    print("Categorical full target stats:\n", cat_stats)


if __name__ == "__main__":
    # test_associate()
    test_datasets_predict()