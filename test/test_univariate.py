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
from typing import Optional

import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split

from src._types import EstimationMode
from src.analysis.univariate.associate import target_associations
from src.analysis.univariate.predict.predict import feature_target_predictions
from src.preprocessing.prepare import (
    prepare_data,
)
from src.testing.datasets import TEST_DATASETS, TestDataset, fast_ds, med_ds, slow_ds

logging.captureWarnings(capture=True)
logger = logging.getLogger("py.warnings")
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())


def print_preds(
    dsname: str, df_cont: Optional[DataFrame], df_cat: Optional[DataFrame], mode: str
) -> None:
    sorter = "acc" if mode == "classify" else "Var exp"

    print(f"Continuous prediction stats (5-fold, tuned) for {dsname}:")
    if df_cont is not None:
        df_cont = df_cont.sort_values(by=sorter, ascending=False).round(5)
        print(df_cont.to_markdown(tablefmt="simple", floatfmt="0.4f"))

    print(f"Categorical prediction stats (5-fold, tuned) for {dsname}:")
    if df_cat is not None:
        df_cat = df_cat.sort_values(by=sorter, ascending=False).round(5)
        print(df_cat.to_markdown(tablefmt="simple", floatfmt="0.4f"))


def do_predict(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname in ["credit-approval_reproduced"]:
        return  # target is constant after dropping NaN

    mode: EstimationMode = "classify" if ds.is_classification else "regress"
    prepared = ds.prepared(load_cached=True)
    X, X_cat, X_cont = prepared.X, prepared.X_cat, prepared.X_cont
    y = prepared.y

    # TODO: make this a CLI option?
    # make fast
    strat = y if mode == "classify" else None
    N = min(500, len(X))
    if N < len(X):
        if mode == "classify":
            ss = StratifiedShuffleSplit(n_splits=1, train_size=N)
        else:
            ss = ShuffleSplit(n_splits=1, train_size=N)
        idx = next(ss.split(X, y))[0]
        X = X.iloc[idx, :]
        y = y.loc[idx]
        X_cat = X_cat.loc[idx, :]
        X_cont = X_cont.loc[idx, :]

    try:
        df_cont, df_cat = feature_target_predictions(
            categoricals=X_cat,
            continuous=X_cont,
            target=y,
            mode=mode,
        )
        print_preds(dsname, df_cont, df_cat, mode)
    except ValueError as e:
        if dsname in ["credit-approval_reproduced"]:
            message = str(e)
            if "constant after dropping NaNs" not in message:
                raise e
    except Exception as e:
        raise ValueError(f"Failed to make univariate predictions for {dsname}") from e


def do_associate(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname in ["credit-approval_reproduced"]:  # const targets
        return
    prepared = ds.prepared(load_cached=False)
    cont, cat = target_associations(prepared)
    return


@fast_ds
def test_associate_fast(dataset: tuple[str, TestDataset]) -> None:
    do_associate(dataset)


@med_ds
def test_associate_med(dataset: tuple[str, TestDataset]) -> None:
    do_associate(dataset)


@slow_ds
def test_associate_slow(dataset: tuple[str, TestDataset]) -> None:
    do_associate(dataset)


@fast_ds
def test_predict_fast(dataset: tuple[str, TestDataset]) -> None:
    do_predict(dataset)


@med_ds
def test_predict_med(dataset: tuple[str, TestDataset]) -> None:
    do_predict(dataset)


@slow_ds
def test_predict_slow(dataset: tuple[str, TestDataset]) -> None:
    do_predict(dataset)


if __name__ == "__main__":
    # test_associate()
    test_datasets_predict()
