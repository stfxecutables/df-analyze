from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import logging
import sys
from math import ceil
from pathlib import Path
from typing import Optional
from warnings import WarningMessage, filterwarnings

import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import KBinsDiscretizer

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


def validate_y(y_encoded: Series, n_splits: int = 5) -> None:
    y_counts = np.bincount(y_encoded)
    min_groups = np.min(y_counts)
    if np.all(n_splits > y_counts):
        raise ValueError(
            f"n_splits={n_splits} cannot be greater than the " "number of members in each class."
        )
    if n_splits > min_groups:
        raise ValueError(
            f"The least populated class in y has only {min_groups}"
            f" members, which is less than n_splits={n_splits}."
        )


def will_work(n_min_cls: int, n_sub: int, n_max: int) -> bool:
    return (n_sub / n_max) * n_min_cls > 1.5


def min_sample(n_min_cls: int, n_max: int) -> int:
    """Why does this work? Not sure at the moment"""
    return ceil(1.5 * n_max / n_min_cls) + 1


def min_subsample(y: Series) -> int:
    """Get smallest possible subsample of y that results in valid internal
    k-folds
    """
    cnts = np.unique(y, return_counts=True)[1]
    n_min_cls = np.min(cnts).item()
    n_max = len(y)
    return min_sample(n_min_cls, n_max)


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


def do_predict(
    dataset: tuple[str, TestDataset]
) -> Optional[
    tuple[Optional[DataFrame], Optional[DataFrame], list[BaseException], list[WarningMessage]]
]:
    filterwarnings("ignore", message="Bins whose width are too small", category=UserWarning)
    dsname, ds = dataset
    if dsname in ["credit-approval_reproduced"]:
        return  # target is constant after dropping NaN
    if dsname in ["internet_usage"]:
        return  # huge multiclass target causes splitting / reduction problems

    mode: EstimationMode = "classify" if ds.is_classification else "regress"
    prepared = ds.prepared(load_cached=True)
    X, X_cat, X_cont = prepared.X, prepared.X_cat, prepared.X_cont
    y = prepared.y

    # TODO: make this a CLI option?
    # make fast
    if mode == "classify":
        strat = y
    else:
        kb = KBinsDiscretizer(n_bins=3, encode="ordinal")
        strat = kb.fit_transform(y.to_numpy().reshape(-1, 1))
    # N = min(1000, len(X))

    if len(X) > 1000:
        if ds.is_classification:
            # this doesn't work for large multiclass
            N = min_subsample(y)
            n_train = max(1000, N)
        else:
            n_train = 1000

        # TODO: improve below loop to just drop undersampled cats
        ss = StratifiedShuffleSplit(n_splits=100, train_size=n_train)
        for idx, _ in ss.split(X, strat):
            X = prepared.X.iloc[idx, :].copy(deep=True)
            y = prepared.y.loc[idx].copy(deep=True)
            X_cat = prepared.X_cat.loc[idx, :].copy(deep=True)
            X_cont = prepared.X_cont.loc[idx, :].copy(deep=True)
            # stop if y is well-sampled for all levels
            if np.unique(y, return_counts=True)[1].min() > 10:
                break
            # otherwise expand train size and try again
            n_train = n_train * 1.025
            if n_train > len(prepared.y):
                break

    try:
        df_cont, df_cat, errs, warns = feature_target_predictions(
            categoricals=X_cat,
            continuous=X_cont,
            target=y,
            mode=mode,
        )
        if len(errs) > 0:
            raise errs[0]
        print_preds(dsname, df_cont, df_cat, mode)
        return df_cont, df_cat, errs, warns
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
    cont, cat = ds.associations(load_cached=False)
    return


def do_associate_cached(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname in ["credit-approval_reproduced"]:  # const targets
        return
    cont, cat = ds.associations(load_cached=True)
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
def test_associate_cached_fast(dataset: tuple[str, TestDataset]) -> None:
    do_associate_cached(dataset)


@med_ds
def test_associate_cached_med(dataset: tuple[str, TestDataset]) -> None:
    do_associate_cached(dataset)


@slow_ds
def test_associate_cached_slow(dataset: tuple[str, TestDataset]) -> None:
    do_associate_cached(dataset)


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
    skip: bool = True
    for dataset in TEST_DATASETS.items():
        dsname, ds = dataset
        # if dsname == "internet_usage":
        #     skip = False
        #     continue
        # if skip:
        #     continue
        if dsname != "nomao":
            continue
        print(f"Starting: {dsname}")
        results = do_predict(dataset)
        print(f"Completed: {dsname}")
        if results is None:
            continue
        df_cont, df_cat, errs, warns = results
        if len(warns) > 0:
            raise ValueError(f"Got warning for {dataset[0]}:\n{warns[0]}")
