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
from sklearn.model_selection import train_test_split

from src._types import EstimationMode
from src.analysis.univariate.associate import feature_target_stats, target_associations
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


def test_datasets_predict() -> None:
    for dsname, ds in TEST_DATASETS.items():
        # if dsname in ["elder", "forest_fires"]:
        # if dsname in [
        #     "elder",
        #     "forest_fires",
        #     "community_crime",
        #     "credit_approval",
        #     "abalone",
        #     "student_dropout",
        #     "wine_quality",
        # ]:
        #     continue
        if dsname != "news_popularity":
            continue
        df = ds.load()
        mode: EstimationMode = "classify" if ds.is_classification else "regress"
        print("=" * 120)
        print(f"Cleaning data for {dsname} {df.shape} ({mode})")
        cats = ds.categoricals
        df, target, cat_cols = prepare_data(
            df=df, target="target", categoricals=cats, is_classification=ds.is_classification
        )

        # TODO: make this a CLI option?
        # make fast
        strat = target if mode == "classify" else None
        N = min(500, len(df))
        if N < len(df):
            df = train_test_split(df, train_size=N, stratify=strat)[0]
            target = target[df.index]

        print(f"Making univariate predictions for {dsname} {df.shape}")
        df_cont, df_cat = feature_target_predictions(
            categoricals=df[cat_cols],
            continuous=df.drop(columns=cats, errors="ignore"),
            target=target,
            mode=mode,
        )
        sorter = "acc" if mode == "classify" else "Var exp"

        print(f"Continuous prediction stats (5-fold, tuned) for {dsname}:")
        if df_cont is not None:
            df_cont = df_cont.sort_values(by=sorter, ascending=False).round(5)
            print(df_cont.to_markdown(tablefmt="simple", floatfmt="0.4f"))

        print(f"Categorical prediction stats (5-fold, tuned) for {dsname}:")
        if df_cat is not None:
            df_cat = df_cat.sort_values(by=sorter, ascending=False).round(5)
            print(df_cat.to_markdown(tablefmt="simple", floatfmt="0.4f"))


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


def do_associate(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname in ["credit-approval_reproduced"]:  # const targets
        return
    prepared = ds.prepared(load_cached=False)
    cont, cat = target_associations(prepared)


@fast_ds
def test_associate_fast(dataset: tuple[str, TestDataset]) -> None:
    do_associate(dataset)


@med_ds
def test_associate_med(dataset: tuple[str, TestDataset]) -> None:
    do_associate(dataset)


@slow_ds
def test_associate_slow(dataset: tuple[str, TestDataset]) -> None:
    do_associate(dataset)


if __name__ == "__main__":
    # test_associate()
    test_datasets_predict()
