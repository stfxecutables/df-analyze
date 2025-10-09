from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
ROOT2 = Path(__file__).resolve().parent.parent / "src"  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
sys.path.append(str(ROOT2))  # isort: skip
# fmt: on


import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series
from pytest import CaptureFixture
from sklearn.utils.validation import check_X_y
from tqdm import tqdm

from df_analyze.splitting import ApproximateStratifiedGroupSplit, OmniKFold
from df_analyze.testing.datasets import (
    FAST_INSPECTION,
    TestDataset,
    fast_ds,
    med_ds,
    slow_ds,
)


def min_examples(
    n_grp: int,
    n_cls: int,
    n_samp: int = 200,
    n_min_per_g: int = 2,
    n_min_per_targ_cls: int = 20,
) -> tuple[Series, Series]:
    g = []
    for i in range(n_grp):
        for _ in range(n_min_per_g):
            g.append(i)

    y = []
    for i in range(n_cls):
        for _ in range(n_min_per_targ_cls):
            y.append(i)

    n_grp_remain = n_samp - len(g)
    n_cls_remain = n_samp - len(y)

    grp_labels = [*range(n_grp)]
    cls_labels = [*range(n_cls)]

    rng = np.random.default_rng()
    p_cls = rng.standard_exponential(size=n_cls)
    p_cls = p_cls / p_cls.sum()

    p_grp = rng.standard_exponential(size=n_grp)
    p_grp = p_grp / p_grp.sum()

    y = y + rng.choice(cls_labels, size=n_cls_remain, replace=True, p=p_cls).tolist()
    g = g + rng.choice(grp_labels, size=n_grp_remain, replace=True, p=p_grp).tolist()

    rng.shuffle(y)
    rng.shuffle(g)

    y_cnts = np.unique(y, return_counts=True)[1]
    if len(y_cnts) != n_cls or y_cnts.min() < n_min_per_targ_cls:
        raise RuntimeError("Impossible!")

    g_cnts = np.unique(g, return_counts=True)[1]
    if len(g_cnts) != n_grp or g_cnts.min() < n_min_per_g:
        raise RuntimeError("Impossible!")
    return Series(name="y", data=y), Series(name="g", data=g)


def random_X_g_y(
    grp_cnts: list[int], y_cnts: list[int]
) -> tuple[DataFrame, Series, Series]:
    assert sum(grp_cnts) == sum(y_cnts)
    pass


def test_unsplittable() -> None:
    g = Series(np.concatenate([np.zeros(50), np.ones(50)]))
    y = Series(np.concatenate([np.ones(45), np.zeros(5), np.zeros(45), np.ones(5)]))
    kf = OmniKFold(
        n_splits=5, is_classification=True, grouped=True, warn_on_fallback=False
    )
    splits, failed = kf.split(y.to_frame(), y, g)
    assert failed

    y = Series(np.concatenate([range(5) for _ in range(5)]))
    kf = OmniKFold(
        n_splits=5, is_classification=True, grouped=False, warn_on_fallback=False
    )
    with pytest.raises(RuntimeError, match="Attempted to split target"):
        kf.split(y.to_frame(), y, g)


def test_should_fail_warn_and_fallback() -> None:
    """
    Test that warnings are printed on initial print failure, but not when
    warn_on_fallback=False
    """
    g = Series(np.concatenate([np.zeros(50), np.ones(50)]))
    y = Series(np.concatenate([np.ones(45), np.zeros(5), np.zeros(45), np.ones(5)]))
    kf = OmniKFold(
        n_splits=5, is_classification=True, grouped=True, warn_on_fallback=True
    )
    with pytest.warns(
        UserWarning,
        match="Could not perform a grouped, stratified split of the target",
    ):
        splits, failed = kf.split(y.to_frame(), y, g)
    assert failed

    kf = OmniKFold(
        n_splits=5, is_classification=True, grouped=True, warn_on_fallback=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        splits, failed = kf.split(y.to_frame(), y, g)
    assert failed


def test_fail_labeled_error_message() -> None:
    """
    Test that the 'labels' argument gets handled correctly when informing user
    of a splitting failure.
    """
    y = Series(np.concatenate([range(5) for _ in range(5)]))
    labels = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
    }
    kf = OmniKFold(
        n_splits=5,
        is_classification=True,
        grouped=False,
        labels=labels,
        warn_on_fallback=False,
    )
    regex = re.compile(
        r"Attempted to split target.*zero.*one.*two.*three.*four", flags=re.DOTALL
    )
    with pytest.raises(RuntimeError, match=regex):
        kf.split(y.to_frame(), y, g_train=None)


def test_approximate_split(capsys: CaptureFixture) -> None:
    rng = np.random.default_rng()
    rows = []
    i = 0
    attempts = 0
    while i < 500:
        n_samp = rng.integers(200, 2000)
        n_cls = rng.integers(2, 5)
        n_grp = rng.integers(2, 10)
        g, y = min_examples(
            n_cls=n_cls, n_grp=n_grp, n_samp=n_samp, n_min_per_targ_cls=20
        )
        train_size = rng.uniform(0.5, 0.9)
        ss = ApproximateStratifiedGroupSplit(
            train_size=train_size,
            is_classification=True,
            grouped=True,
            warn_on_fallback=False,
            warn_on_large_size_diff=False,
            df_analyze_phase="Initial holdout split",
        )
        try:
            (ix_train, ix_test), group_fail = ss.split(y.to_frame(), y, g)
        except RuntimeError as e:
            if attempts > 50:
                raise RuntimeError("Couldn't generate splittable data") from e
            attempts += 1
            continue
        attempts = 0
        desired = ss.desired_train_size
        achieved = len(ix_train) / len(y)
        row = DataFrame(
            {
                "N": n_samp,
                "c": n_cls,
                "g": n_grp,
                "grp_fail": group_fail,
                "train": train_size,
                "actual": achieved,
                "diff": achieved - desired,
            },
            index=[i],
        )
        rows.append(row)
        i += 1
    df = pd.concat(rows, axis=0, ignore_index=False)
    with capsys.disabled():
        print("")
        with pd.option_context("display.max_rows", 50):
            print(df)
            print("Mean diff:", df["diff"].mean(), "P(grp_fail):", df["grp_fail"].mean())


def do_prep_split_cached(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname in ["internet_usage", "dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc"]:
        with pytest.raises(
            ValueError, match=r".*Target 'target' has undersampled levels.*"
        ):
            prep = ds.prepared(load_cached=True)
            prep.split(train_size=0.6)
        return

    try:
        prep = ds.prepared(load_cached=True)
        prep.split(train_size=0.6)
    except ValueError as e:
        if dsname == "credit-approval_reproduced":
            message = str(e)
            assert "is constant" in message
        else:
            raise e
    except Exception as e:
        raise ValueError(f"Could not prepare data: {dsname}") from e


@fast_ds
@pytest.mark.cached
def test_prep_cached_fast(dataset: tuple[str, TestDataset]) -> None:
    do_prep_split_cached(dataset)


@med_ds
@pytest.mark.cached
def test_prep_cached_med(dataset: tuple[str, TestDataset]) -> None:
    do_prep_split_cached(dataset)


@slow_ds
@pytest.mark.cached
def test_prep_cached_slow(dataset: tuple[str, TestDataset]) -> None:
    do_prep_split_cached(dataset)


def visual_sanity_check() -> None:
    matplotlib.use("QtAgg")
    clses_grps = []
    for _ in range(20):
        for n_cls in np.random.randint(2, 6, 3):
            for n_grp in np.random.randint(2, 6, 3):
                clses_grps.append((n_cls, n_grp))
    np.random.default_rng().shuffle(clses_grps)

    for n_cls, n_grp in clses_grps:
        y, g = min_examples(
            n_grp=n_grp,
            n_cls=n_cls,
            n_samp=200,
            n_min_per_g=2,
            n_min_per_targ_cls=20,
        )
        fig, axes = plt.subplots(ncols=2)
        axes[0].hist(g, bins=len(g.unique()) * 2, color="black")
        axes[0].set_title(f"Groups={n_grp}")

        axes[1].hist(y, bins=len(y.unique()) * 2, color="black")
        axes[1].set_title(f"Target={n_cls}")
        plt.show()


if __name__ == "__main__":
    visual_sanity_check()
