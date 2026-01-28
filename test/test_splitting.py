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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from tqdm import tqdm

from df_analyze.splitting import ApproximateStratifiedGroupSplit, OmniKFold
from df_analyze.testing.datasets import (
    TestDataset,
    fast_ds,
    med_ds,
    random_grouped_data,
    slow_ds,
)


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
        g, y = random_grouped_data(
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


def test_degenerate_group_splitting(capsys: CaptureFixture) -> None:
    did_error = False
    okf_splits: list[tuple[np.ndarray, np.ndarray]]
    rows: list[DataFrame]  # https://github.com/pandas-dev/pandas-stubs/issues/902

    with capsys.disabled():
        i = 0
        attempts = 0
        rows = []
        N_ITER = 50
        pbar = tqdm(total=N_ITER)
        n_errors = 0
        while i < N_ITER:
            rng = np.random.default_rng()
            seed = rng.integers(0, 2**32 - 1)
            rng = np.random.default_rng(seed=seed)

            n_samp = rng.integers(100, 20000)
            n_cls = rng.integers(2, 3)
            n_grp = rng.integers(2, 10)
            y, g = random_grouped_data(
                n_cls=n_cls, n_grp=n_grp, n_samp=n_samp, n_min_per_targ_cls=20
            )
            g[:] = np.ones_like(y)  # make degenerate
            g_singular = g.copy()
            g_id = g.copy()
            g_singular[:] = np.zeros_like(g)
            g_id[:] = np.arange(len(g))

            okf = OmniKFold(
                n_splits=5,
                is_classification=True,
                grouped=True,
                labels=None,
                shuffle=False,
                seed=seed,
                warn_on_fallback=True,
            )
            okf2 = OmniKFold(
                n_splits=5,
                is_classification=True,
                grouped=True,
                labels=None,
                shuffle=False,
                seed=seed,
                warn_on_fallback=True,
            )

            assert okf.kf is StratifiedGroupKFold
            sgkf = okf.kf(n_splits=5, shuffle=False)
            skf = StratifiedKFold(n_splits=5, shuffle=False)

            try:
                for g, degen in [(g_singular, "singular"), (g_id, "ids")]:
                    okf_splits, fails = okf.split(
                        X_train=y.to_frame(), y_train=y, g_train=g
                    )
                    okf_splits2, fails2 = okf2.split(
                        X_train=y.to_frame(), y_train=y, g_train=g
                    )
                    sgkf_splits = [*sgkf.split(X=y.to_frame(), y=y, groups=g)]
                    skf_splits = [*skf.split(X=y.to_frame(), y=y)]
                    for k in range(len(okf_splits)):
                        is_test = True
                        item = 1 if is_test else 0
                        okf_ix_train = okf_splits[k][item]
                        okf_ix_train2 = okf_splits2[k][item]
                        skf_ix_train = skf_splits[k][item]
                        sgkf_ix_train = sgkf_splits[k][item]

                        if not np.array_equal(
                            okf_ix_train, skf_ix_train
                        ) or not np.array_equal(okf_ix_train, okf_ix_train2):
                            did_error = True
                            y_unqs_okf, y_cnts_okf = np.unique(
                                y[okf_ix_train], return_counts=True
                            )
                            y_unqs_skf, y_cnts_skf = np.unique(
                                y[skf_ix_train], return_counts=True
                            )
                            y_unqs_sgkf, y_cnts_sgkf = np.unique(
                                y[sgkf_ix_train], return_counts=True
                            )
                            y_unqs, y_cnts = np.unique(y, return_counts=True)

                            y_rats = y_cnts / y_cnts.sum()
                            okf_rats = y_cnts_okf / y_cnts_okf.sum()
                            skf_rats = y_cnts_skf / y_cnts_skf.sum()
                            sgkf_rats = y_cnts_sgkf / y_cnts_sgkf.sum()

                            info = {
                                "y_train classes                   ": y_rats,
                                "OmniKfold train classes:          ": okf_rats,
                                "StratifiedGroupKfold train classes": sgkf_rats,
                                "StratifiedKfold train classes     ": skf_rats,
                            }
                            full_infos = []
                            for line, ratios in info.items():
                                full_infos.append(f"{line}{ratios.round(4)}")

                            # just make sure different split issues aren't due to counts
                            okf_n = len(okf_ix_train)
                            skf_n = len(skf_ix_train)
                            sgkf_n = len(sgkf_ix_train)
                            n_tr_max = max(okf_n, skf_n, sgkf_n)
                            assert abs(okf_n - skf_n) / n_tr_max < 0.01
                            assert abs(okf_n - sgkf_n) / n_tr_max < 0.01
                            assert abs(sgkf_n - skf_n) / n_tr_max < 0.01
                            n_tr_split = n_tr_max

                            okf_skf_overlap = set(okf_ix_train).intersection(skf_ix_train)
                            okf_sgkf_overlap = set(okf_ix_train).intersection(
                                sgkf_ix_train
                            )
                            skf_sgkf_overlap = set(skf_ix_train).intersection(
                                sgkf_ix_train
                            )

                            # full_infos.append(
                            #     f"OmniKFold/StratifiedKFold            ({okf_n}/{skf_n}) n_overlap: {len(okf_skf_overlap)}"
                            # )
                            # full_infos.append(
                            #     f"OmniKFold/StratifiedGroupKFold       ({okf_n}/{sgkf_n}) n_overlap: {len(okf_sgkf_overlap)}"
                            # )
                            # full_infos.append(
                            #     f"StratifiedKFold/StratifiedGroupKFold ({skf_n}/{sgkf_n}) n_overlap: {len(skf_sgkf_overlap)}"
                            # )

                            # full_info = "\n".join(full_infos)
                            row = DataFrame(
                                {
                                    "degen": degen,
                                    "seed": seed,
                                    "fallback": fails,
                                    "n_samp": len(y),
                                    "n_cls": n_cls,
                                    "p_cls_min": np.min(y_rats),
                                    "p_cls_max": np.max(y_rats),
                                    "okf_cls_min": np.min(okf_rats),
                                    "skf_cls_min": np.min(skf_rats),
                                    "sgkf_cls_min": np.min(sgkf_rats),
                                    "okf_cls_max": np.max(okf_rats),
                                    "skf_cls_max": np.max(skf_rats),
                                    "sgkf_cls_max": np.max(sgkf_rats),
                                    "p(O-kf/S-kf)": len(okf_skf_overlap) / n_tr_split,
                                    "p(S-kf/SG-kf)": len(skf_sgkf_overlap) / n_tr_split,
                                    "p(O-kf/SG-kf)": len(okf_sgkf_overlap) / n_tr_split,
                                },
                                index=[n_errors],
                            )
                            rows.append(row)
                            i += 1
                            n_errors += 1
                            pbar.update()
                            continue

                            # raise ValueError(
                            #     f"Splits don't match for fold {k} when grouper is {degen}:\n"
                            #     f"{full_info}"
                            # )
            except RuntimeError as e:
                if attempts > 50:
                    raise RuntimeError(
                        f"Couldn't generate splittable data for seed: {seed}"
                    ) from e
                attempts += 1
                continue
            attempts = 0
            i += 1
            pbar.update()
        pbar.close()
    if not did_error:
        return

    if len(rows) == 0:
        raise RuntimeError("Couldn't generate any splittable data")

    with capsys.disabled():
        df = pd.concat(rows, axis=0, ignore_index=False)

        pd.options.display.max_rows = N_ITER
        pd.options.display.width = 300
        pd.options.display.max_columns = 20
        print(df)
        print(n_errors)


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
        y, g = random_grouped_data(
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
