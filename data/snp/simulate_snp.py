from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from math import ceil
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm import tqdm
from typing_extensions import Literal

from src.enumerables import ClassifierScorer
from src.models.dummy import DummyClassifier
from src.models.lgbm import LightGBMClassifier

DATA = Path(__file__).resolve().parent


def random_single_feature(
    n_samples: int = 5000,
    min_n_variants: int = 2,
    max_n_variants: int = 8,
    min_variant_samples: int = 20,
) -> tuple[ndarray, ndarray]:
    # ps = -np.sort(-np.random.beta(1.5, 10, n_variants))
    # ps = -np.sort(-np.random.exponential(10, n_variants))
    n_variants = np.random.randint(min_n_variants, max_n_variants)
    x_base = np.repeat(np.arange(n_variants), min_variant_samples)
    n_samples = n_samples - len(x_base)
    if n_samples < 0:
        raise ValueError(
            "Insufficient number of samples to ensure presence of rare variants"
        )

    a = np.random.uniform(1, 10)
    ps = -np.sort(-np.exp(a * np.linspace(0, 1, n_variants)))

    # ensure we have at expect at least 10 examples of min class
    M = min_variant_samples
    pmin = M / n_samples
    # sort so first of ps is smallest, i.e. least likely
    ps = -np.sort(np.clip(ps, a_min=pmin, a_max=None))
    ps = ps / np.sum(ps)
    x_rand = np.random.choice(np.arange(n_variants), size=n_samples, replace=True, p=ps)
    x = np.concatenate([x_base, x_rand])
    count = 0
    while len(np.unique(x)) < n_variants and count < 50:
        x_rand = np.random.choice(
            np.arange(n_variants), size=n_samples, replace=True, p=ps
        )
        x = np.concatenate([x_base, x_rand])
        count += 1
    if count >= 50:
        raise RuntimeError("Could not generate data with sufficient variant samples")
    return x, ps


def random_feature_set(
    n_samples: int,
    n_features: int,
    min_n_variants: int,
    max_n_variants: int,
    min_variant_samples: int,
    n_predictive_combinations: int,
    predictiveness: float,
) -> tuple[DataFrame, Series]:
    feats = []
    all_ps = []
    for _ in range(n_features):
        feat, ps = random_single_feature(
            n_samples=n_samples,
            min_n_variants=min_n_variants,
            max_n_variants=max_n_variants,
            min_variant_samples=min_variant_samples,
        )
        feats.append(feat)
        all_ps.append(ps)

    n_indicators = np.random.randint(1, n_predictive_combinations + 1, size=n_features)
    all_indicators = [np.arange(n_ind) for n_ind in n_indicators]
    idxs = []
    for feat in feats:
        idx = np.zeros_like(feat, dtype=bool)
        for indicators in all_indicators:
            for indicator in indicators:
                idx = idx | (indicator == feat)
        idxs.append(idx)

    idx = np.stack(idxs, axis=1)
    candidates = idx.all(axis=1)
    df = DataFrame(
        np.stack(feats, axis=1), columns=[f"fp{i+1}" for i in range(n_features)]
    )
    y = Series(data=np.asarray(candidates, dtype=np.int64).ravel(), name="target")

    n_match = y.sum()
    if n_match * predictiveness >= min_variant_samples:
        return df, y
    return df, y


def make_snp_set_x(
    n_samples: int = 5000,
    predictive_set_min_size: int = 3,
    predictive_set_max_size: int = 10,
    min_n_variants: int = 2,
    max_n_variants: int = 8,
    predictiveness: float = 0.95,
    n_predictive_combinations: int = 1,
    min_variant_samples: int = 1000,
) -> tuple[DataFrame, Series]:
    """
    Parameters
    ----------
    predictiveness: float
        Value in [0, 1] that tells how strongly a predictive variant predicts
        the target. E.g. given a predictive set of size p (p features) then
        given the pi* is indicator value for pi, then when pi = pi* for all i,
        there is a `predictiveness` probability the target is 1.

    n_predictive_combinations: int = 1
        Maximum number of pi* for each pi.


    A "predictive set" is a small set of feature that *collectively* predicts
    the target, though each SNP in the set in isolation is non-predictive.

    A predictive set feature has a random number of *variants* i.e. possible
    classes. These
    """
    ...
    n_predictive_combinations = min(6, n_predictive_combinations)
    n_predictive_combinations = max(1, n_predictive_combinations)
    pmin = predictive_set_min_size
    pmax = predictive_set_max_size
    P = np.random.randint(pmin, pmax + 1)  # number of features
    count = 0
    MAX_N_ITER = 50
    while count < MAX_N_ITER:
        df, y_raw = random_feature_set(
            n_samples=n_samples,
            n_features=P,
            min_n_variants=min_n_variants,
            max_n_variants=max_n_variants,
            min_variant_samples=min_variant_samples,
            n_predictive_combinations=n_predictive_combinations,
            predictiveness=predictiveness,
        )
        n_match = y_raw.sum()
        if n_match * predictiveness >= min_variant_samples:
            break
        count += 1
    if count >= MAX_N_ITER:
        raise RuntimeError("Could not generate data with specified params")

    # limit predictiveness
    y = y_raw.copy()
    idx_cases = y == 1
    n_potential = idx_cases.sum()
    n_cases = min(n_samples - 1, ceil(predictiveness * n_potential))
    # now make only `predictiveness` of cases actually = 1
    n_phony = n_potential - n_cases
    positions = np.random.choice(np.where(idx_cases)[0], size=n_phony, replace=False)
    y[positions] = 0

    return df, y


def simulate_snp_data(
    n_samples: int = 5000,
    n_snp: int = 30000,
    n_predictive_sets: int = 3,
    predictive_set_min_size: int = 3,
    predictive_set_max_size: int = 10,
    min_n_variants: int = 2,
    max_n_variants: int = 8,
    predictiveness: float = 0.90,
    n_predictive_combinations: int = 2,
    min_variant_samples: int = 20,
) -> tuple[DataFrame, DataFrame, Series]:
    """
    Given p = n_predictive, how can we construct a sample of data X from
    dist(X) such that y = f(X)
    """
    ...
    dfs, ys = [], []
    n_feat = 0
    n_pred = 0
    attempts = 0
    pbar = tqdm(desc="Generating data")
    while n_pred < n_predictive_sets:
        try:
            df, y = make_snp_set_x(
                n_samples=n_samples,
                predictive_set_min_size=predictive_set_min_size,
                predictive_set_max_size=predictive_set_max_size,
                min_n_variants=min_n_variants,
                max_n_variants=max_n_variants,
                predictiveness=predictiveness,
                n_predictive_combinations=n_predictive_combinations,
                min_variant_samples=min_variant_samples,
            )
            df.rename(columns=lambda s: f"p{n_pred + 1}_{s}", inplace=True)
            dfs.append(df)
            ys.append(y)
            n_feat += df.shape[1]
            n_pred += 1
        except RuntimeError as e:
            pbar.update()
            attempts += 1
            if attempts > 1000:
                pbar.close()
                raise RuntimeError("Could not generate data") from e
    pbar.close()

    df_pred = pd.concat(dfs, axis=1)
    if len(ys) > 1:
        y = Series(data=np.any(np.stack(ys, axis=1), axis=1), name="target")

    n_feat_remain = n_snp - n_feat

    xs = []
    for _ in range(n_feat_remain):
        xs.append(
            random_single_feature(
                n_samples=n_samples,
                min_n_variants=min_n_variants,
                max_n_variants=max_n_variants,
                min_variant_samples=min_variant_samples,
            )[0]
        )
    x_nonpred = DataFrame(
        np.stack(xs, axis=1), columns=[f"f{i+1}" for i in range(len(xs))]
    )
    df = pd.concat([df_pred, x_nonpred], axis=1)
    return df, df_pred, y


def show_distributions() -> None:
    fig, axes = plt.subplots(nrows=8, ncols=12, sharex=False, sharey=True)
    all_ps = []
    xs = []
    n_vars = []
    for i, ax in enumerate(axes.flat):
        n_variants = np.random.randint(3, 7)
        # ps = -np.sort(-np.random.beta(1.5, 10, n_variants))
        # ps = -np.sort(-np.random.exponential(10, n_variants))
        a = np.random.uniform(1, 10)
        ps = -np.sort(-np.exp(a * np.linspace(0, 1, n_variants)))
        pmin = 10 / 2000
        ps = np.clip(ps, a_min=pmin, a_max=None)
        ps = ps / np.sum(ps)
        x = np.random.choice(np.arange(n_variants), size=2000, replace=True, p=ps)
        xs.append(x)
        all_ps.append(ps)
        n_vars.append(n_variants)

    sorter = [np.sum(np.log(ps ** (-ps))) for ps in all_ps]
    idx = -np.argsort(sorter)
    r = np.empty(shape=[len(sorter)], dtype=object)
    r[:] = all_ps
    all_ps = r[idx].copy()
    r[:] = xs
    xs = r[idx].copy()
    r[:] = n_vars
    n_vars = r[idx].copy()

    for ax, x, ps, n_var in zip(axes.flat, xs, all_ps, n_vars):
        ax.hist(x, bins=n_var, color="black")

    fig.set_size_inches(w=16, h=10)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    for _ in range(10):
        try:
            df, df_pred, y = simulate_snp_data(
                n_samples=2000,
                # n_snp=30_000,
                n_snp=3_000,
                n_predictive_sets=2,
                predictive_set_min_size=3,
                predictive_set_max_size=8,
                min_n_variants=2,
                max_n_variants=8,
                predictiveness=0.9,
                n_predictive_combinations=2,
                min_variant_samples=20,
            )
        except RuntimeError as e:
            print(e)
            continue

        df_tr, df_ts, y_tr, y_ts = train_test_split(df, y, test_size=0.5, stratify=y)
        x_tr, x_ts, y_tr, y_ts = train_test_split(df_pred, y, test_size=0.5, stratify=y)

        model = LGBMClassifier(verbosity=-1, n_jobs=-1)
        model.fit(df_tr, y_tr)
        full_score = model.score(df_ts, y_ts)

        model = LGBMClassifier(verbosity=-1, n_jobs=-1)
        model.fit(x_tr, y_tr)
        true_score = model.score(x_ts, y_ts)
        guess = max(y.mean(), 1 - y.mean())
        print(
            f"n_feat={df.shape[1]}, mean={y.mean()}, score={full_score}, guess={guess}, true={true_score}"
        )
