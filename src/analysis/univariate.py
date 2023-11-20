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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Index, Series
from scipy.stats import chisquare, kurtosis, kurtosistest, skew, skewtest
from scipy.stats import differential_entropy as dentropy
from typing_extensions import Literal

from src._types import EstimationMode


def describe_continuous(df: DataFrame, column: str) -> DataFrame:
    cols = [
        "min",
        "mean",
        "max",
        "sd",
        "p05",
        "median",
        "p95",
        "iqr",
        "skew",
        "skew_p",
        "kurt",
        "kurt_p",
        "entropy",
        "nan_n",
        "nan_freq",
    ]
    x = df[column].to_numpy()
    percs = np.nanpercentile(x, q=[0.05, 0.25, 0.5, 0.75, 0.95])
    idx = np.isnan(x)
    n_nan = idx.sum()
    x = x[~idx].ravel()
    p_skew = skewtest(x, nan_policy="omit")[1] if len(x) > 8 else np.nan
    p_kurt = kurtosistest(x, nan_policy="omit")[1] if len(x) > 20 else np.nan
    try:
        entrop = dentropy(x)
    except ValueError:
        entrop = np.nan
    data = {
        "min": np.nanmin(x),
        "mean": np.nanmean(x),
        "max": np.nanmax(x),
        "sd": np.nanstd(x, ddof=1),
        "p05": percs[0],
        "median": percs[2],
        "p95": percs[4],
        "iqr": percs[3] - percs[1],
        "skew": skew(x),
        "skew_p": p_skew,
        "kurt": kurtosis(x, nan_policy="omit"),
        "kurt_p": p_kurt,
        "entropy": entrop,
        "nan_n": n_nan,
        "nan_freq": n_nan / len(df[column]),
    }
    return DataFrame(
        data=data,
        index=[column],
        columns=cols,
    )


def describe_categorical(df: DataFrame, column: str) -> DataFrame:
    cols = [
        "n_levels",  # number of categories
        "nans",  # NaN count
        "nan_freq",  # NaN frequency
        "min_freq",  # Frequency of least-frequent level
        "max_freq",  # Frequency of most-frequent level
        "med_freq",  # Median of level frequencies
        "homog",  # chi-square
        "homog_p",  # chi-square p-value
    ]
    x = df[column]
    nulls = x.isnull()
    unqs, cnts = np.unique(x, return_counts=True)
    freqs = cnts / np.sum(cnts)
    result = chisquare(f_obs=freqs)
    chi = result.statistic
    p = result.pvalue
    data = {
        "n_levels": len(unqs),
        "nans": nulls.sum(),
        "nan_freq": nulls.mean(),
        "min_freq": np.min(freqs),
        "max_freq": np.max(freqs),
        "med_freq": np.median(freqs),
        "homog": chi,
        "homog_p": p,
    }
    return DataFrame(
        data=data,
        index=[column],
        columns=cols,
    )


def describe_all_features(
    continuous: DataFrame,
    categoricals: DataFrame,
    target: Series,
    mode: EstimationMode,
) -> tuple[DataFrame, DataFrame]:
    """
    Parameters
    ----------
    continuous: DataFrame
        DataFrame of continuous and ordinal features

    categoricals: DataFrame
        DataFrame of categorical (un-dummified) features

    target: Series
        Variable to predict

    mode: EstimationMode
        Whether a classification or regression task (nature of target).

    Returns
    -------


    Notes
    -----
    univariate descriptive stats for each feature (robust and non-robust
    measures of scale and location—e.g. mean, median, SD, IQR, and some higher
    moments—e.g. skew, kurtosis), entropy

    For each non-categorical feature:
        - min, mean, max, sd
        - p05, median, p95, IQR
        - skew, kurtosis
        - differential (continuous) entropy (scipy.stats.differential_entropy)

    For each categorical feature:
        - n_cats / n_levels (number of categories)
        - min_freq, max_freq, rng_freq (i.e. min/max/range class proportions)
        - homogeneity / class balance (scipy.stats.chisquare)

    For each ordinal feature:
        - min, mean, max, sd
        - p05, median, p95, IQR
        - skew, kurtosis
        - differential (continuous) entropy (scipy.stats.differential_entropy)

    """
    if mode == "classify":
        categoricals = pd.concat([categoricals, target], axis=1)
    else:
        continuous = pd.concat([continuous, target], axis=1)

    descs_cont: list[DataFrame] = []
    for col in continuous.columns:
        descs_cont.append(describe_continuous(continuous, col))
    df_cont = pd.concat(descs_cont, axis=0)

    descs_cat: list[DataFrame] = []
    for col in categoricals.columns:
        descs_cat.append(describe_categorical(categoricals, col))
    df_cat = pd.concat(descs_cat, axis=0)

    return df_cont, df_cat


def feature_target_stats(df: DataFrame) -> DataFrame:
    """
    For each non-categorical (including ordinal) feature:

        Binary classificataion target:
            - t-test
            - Mann-Whitney U
            - Cohen's d
            - AUROC
            - Mutual Info (sklearn.feature_selection.mutual_info_classif)
            - largest class proportion

        Multiclass classification target (for each target class / level):
            - as above

        Multiclass classification target (means over each target class above):
            - as above

        Regression target:
            - Pearson correlation (scipy.stats.pearsonr)
            - Spearman correlation (scipy.stats.spearmanr)
            - F-statistic (sklearn.feature_selection.f_regression)
            - Mutual Info (sklearn.feature_selection.mutual_info_regression)

    For each categorical feature:

        Binary classificataion target:
            - Cramer's V
            - Mutual Info (sklearn.feature_selection.mutual_info_classif)

        Multiclass classification target (for each target class / level):
            - as above

        Multiclass classification target (means over each target class above):
            - means of above

        Regression target:
            - Mutual Info (sklearn.feature_selection.mutual_info_regression)
            - Kruskal-Wallace H? (scipy.stats.kruskal) (or ANOVA)
            - mean AUROC of each level? (No...)
            - max AUROC of each level? (Yes?)

        Kruskal-Wallace H basically looks at the distribution of continuous
        values for each level of the categorical, and compares if the medians
        are different. Thus, a low H value implies each level sort of looks
        the same on the continuous target, and implies there is not much
        predictive value of the categorical variable, whereas a high H value
        implies the opposite.

            - Pearson correlation (scipy.stats.pearsonr)
            - Spearman correlation (scipy.stats.spearmanr)
            - F-statistic (sklearn.feature_selection.f_regression)
            - Mutual Info (sklearn.feature_selection.mutual_info_regression)

    """
    raise NotImplementedError()


if __name__ == "__main__":
    cat_sizes = np.random.randint(1, 20, 30)

    y_cont = Series(np.random.uniform([250]), name="target")
    y_cat = Series(np.random.randint(0, 6, 250), name="target")
    X_cont = np.random.standard_normal([250, 30])
    X_cat = np.full([250, 30], fill_value=np.nan)
    for i, catsize in enumerate(cat_sizes):
        X_cat[:, i] = np.random.randint(0, catsize, X_cat.shape[0])

    cnames = [f"f{i}" for i in range(X_cont.shape[1])]
    df_cont = DataFrame(data=X_cont, columns=cnames)
    df_cat = DataFrame(data=X_cat, columns=cnames)

    for cname in cnames:
        desc = describe_continuous(df_cont, cname)
        print(desc)

    for cname in cnames:
        desc = describe_categorical(df_cat, cname)
        print(desc)

    desc_cont, desc_cat = describe_all_features(
        continuous=df_cont,
        categoricals=df_cat,
        target=y_cont,
        mode="regress",
    )
    print(desc_cont)
    print(desc_cat)

    desc_cont, desc_cat = describe_all_features(
        continuous=df_cont,
        categoricals=df_cat,
        target=y_cat,
        mode="classify",
    )
    print(desc_cont)
    print(desc_cat)
