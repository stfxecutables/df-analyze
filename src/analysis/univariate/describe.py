from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import (
    chisquare,
    kurtosis,
    kurtosistest,
    skew,
    skewtest,
)
from scipy.stats import differential_entropy as dentropy

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
