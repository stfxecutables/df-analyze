from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import sys
import warnings
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
from scipy.stats import (
    brunnermunzel,
    chisquare,
    kruskal,
    kurtosis,
    kurtosistest,
    mannwhitneyu,
    pearsonr,
    skew,
    skewtest,
    spearmanr,
    ttest_ind,
)
from scipy.stats import differential_entropy as dentropy
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import mutual_info_classif as minfo_cat
from sklearn.feature_selection import mutual_info_regression as minfo_cont
from sklearn.preprocessing import LabelEncoder
from typing_extensions import Literal

from src._types import EstimationMode
from src.analysis.metrics import auroc, cohens_d, cramer_v


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


def cont_feature_cat_target_level_stats(x: Series, y: Series, level: Any) -> DataFrame:
    stats = [
        "t",
        "t_p",
        "U",
        "U_p",
        "W",
        "W_p",
        "cohen_d",
        "AUROC",
        "corr",
        "corr_p",
        "mut_info",
    ]

    idx_level = y == level
    y_bin = idx_level.astype(float)
    g0 = x[~idx_level]
    g1 = x[idx_level]
    tt_res = ttest_ind(g0, g1, equal_var=False)
    t, t_p = tt_res.statistic, tt_res.pvalue  # type: ignore
    U_res = mannwhitneyu(g0, g1)
    U, U_p = U_res.statistic, U_res.pvalue
    W_res = brunnermunzel(g0, g1)
    W, W_p = W_res.statistic, W_res.pvalue
    r_res = pearsonr(x, y_bin)
    r, r_p = r_res.statistic, r_res.pvalue  # type: ignore

    data = {
        "t": t,
        "t_p": t_p,
        "U": U,
        "U_p": U_p,
        "W": W,
        "W_p": W_p,
        "cohen_d": cohens_d(g0, g1),
        "AUROC": auroc(x, idx_level.astype(int)),
        "corr": r,
        "corr_p": r_p,
        "mut_info": minfo_cat(x.to_frame(), y_bin),
    }

    return DataFrame(
        data=data,
        index=[f"{x.name}_{y.name}.{level}"],
        columns=stats,
    )


def cont_feature_cont_target_stats(x: Series, y: Series) -> DataFrame:
    stats = [
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
        "F",
        "F_p",
        "mut_info",
    ]

    xx = x.to_numpy().ravel()
    yy = y.to_numpy().ravel()

    r_res = pearsonr(xx, yy)
    r, r_p = r_res.statistic, r_res.pvalue  # type: ignore
    rs_res = spearmanr(xx, yy)
    rs, rs_p = rs_res.statistic, rs_res.pvalue  # type: ignore
    F, F_p = f_regression(xx.reshape(-1, 1), yy)

    data = {
        "pearson_r": r,
        "pearson_p": r_p,
        "spearman_r": rs,
        "spearman_p": rs_p,
        "F": F,
        "F_p": F_p,
        "mut_info": minfo_cont(xx.reshape(-1, 1), yy),
    }

    return DataFrame(
        data=data,
        index=[f"{x.name}_{y.name}"],
        columns=stats,
    )


def continuous_feature_target_stats(
    continuous: DataFrame,
    column: str,
    target: Series,
    mode: EstimationMode,
) -> DataFrame:
    ...
    x = continuous[column]
    y = target
    if mode == "classify":
        levels = np.unique(y).tolist()
        descs = []
        for level in levels:
            desc = cont_feature_cat_target_level_stats(x, y, level=level)
            descs.append(desc)
        desc = pd.concat(descs, axis=0)
        is_multiclass = len(levels) > 2
        if is_multiclass:
            # TODO: collect mean stats when this makes sense?
            # TODO: collect some other fancy stat?
            ...
        return desc

    return cont_feature_cont_target_stats(x, y)


def cat_feature_cont_target_stats(x: Series, y: Series) -> DataFrame:
    stats = [
        "mut_info",  # sklearn.feature_selection.mutual_info_regression
        "H",  # Kruskal-Wallace H
        "H_p",
    ]

    xx = x.to_numpy().ravel()
    yy = y.to_numpy().ravel()

    minfo = minfo_cont(xx.reshape(-1, 1), y, discrete_features=True)
    H, H_p = kruskal(x, y)

    data = {
        "mut_info": minfo,
        "H": H,
        "H_p": H_p,
    }

    return DataFrame(
        data=data,
        index=[f"{x.name}_{y.name}"],
        columns=stats,
    )


def cat_feature_cat_target_level_stats(x: Series, y: Series, level: str) -> DataFrame:
    stats = ["cramer_v", "mut_info"]

    xx = x.to_numpy().ravel()

    idx_level = y == level
    y_bin = idx_level.astype(float)
    g0 = x[~idx_level]
    g1 = x[idx_level]

    V = cramer_v(x, y_bin)
    minfo = minfo_cat(xx.reshape(-1, 1), y_bin, discrete_features=True)

    data = {
        "cramer_v": V,
        "mut_into": minfo,
    }

    return DataFrame(
        data=data,
        index=[f"{x.name}_{y.name}.{level}"],
        columns=stats,
    )


def categorical_feature_target_stats(
    categoricals: DataFrame,
    column: str,
    target: Series,
    mode: EstimationMode,
) -> DataFrame:
    ...
    x = categoricals[column]
    y = target
    if mode == "classify":
        levels = np.unique(y).tolist()
        descs = []
        for level in levels:
            desc = cat_feature_cat_target_level_stats(x, y, level=level)
            descs.append(desc)
        desc = pd.concat(descs, axis=0)
        is_multiclass = len(levels) > 2
        if is_multiclass:
            V = cramer_v(x, y)
            minfo = minfo_cat(x.to_numpy().reshape(-1, 1), y, discrete_features=True)
            df = DataFrame(data={"cramer_v": V, "mut_info": minfo}, index=[f"{x.name}_{y.name}"])
            desc = pd.concat([desc, df], axis=0)
            # TODO: collect mean stats when this makes sense?
            # TODO: collect some other fancy stat?
            ...
        return desc

    return cat_feature_cont_target_stats(x, y)


def feature_target_stats(
    continuous: DataFrame,
    categoricals: DataFrame,
    target: Series,
    mode: EstimationMode,
) -> tuple[DataFrame, DataFrame]:
    """
    For each non-categorical (including ordinal) feature:

        Binary classification target:
            - t-test
            - Mann-Whitney U
            - Brunner-Munzel W
            - Cohen's d
            - AUROC
            - Mutual Info (sklearn.feature_selection.mutual_info_classif)

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
    df_conts = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        for col in continuous.columns:
            df_conts.append(
                continuous_feature_target_stats(
                    continuous=continuous,
                    column=col,
                    target=target,
                    mode=mode,
                )
            )
        df_cont = pd.concat(df_conts, axis=0)

    df_cats = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        for col in categoricals.columns:
            df_cats.append(
                categorical_feature_target_stats(
                    categoricals=categoricals,
                    column=col,
                    target=target,
                    mode=mode,
                )
            )
        df_cat = pd.concat(df_cats, axis=0)

    return df_cont, df_cat


def test() -> None:
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

    # for cname in cont_names:
    #     desc = describe_continuous(df_cont, cname)
    #     print(desc)

    # for cname in cat_names:
    #     desc = describe_categorical(df_cat, cname)
    #     print(desc)

    # desc_cont, desc_cat = describe_all_features(
    #     continuous=df_cont,
    #     categoricals=df_cat,
    #     target=y_cont,
    #     mode="regress",
    # )
    # print(desc_cont)
    # print(desc_cat)

    # desc_cont, desc_cat = describe_all_features(
    #     continuous=df_cont,
    #     categoricals=df_cat,
    #     target=y_cat,
    #     mode="classify",
    # )
    # print(desc_cont)
    # print(desc_cat)

    res = feature_target_stats(
        continuous=df_cont, categoricals=df_cat, target=y_cat, mode="classify"
    )
    print("Categorical target stats:\n", res[1])
    res = feature_target_stats(
        continuous=df_cont, categoricals=df_cat, target=y_cont, mode="regress"
    )
    print("Continuous target stats:\n", res[0])


if __name__ == "__main__":
    test()
