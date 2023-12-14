from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import warnings
from pathlib import Path
from typing import (
    Any,
)

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.stats import (
    brunnermunzel,
    kruskal,
    mannwhitneyu,
    pearsonr,
    spearmanr,
    ttest_ind,
)
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif as minfo_cat
from sklearn.feature_selection import mutual_info_regression as minfo_cont
from sklearn.preprocessing import LabelEncoder

from src._types import EstimationMode
from src.analysis.metrics import auroc, cohens_d, cramer_v
from src.preprocessing.inspection.inspection import InspectionResults
from src.preprocessing.prepare import PreparedData


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
    """
    Parameters
    ----------

    x: Series
        Continuous feature

    y: Series
        Continuous target

    Returns
    -------
    stats: DataFrame
        Table of stats
    """
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
    is_classification: bool,
) -> DataFrame:
    ...
    x = continuous[column]
    y = target
    if len(x) != len(y):
        raise ValueError("Continuous features and target do not have same number of samples.")
    if is_classification:
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

    xx = x.astype(str).to_numpy().reshape(-1, 1)
    x_enc = np.asarray(LabelEncoder().fit_transform(xx)).reshape(-1, 1)
    yy = y.to_numpy().ravel()

    minfo = minfo_cont(x_enc, y, discrete_features=True)
    H, H_p = kruskal(x_enc.ravel(), y)

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
    # stats = ["cramer_v", "mut_info"]
    # stats = ["cramer_v", "kl_div"]
    stats = ["cramer_v"]

    idx_level = y == level
    y_bin = idx_level.astype(float)

    # minfo = minfo_cat(xx.reshape(-1, 1), y_bin, discrete_features=True)

    data = {
        "cramer_v": cramer_v(x, y_bin),
        # "kl_div": relative_entropy(x, y_bin),  # need x and y have same n_cls
        # "mut_into": minfo,  # always NaN
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
    is_classification: bool,
) -> DataFrame:
    ...
    x = categoricals[column]
    y = target

    if is_classification:
        xx = x.astype(str).to_numpy().reshape(-1, 1)
        x_enc = np.asarray(LabelEncoder().fit_transform(xx)).ravel()
        xs = Series(data=x_enc, name=x.name)
        levels = np.unique(y.astype(str)).tolist()
        is_multiclass = len(levels) > 2

        descs = []
        for level in levels:
            desc = cat_feature_cat_target_level_stats(xs, y, level=level)
            descs.append(desc)
        desc = pd.concat(descs, axis=0)

        if is_multiclass:
            V = cramer_v(x_enc.reshape(-1, 1), y)
            minfo = minfo_cat(x_enc.reshape(-1, 1), y, discrete_features=True)
            df = DataFrame(data={"cramer_v": V, "mut_info": minfo}, index=[f"{x.name}_{y.name}"])
            desc = pd.concat([desc, df], axis=0)
            # TODO: collect mean stats when this makes sense?
            # TODO: collect some other fancy stat?
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df_conts = []
        df_cats = []

        for col in continuous.columns:
            df_conts.append(
                continuous_feature_target_stats(
                    continuous=continuous,
                    column=col,
                    target=target,
                    is_classification=mode == "classify",
                )
            )

        for col in categoricals.columns:
            df_cats.append(
                categorical_feature_target_stats(
                    categoricals=categoricals,
                    column=col,
                    target=target,
                    is_classification=mode == "classify",
                )
            )

        df_cont = pd.concat(df_conts, axis=0)
        df_cat = pd.concat(df_cats, axis=0)

    return df_cont, df_cat


def target_associations(
    data: PreparedData,
) -> Any:
    ...

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df_conts = []
        df_cats = []

        cont = data.X_cont
        for col in cont.columns:
            df_conts.append(
                continuous_feature_target_stats(
                    continuous=cont,
                    column=col,
                    target=data.y,
                    is_classification=data.is_classification,
                )
            )

        # Currently X_cat is the raw cat data, neither label- nor one-hot-
        # encoded.
        cats = data.X_cat
        for col in cats.columns:
            df_cats.append(
                categorical_feature_target_stats(
                    categoricals=cats,
                    column=col,
                    target=data.y,
                    is_classification=data.is_classification,
                )
            )

        df_cont = pd.concat(df_conts, axis=0) if len(df_conts) > 0 else None
        df_cat = pd.concat(df_cats, axis=0) if len(df_cats) > 0 else None

    return df_cont, df_cat
