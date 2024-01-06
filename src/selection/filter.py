from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from math import ceil
from pathlib import Path
from typing import (
    Optional,
    Union,
)
from warnings import warn

import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler

from src.analysis.univariate.associate import (
    AssocResults,
    CatAssociation,
    CatClsStats,
    CatRegStats,
    ContAssociation,
    ContClsStats,
    ContRegStats,
)
from src.analysis.univariate.predict.predict import PredResults
from src.cli.cli import ProgramOptions
from src.enumerables import ClsScore, RegScore
from src.hypertune import CLASSIFIER_TEST_SCORERS as CLS_SCORERS
from src.hypertune import REGRESSION_TEST_SCORERS as REG_SCORERS
from src.preprocessing.prepare import PreparedData


def _default_from_X(
    X: DataFrame,
    default_p: float,
    default_min: int,
    default_max: int,
    requested: Optional[Union[int, float]] = None,
) -> int:
    if X is None or X.empty:
        return 0

    n_feat = X.shape[1]

    if isinstance(requested, int):
        # do not limit to max if user specifies valid int number of features
        return min(n_feat, requested)

    if isinstance(requested, float):
        if (requested < 0) or (requested > 1):
            warn(
                f"Invalid percent of features requested: {requested}. Must be "
                f"value in [0, 1]. Setting to default value of {default_p}"
            )
            requested = default_p
        else:  # valid float
            percent = requested
            n = ceil(percent * n_feat)
            return min(n_feat, n)

    # now either caller provided invalid request or it was None
    if requested is None:
        requested = default_p

    assert isinstance(requested, float)

    if n_feat <= default_min:
        return n_feat
    n = ceil(requested * n_feat)

    return min(default_max, n)


def n_cont_select_default(
    prepared: PreparedData, requested: Optional[Union[int, float]] = None
) -> int:
    return _default_from_X(
        prepared.X_cont,
        default_p=0.2,
        default_min=10,
        default_max=50,
        requested=requested,
    )


def n_cat_select_default(
    prepared: PreparedData, requested: Optional[Union[int, float]] = None
) -> int:
    return _default_from_X(
        prepared.X_cat,
        default_p=0.1,
        default_min=5,
        default_max=20,
        requested=requested,
    )


def n_total_select_default(
    prepared: PreparedData, requested: Optional[Union[int, float]] = None
) -> int:
    return _default_from_X(
        pd.concat([prepared.X_cat, prepared.X_cont], axis=1),
        default_p=0.25,
        default_min=8,
        default_max=50,
        requested=requested,
    )


def filter_by_univariate_associations(
    associations: AssocResults,
    cont_metric: Optional[ContAssociation] = None,
    cat_metric: Optional[CatAssociation] = None,
    n_cont: Optional[Union[int, float]] = None,
    n_cat: Optional[Union[int, float]] = None,
    n_total: Optional[Union[int, float]] = None,
    significant_only: bool = False,
) -> list[str]:
    is_cls = associations.is_classification
    if is_cls:
        if not isinstance(cont_metric, ContClsStats):
            cont_metric = ContClsStats.default()
        if not isinstance(cat_metric, CatClsStats):
            cat_metric = CatClsStats.default()
    else:
        if not isinstance(cont_metric, ContRegStats):
            cont_metric = ContRegStats.default()
        if not isinstance(cat_metric, CatRegStats):
            cat_metric = CatRegStats.default()

    cont_stats = associations.conts
    cat_stats = associations.cats

    if (cont_stats is None) and (cat_stats is None):
        raise RuntimeError("No association stats for either cont or cat features.")

    if significant_only:
        if cont_metric.has_significance() and cont_stats is not None:
            ps = cont_metric.p_value()
            idx_keep = cont_stats[ps] > 0.05
            cont_stats = cont_stats.loc[idx_keep]

        if cat_metric.has_significance() and cat_stats is not None:
            ps = cat_metric.p_value()
            idx_keep = cat_stats[ps] > 0.05
            cat_stats = cat_stats.loc[idx_keep]

    use_total = n_total is not None

    if all(n is not None for n in [n_cat, n_cont, n_total]):
        warn(
            "Cannot specify all of `n_cat`, `n_cont`, and `n_total`. Will "
            "ignore count specified for `n_total`."
        )
        use_total = False
    if n_cat is None:
        n_cat = n_cat_select_default(prepared, n_cat)
    if n_cont is None:
        n_cont = n_cont_select_default(prepared, n_cont)
    if n_total is None:
        n_total = n_total_select_default(prepared, n_total)

    if cont_stats is not None:
        cont_stats = (
            cont_stats.abs()
            .sort_values(  # type: ignore
                by=cont_metric.value, ascending=not cont_metric.higher_is_better()
            )
            .loc[:, cont_metric.value]
        )
    if cat_stats is not None:
        cat_stats = (
            cat_stats.abs()
            .sort_values(  # type: ignore
                by=cat_metric.value, ascending=not cat_metric.higher_is_better()
            )
            .loc[:, cat_metric.value]
        )

    if use_total:
        # normalize stats to max value
        # this doesn't really make sense due to differing scales of metrics...
        # what if we replace values by ranks instead? Still no, because then
        # if we have, say, 5 categoricals that are all useless, we always use
        # all of them...
        cont_scaled = MinMaxScaler().fit_transform(cont_stats)  # type: ignore
        cat_scaled = MinMaxScaler().fit_transform(cat_stats)  # type: ignore
        cont_stats = Series(index=cont_stats.index, data=cont_scaled)
        cat_stats = Series(index=cat_stats.index, data=cat_scaled)

        all_stats = pd.concat([cont_stats, cat_stats])
        all_stats.name = "metric"

    cont_cols = []
    if cont_stats is not None:
        cont_stats = (
            cont_stats.abs()
            .sort_values(by=cont_metric.value, ascending=not cont_metric.higher_is_better())
            .iloc[:n_cont]
        )
        cont_cols = cont_stats.columns.to_list()

    cat_cols = []
    if cat_stats is not None:
        cat_stats = (
            cat_stats.abs()
            .sort_values(by=cat_metric.value, ascending=not cat_metric.higher_is_better())
            .iloc[:n_cat]
        )
        cat_cols = cat_stats.columns.to_list()

    return cont_cols + cat_cols


def dummy_adjust(df: DataFrame) -> DataFrame:
    idx = df["model"] == "dummy"
    dummy_cont = df.loc[idx, :].drop(columns="model")
    linear_cont = df.loc[~idx, :].drop(columns="model")


def filter_by_univariate_predictions(
    predictions: PredResults,
    cont_metric: RegScore = RegScore.MAE,
    cat_metric: ClsScore = ClsScore.Accuracy,
    n_cont: Optional[Union[int, float]] = None,
    n_cat: Optional[Union[int, float]] = None,
    significant_only: bool = False,
) -> list[str]:
    cont_preds = predictions.conts
    cat_preds = predictions.cats

    if (cont_preds is None) and (cat_preds is None):
        raise RuntimeError("No univariate predictions for either cont or cat features.")

    if cont_preds is not None:
        idx = cont_preds["model"] == "dummy"
        dummy_cont = cont_preds.loc[idx, :].drop(columns="model")
        linear_cont = cont_preds.loc[~idx, :].drop(columns="model")

    if cat_preds is not None:
        idx = cat_preds["model"] == "dummy"
        dummy_cat = cat_preds.loc[idx, :].drop(columns="model")
        linear_cat = cat_preds.loc[~idx, :].drop(columns="model")

    # columns are "model" (either "dummy" or "sgd-linear") and then metrics
    if significant_only:  # here this means "beats dummy"
        col = cont_metric.value
        # idx_keep = cont_stats[ps] > 0.05
        # cont_stats = cont_stats.loc[idx_keep]

        idx_keep = cat_stats[ps] > 0.05
        cat_stats = cat_stats.loc[idx_keep]


class FilterSelected:
    features: list[str]
    selected: list[str]
    scores: list[float]


def filter_select_features(
    associations: AssocResults,
    predictions: PredResults,
    options: ProgramOptions,
) -> FilterSelected:
    ...
