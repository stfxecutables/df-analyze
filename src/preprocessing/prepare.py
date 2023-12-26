from __future__ import annotations

from functools import partial
from typing import Any, Optional, Union

import numpy as np
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer  # noqa

from src._constants import N_TARG_LEVEL_MIN
from src.enumerables import NanHandling
from src.preprocessing.cleaning import (
    clean_regression_target,
    deflate_categoricals,
    drop_target_nans,
    drop_unusable,
    encode_categoricals,
    encode_target,
    handle_continuous_nans,
    normalize_continuous,
)
from src.preprocessing.inspection.inspection import (
    InspectionResults,
    convert_categoricals,
    inspect_target,
    unify_nans,
)
from src.timing import timed


class PreparedData:
    def __init__(
        self,
        X: DataFrame,
        X_cont: DataFrame,
        X_cat: DataFrame,
        y: Series,
        labels: Optional[dict[int, str]],
        inspection: InspectionResults,
        is_classification: bool,
        info: dict[str, Any],
    ) -> None:
        self.is_classification: bool = is_classification
        self.inspection: InspectionResults = inspection
        self.info: dict[str, Any] = info

        X, X_cont, X_cat, y = self.validate(X, X_cont, X_cat, y)
        self.X: DataFrame = X
        self.X_cont: DataFrame = X_cont
        self.X_cat: DataFrame = X_cat
        self.y: Series = y
        self.target = self.y.name
        self.labels = labels

    def validate(
        self, X: DataFrame, X_cont: DataFrame, X_cat: DataFrame, y: Series
    ) -> tuple[DataFrame, DataFrame, DataFrame, Series]:
        n_samples = len(X)
        if len(X_cont) != n_samples:
            raise ValueError(
                f"Continuous data number of samples ({len(X_cont)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )
        if len(X_cat) != n_samples:
            raise ValueError(
                f"Categorical data number of samples ({len(X_cat)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )
        if len(y) != n_samples:
            raise ValueError(
                f"Target number of samples ({len(X_cat)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )

        if self.is_classification and np.bincount(y).min() < N_TARG_LEVEL_MIN:
            raise ValueError(f"Target {y} has undersampled levels")

        # Handle some BS due to stupid Pandas index behaviour
        X.reset_index(drop=True, inplace=True)
        X_cont.index = X.index.copy(deep=True)
        X_cat.index = X.index.copy(deep=True)
        y.index = X.index.copy(deep=True)
        return X, X_cont, X_cat, y


def prepare_target(
    df: DataFrame,
    target: str,
    is_classification: bool,
    _warn: bool = True,
) -> tuple[DataFrame, Series, Optional[dict[int, str]]]:
    y = df[target]
    df = df.drop(columns=target)
    if is_classification:
        df, y, labels = encode_target(df, y, _warn=_warn)
    else:
        labels = None
        df, y = clean_regression_target(df, y)
    return df, y, labels


def prepare_data(
    df: DataFrame,
    target: str,
    results: InspectionResults,
    is_classification: bool,
    _warn: bool = True,
) -> PreparedData:
    """
    Returns
    -------
    X_encoded: DataFrame
        All encoded and processed predictors.

    X_cat: DataFrame
        The categorical variables remaining after processing (no encoding,
        for univariate metrics and the like).

    X_cont: DataFrame
        The continues variables remaining after processing (no encoding,
        for univariate metrics and the like).

    y: Series
        The regression or classification target, also encoded.

    info: dict[str, str]
        Other information regarding warnings and cleaning effects.

    """
    times: dict[str, float] = {}
    timer = partial(timed, times=times)
    df = timer(unify_nans)(df)
    df = timer(convert_categoricals)(df, target)
    info = timer(inspect_target)(df, target, is_classification=is_classification)
    df, n_targ_drop = timer(drop_target_nans)(df, target)
    if is_classification:
        df, y, labels = timer(encode_target)(df, df[target])
    else:
        df, y = timer(clean_regression_target)(df, df[target])
        labels = None

    df = timer(drop_unusable)(df, results, _warn=_warn)
    df, X_cont, n_ind_added = handle_continuous_nans(
        df=df, target=target, results=results, nans=NanHandling.Median
    )
    X_cont = normalize_continuous(X_cont, robust=True)

    df = timer(deflate_categoricals)(df, results, _warn=_warn)
    df, X_cat = timer(encode_categoricals)(df, target, results=results, warn_explosion=_warn)

    X = df.drop(columns=target).reset_index(drop=True)
    return PreparedData(
        X=X,
        X_cont=X_cont,
        X_cat=X_cat,
        y=y,
        labels=labels,
        info={
            "n_samples_dropped_via_target_NaNs": n_targ_drop,
            "n_cont_indicator_added": n_ind_added,
            "target": info,
            "runtimes": times,
        },
        inspection=results,
        is_classification=is_classification,
    )
