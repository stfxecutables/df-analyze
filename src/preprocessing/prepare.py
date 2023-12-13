from functools import partial
from typing import Any

from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer  # noqa

from src.enumerables import NanHandling
from src.preprocessing.cleaning import (
    clean_regression_target,
    deflate_categoricals,
    drop_target_nans,
    drop_unusable,
    encode_categoricals,
    encode_target,
    handle_continuous_nans,
)
from src.preprocessing.inspection.inspection import (
    InspectionResults,
    convert_categoricals,
    inspect_target,
    unify_nans,
)
from src.timing import timed


def prepare_target(
    df: DataFrame,
    target: str,
    is_classification: bool,
    _warn: bool = True,
) -> tuple[DataFrame, Series]:
    y = df[target]
    df = df.drop(columns=target)
    if is_classification:
        df, y = encode_target(df, y, _warn=_warn)
    else:
        df, y = clean_regression_target(df, y)
    return df, y


def prepare_data(
    df: DataFrame,
    target: str,
    results: InspectionResults,
    is_classification: bool,
    _warn: bool = True,
) -> tuple[DataFrame, Series, DataFrame, dict[str, Any]]:
    """
    Returns
    -------
    X_encoded: DataFrame
        All encoded and processed predictors.

    target: Series
        The regression or classification target, also encoded.

    X_cat: DataFrame
        The categorical variables remaining after processing (no encoding,
        for univariate metrics and the like).

    info: dict[str, str]
        Other information regarding warnings and cleaning effects.

    """
    times: dict[str, float] = {}
    timer = partial(timed, times=times)
    df = timer(unify_nans)(df)
    df = timer(convert_categoricals)(df, target)
    info = timer(inspect_target)(df, target, is_classification=is_classification)
    df, n_targ_drop = timer(drop_target_nans)(df, target)
    y = df[target]

    df = timer(drop_unusable)(df, results, _warn=_warn)
    df, n_ind_added = handle_continuous_nans(
        df=df, target=target, results=results, nans=NanHandling.Mean
    )

    df = timer(deflate_categoricals)(df, results, _warn=_warn)
    df, cats = timer(encode_categoricals)(df, target, results=results, warn_explosion=_warn)

    df = df.drop(columns=target)
    if is_classification:
        df, y = timer(encode_target)(df, y)
    else:
        df, y = timer(clean_regression_target)(df, y)
    return (
        df,
        y,
        cats,
        {
            "n_samples_dropped_via_target_NaNs": n_targ_drop,
            "n_cont_indicator_added": n_ind_added,
            "target": info,
            "runtimes": times,
        },
    )
