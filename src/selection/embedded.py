import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from pandas import DataFrame
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

from src.cli.cli import ProgramOptions
from src.enumerables import DfAnayzeEmbedSelector
from src.preprocessing.prepare import PreparedData
from src.selection.filter import FilterSelected
from src.selection.models import (
    LightGBMClassifier,
    LightGBMRegressor,
    SGDClassifierSelector,
    SGDRegressorSelector,
)


@dataclass
class EmbedSelected:
    features: list[str]
    selected: list[str]
    scores: list[float]


def embed_select_features(
    prep_train: PreparedData,
    filtered: Optional[FilterSelected],
    options: ProgramOptions,
) -> EmbedSelected:
    ...
    y = prep_train.y
    X = prep_train.X
    X_train = X.loc[:, filtered.selected] if filtered is not None else X
    is_cls = options.is_classification

    if options.embed_select is None:
        return EmbedSelected(
            features=X.columns.to_list(),
            selected=filtered.features,
            scores=[],
        )

    if options.embed_select is DfAnayzeEmbedSelector.Linear:
        model = SGDClassifierSelector() if is_cls else SGDRegressorSelector()

    else:
        model = LightGBMClassifier() if is_cls else LightGBMRegressor()
        # see https://github.com/microsoft/LightGBM/issues/6202#issuecomment-1820286842
        X_train = X_train.rename(columns=lambda col: re.sub(r"[\[\]\{\},:\"]+", "", str(col)))

    model.htune_optuna(X_train=X_train, y_train=y, n_trials=100, n_jobs=-1)
    # `coefs` are floats if Linear, int32 if LGBM
    scores = np.ravel(
        model.tuned_model.coef_
        if options.embed_select is DfAnayzeEmbedSelector.Linear
        else model.tuned_model.feature_importances_
    )

    selector = SelectFromModel(
        model.tuned_model,
        prefit=True,
    )
    idx = np.array(selector.get_support()).astype(bool)
    selected = X_train.loc[:, idx].columns.to_list()  # type: ignore

    return EmbedSelected(
        features=X_train.columns.to_list(), selected=selected, scores=scores.tolist()
    )
