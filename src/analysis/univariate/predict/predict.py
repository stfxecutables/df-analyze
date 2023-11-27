from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import warnings
from pathlib import Path
from typing import (
    Optional,
)

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from src._types import EstimationMode
from src.analysis.univariate.predict.models import (
    CLS_MODELS,
    REG_MODELS,
    SVMClassifier,
    SVMRegressor,
)


def continuous_feature_target_preds(
    continuous: DataFrame,
    column: str,
    target: Series,
    mode: EstimationMode,
) -> DataFrame:
    X = continuous[column].to_numpy().reshape(-1, 1)
    X = MinMaxScaler().fit_transform(X)
    y = target
    is_multi = False
    if mode == "classify":
        y = Series(data=LabelEncoder().fit_transform(target), name=target.name)  # type: ignore
        is_multi = len(np.unique(y)) > 2
    models = REG_MODELS if mode == "regress" else CLS_MODELS
    # if is_multi and len(y) > 5000:  # takes way too long
    if len(y.ravel()) > 5000:  # takes way too long
        models = [m for m in models if m not in (SVMRegressor, SVMClassifier)]
    scores = []
    pbar = tqdm(
        models, total=len(models), desc=models[0].__class__.__name__, leave=False, position=1
    )
    for model_cls in models:
        pbar.set_description(f"Tuning {model_cls.__name__}")
        model = model_cls()
        score, spam = model.evaluate(X, y)
        score.insert(0, "model", model.short)
        score.index = pd.Index([column], name="feature")
        scores.append(score)
        pbar.update()
    pbar.clear()
    return pd.concat(scores, axis=0)


def categorical_feature_target_preds(
    categoricals: DataFrame,
    column: str,
    target: Series,
    mode: EstimationMode,
) -> DataFrame:
    """Must be UN-ENCODED categoricals"""
    X = pd.get_dummies(categoricals[column], dummy_na=True, dtype=float).to_numpy()
    y = target
    if mode == "classify":
        y = Series(data=LabelEncoder().fit_transform(target), name=target.name)  # type: ignore
    models = REG_MODELS if mode == "regress" else CLS_MODELS
    # if is_multi and len(y) > 5000:  # takes way too long
    if len(y.ravel()) > 5000:  # takes way too long
        models = [m for m in models if m not in (SVMRegressor, SVMClassifier)]
    scores = []
    pbar = tqdm(
        models, total=len(models), desc=models[0].__class__.__name__, leave=False, position=1
    )
    for model_cls in models:
        pbar.set_description(f"Tuning {model_cls.__name__}")
        model = model_cls()
        score, spam = model.evaluate(X, y)
        score.insert(0, "model", model.short)
        score.index = pd.Index([column], name="feature")
        scores.append(score)
        pbar.update()
    pbar.clear()
    return pd.concat(scores, axis=0)


def feature_target_predictions(
    categoricals: DataFrame,
    continuous: DataFrame,
    target: Series,
    mode: EstimationMode,
) -> tuple[Optional[DataFrame], Optional[DataFrame]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df_conts = []
        df_cats = []

        for col in tqdm(
            continuous.columns,
            desc="Predicting continuous features",
            total=continuous.shape[1],
            leave=True,
            position=0,
        ):
            df_conts.append(
                continuous_feature_target_preds(
                    continuous=continuous,
                    column=col,
                    target=target,
                    mode=mode,
                )
            )

        for col in tqdm(
            categoricals.columns,
            desc="Predicting categorical features",
            total=categoricals.shape[1],
            leave=True,
            position=0,
        ):
            df_cats.append(
                categorical_feature_target_preds(
                    categoricals=categoricals,
                    column=col,
                    target=target,
                    mode=mode,
                )
            )

        df_cont = pd.concat(df_conts, axis=0) if len(df_conts) != 0 else None
        df_cat = pd.concat(df_cats, axis=0) if len(df_cats) != 0 else None

    return df_cont, df_cat
