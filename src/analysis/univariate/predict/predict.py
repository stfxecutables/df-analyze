from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import traceback
import warnings
from pathlib import Path
from typing import (
    Optional,
    Union,
)
from warnings import WarningMessage

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
) -> tuple[DataFrame, Optional[Exception], list[WarningMessage]]:
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("once")
        try:
            x = continuous[column].to_numpy().reshape(-1, 1)
            y = target
            models = REG_MODELS if mode == "regress" else CLS_MODELS
            scores = []
            # pbar = tqdm(
            #     models, total=len(models), desc=models[0].__class__.__name__, leave=False, position=1
            # )
            for model_cls in models:
                # pbar.set_description(f"Tuning {model_cls.__name__}")
                model = model_cls()
                score, spam = model.evaluate(x, y)
                score.insert(0, "model", model.short)
                score.index = pd.Index([column], name="feature")
                scores.append(score)
                # pbar.update()
            # pbar.clear()
            return pd.concat(scores, axis=0), None, warns
        except Exception as e:
            traceback.print_exc()
            print(f"Got error generating predictions for column {column}: {e}")
            return DataFrame(), e, warns


def categorical_feature_target_preds(
    categoricals: DataFrame,
    column: str,
    target: Series,
    mode: EstimationMode,
) -> tuple[DataFrame, Optional[Exception], list[WarningMessage]]:
    """Must be UN-ENCODED categoricals"""
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("once")
        try:
            X = pd.get_dummies(categoricals[column], dummy_na=True, dtype=float).to_numpy()
            y = target
            if mode == "classify":
                y = Series(data=LabelEncoder().fit_transform(target), name=target.name)  # type: ignore
            models = REG_MODELS if mode == "regress" else CLS_MODELS
            scores = []
            # pbar = tqdm(
            #     models, total=len(models), desc=models[0].__class__.__name__, leave=False, position=1
            # )
            for model_cls in models:
                # pbar.set_description(f"Tuning {model_cls.__name__}")
                model = model_cls()
                score, spam = model.evaluate(X, y)
                score.insert(0, "model", model.short)
                score.index = pd.Index([column], name="feature")
                scores.append(score)
                # pbar.update()
            # pbar.clear()
            return pd.concat(scores, axis=0), None, warns
        except Exception as e:
            traceback.print_exc()
            print(f"Got error generating predictions for column {column}: {e}")
            return DataFrame(), e, warns


def feature_target_predictions(
    categoricals: DataFrame,
    continuous: DataFrame,
    target: Series,
    mode: EstimationMode,
) -> tuple[Optional[DataFrame], Optional[DataFrame], list[BaseException], list[WarningMessage]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df_conts: list[DataFrame]
        df_cats: list[DataFrame]
        results: list[tuple[DataFrame, Optional[Exception], list[WarningMessage]]]

        results = Parallel(n_jobs=-1)(
            delayed(continuous_feature_target_preds)(
                continuous=continuous, column=col, target=target, mode=mode
            )
            for col in tqdm(
                continuous.columns,
                desc="Predicting continuous features",
                total=continuous.shape[1],
                leave=True,
                position=0,
            )
        )  # type: ignore
        if len(results) > 0:
            df_conts, cont_errors, cont_warns = list(zip(*results))
        else:
            df_conts, cont_errors, cont_warns = [], [], []

        results = Parallel(n_jobs=-1)(
            delayed(categorical_feature_target_preds)(
                categoricals=categoricals, column=col, target=target, mode=mode
            )
            for col in tqdm(
                categoricals.columns,
                desc="Predicting categorical features",
                total=categoricals.shape[1],
                leave=True,
                position=0,
            )
        )  # type: ignore

        if len(results) > 0:
            df_cats, cat_errors, cat_warns = list(zip(*results))
        else:
            df_cats, cat_errors, cat_warns = [], [], []

        df_cont = pd.concat(df_conts, axis=0) if len(df_conts) != 0 else None
        df_cat = pd.concat(df_cats, axis=0) if len(df_cats) != 0 else None
        errs = [*cont_errors, *cat_errors]
        errs = [e for e in errs if e is not None]
        all_warns = []
        if isinstance(cont_warns, tuple):
            cont_warns = list(cont_warns)
        if isinstance(cat_warns, tuple):
            cat_warns = list(cat_warns)
        for warns in cont_warns + cat_warns:
            all_warns.extend(warns)
        all_warns = [w for w in all_warns if w is not None]

    return df_cont, df_cat, errs, all_warns
