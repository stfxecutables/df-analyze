from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Type, Union

from joblib import Parallel, delayed

if TYPE_CHECKING:
    from src.cli.cli import ProgramOptions
from math import ceil
from time import perf_counter
from typing import Any, Literal, Type, Union

import numpy as np
from joblib import Parallel, delayed
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm

from src._constants import DEFAULT_N_STEPWISE_SELECT
from src.cli.cli import ProgramOptions
from src.enumerables import WrapperSelectionModel
from src.models.base import DfAnalyzeModel
from src.models.knn import KNNClassifier, KNNRegressor
from src.models.lgbm import LightGBMClassifier, LightGBMRegressor
from src.models.linear import ElasticNetRegressor, SGDClassifierSelector, SGDRegressorSelector
from src.preprocessing.prepare import PreparedData
from src.selection.filter import FilterSelected


def get_dfanalyze_score(
    model_cls: Type[DfAnalyzeModel],
    X: DataFrame,
    y: Series,
    selection_idx: ndarray,
    feature_idx: int,
    is_forward: bool,
) -> float:
    candidate_idx = selection_idx.copy()
    candidate_idx[feature_idx] = True
    if not is_forward:
        candidate_idx = ~candidate_idx
    X_new = X.loc[:, candidate_idx]
    model = model_cls()
    return model.cv_score(X_new, y)


def n_feat_int(prepared: PreparedData, n_features: Union[int, float, None]) -> int:
    n_feat = prepared.X.shape[1]
    if n_features is None:
        return min(n_feat - 1, DEFAULT_N_STEPWISE_SELECT)
    if isinstance(n_features, float):
        return min(n_feat - 1, ceil(n_features * n_feat))
    return min(n_feat - 1, n_features)


class StepwiseSelector:
    def __init__(
        self,
        prep_train: PreparedData,
        options: ProgramOptions,
        n_features: Union[int, float, None] = None,
        direction: Literal["forward", "backward"] = "forward",
    ) -> None:
        self.is_forward = direction == "forward"
        self.n_features: int = n_feat_int(prep_train, n_features)
        self.total_feats: int = prep_train.X.shape[1]
        self.prepared = prep_train
        self.options = options
        self.model = options.wrapper_model

        # selection_idx is True for selected/excluded features in forward/backward select
        self.selection_idx = np.zeros(shape=self.total_feats, dtype=bool)
        self.scores = np.full(shape=self.total_feats, fill_value=np.nan)
        self.remaining: list[str] = prep_train.X.columns.to_list()

        self.n_iterations = (
            self.n_features if self.is_forward else self.total_feats - self.n_features
        )

    def fit(self) -> None:
        ddesc = "Forward" if self.is_forward else "Backward"
        for _ in tqdm(
            range(self.n_iterations),
            total=self.n_iterations,  # type: ignore
            desc=f"{ddesc} feature selection: ",
            leave=True,
        ):  # type: ignore
            new_feature_idx, score = self._get_best_new_feature()
            self.selection_idx[new_feature_idx] = True
            self.scores[new_feature_idx] = score

        if not self.is_forward:
            self.selection_idx = ~self.selection_idx
        self.support_ = self.selection_idx
        self.scores = self.scores[~np.isnan(self.scores)]

    def estimate_runtime(self) -> float:
        if self.is_forward:
            # better approximation than using the first iteration
            orig = self.selection_idx.copy()
            half = ceil(self.n_features / 2)
            self.selection_idx[:half] = True
        start = perf_counter()
        self._get_best_new_feature()
        elapsed = perf_counter() - start
        if self.is_forward:
            self.selection_idx = orig  # type: ignore

        total = self.n_iterations * elapsed
        return round(total / 60, 1)

    def _get_best_new_feature(self) -> tuple[int, float]:
        # Return the best new feature to add to the current_mask, i.e. return
        # the best new feature to add (resp. remove) when doing forward
        # selection (resp. backward selection)
        candidate_idx = np.flatnonzero(~self.selection_idx)
        model_enum = self.options.wrapper_model
        is_cls = self.prepared.is_classification
        if model_enum is WrapperSelectionModel.LGBM:
            model_cls = LightGBMClassifier if is_cls else LightGBMRegressor
        elif model_enum is WrapperSelectionModel.KNN:
            model_cls = KNNClassifier if is_cls else KNNRegressor
        else:
            model_cls = SGDClassifierSelector if is_cls else ElasticNetRegressor

        scores: list[tuple[float, int]] = Parallel()(
            delayed(get_dfanalyze_score)(  # type: ignore
                model_cls=model_cls,
                X=self.prepared.X,
                y=self.prepared.y,
                selection_idx=self.selection_idx,
                feature_idx=feature_idx,
                is_forward=self.is_forward,
            )
            for feature_idx in candidate_idx
        )
        # scores_dict = {idx: score for score, idx in scores}
        # feature_idx = max(scores_dict, key=lambda feature_idx: scores_dict[feature_idx])
        # score = scores_dict[feature_idx]
        # return feature_idx, score

        if np.isnan(scores).sum() != 0:
            print("???")
        idx = np.argmax(scores)
        selected_idx = candidate_idx[idx]
        score = scores[idx]
        return selected_idx, score


def stepwise_select(
    prep_train: PreparedData, options: ProgramOptions
) -> Optional[tuple[list[str], dict[str, float]]]:
    if options.wrapper_select is None:
        return
    selector = StepwiseSelector(
        prep_train=prep_train,
        options=options,
        n_features=options.n_feat_wrapper,
        direction=options.wrapper_select.direction(),
    )
    selector.fit()
    selected_idx = selector.support_
    selected_feats = prep_train.X.loc[:, selected_idx].columns.to_list()
    scores = {}
    for feat, score in zip(selected_feats, selector.scores):
        scores[feat] = score
    return selected_feats, scores
