from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Type, Union

from joblib import Parallel, delayed

if TYPE_CHECKING:
    from src.cli.cli import ProgramOptions
from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Literal, Type, Union

import numpy as np
from joblib import Parallel, delayed
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame, Series
from tqdm import tqdm

from src._constants import DEFAULT_N_STEPWISE_SELECT
from src.cli.cli import ProgramOptions
from src.enumerables import Scorer, WrapperSelectionModel
from src.models.base import DfAnalyzeModel
from src.models.knn import KNNClassifier, KNNRegressor
from src.models.lgbm import LightGBMClassifier, LightGBMRegressor
from src.models.linear import ElasticNetRegressor, SGDClassifierSelector
from src.preprocessing.prepare import PreparedData


@dataclass
class RedundantFeatures:
    best: str
    best_idx: int
    best_score: float
    features: list[str]
    scores: list[float]
    metric: str

    def n_feat(self) -> int:
        return len(self.features)

    def to_markdown_section(self, iteration: int, is_cls: bool) -> str:
        lines = []
        if iteration == 0:
            lines.append("# Redundant Stepwise Selection Results\n\n")
            lines.append(f"Metric: {self.metric}\n\n")

        lines.append(f"* {self.best} ({self.metric}={self.best_score:0.5f}) - ")
        lines.append(f"[Iteration {iteration: >3d}]\n\n")
        if len(self.features) == 0:
            return "".join(lines)[:-1]  # ignore last \n

        lines.append("Redundants:\n\n")
        df = DataFrame(
            data=self.scores, index=self.features, columns=["score"]
        ).sort_values(by="score", ascending=not is_cls)
        lines.append(df.to_markdown(tablefmt="simple", floatfmt="0.4f", index=True))
        lines.append("\n\n")
        return "".join(lines)


def get_dfanalyze_score(
    model_cls: Type[DfAnalyzeModel],
    X: DataFrame,
    y: Series,
    metric: Scorer,
    selection_idx: ndarray,
    feature_idx: int,
    is_forward: bool,
    test: bool,
) -> float:
    candidate_idx = selection_idx.copy()
    candidate_idx[feature_idx] = True
    if not is_forward:
        candidate_idx = ~candidate_idx

    X_new = X.loc[:, candidate_idx]
    model = model_cls()
    return model.cv_score(X_new, y, test=test, metric=metric)


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
        test: bool = False,
    ) -> None:
        self.is_forward = direction == "forward"
        self.n_features: int = n_feat_int(prep_train, n_features)
        self.total_feats: int = prep_train.X.shape[1]
        self.prepared = prep_train
        self.options = options
        self.model = options.wrapper_model
        self.redundant = options.redundant_selection
        self.test = test

        # selection_idx is True for selected/excluded features in forward/backward select
        self.selection_idx = np.zeros(shape=self.total_feats, dtype=bool)
        self.scores = np.full(shape=self.total_feats, fill_value=np.nan)
        self.remaining: list[str] = prep_train.X.columns.to_list()
        self.ordered_scores: list[tuple[int, float]] = []
        self.redundant_results: list[RedundantFeatures] = []
        self.redundant_early_stop: bool = False

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
            position=0,
        ):  # type: ignore
            if np.all(self.selection_idx):
                self.redundant_early_stop = True
                break

            if not self.redundant:
                new_feature_idx, score = self._get_best_new_feature()
                self.selection_idx[new_feature_idx] = True
                self.scores[new_feature_idx] = score
                self.ordered_scores.append((new_feature_idx, score))
            else:
                # TODO: save in self.selected the ordered features and scores
                selected_idx, results = self._get_best_new_features()
                score = results.best_score
                best_idx = results.best_idx
                self.scores[best_idx] = score
                self.selection_idx[selected_idx] = True
                self.ordered_scores.append((best_idx, score))
                self.redundant_results.append(results)

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

    def _get_best_new_features(self) -> tuple[NDArray[np.int64], RedundantFeatures]:
        """Return the set of selected features via the greedy method"""
        model_enum = self.options.wrapper_model
        is_cls = self.prepared.is_classification
        metric = (
            self.options.htune_cls_metric if is_cls else self.options.htune_reg_metric
        )
        if model_enum is WrapperSelectionModel.LGBM:
            model_cls = LightGBMClassifier if is_cls else LightGBMRegressor
        elif model_enum is WrapperSelectionModel.KNN:
            model_cls = KNNClassifier if is_cls else KNNRegressor
        else:
            model_cls = SGDClassifierSelector if is_cls else ElasticNetRegressor

        # loop only over un-flagged features
        candidate_idx = np.flatnonzero(~self.selection_idx)
        all_scores: list[float] = Parallel(n_jobs=-1)(
            delayed(get_dfanalyze_score)(  # type: ignore
                model_cls=model_cls,
                X=self.prepared.X,
                y=self.prepared.y,
                metric=metric,
                selection_idx=self.selection_idx,
                feature_idx=feature_idx,
                is_forward=self.is_forward,
                test=self.test,
            )
            for feature_idx in tqdm(
                candidate_idx,
                total=len(candidate_idx),
                desc="Getting best new feature",
                position=1,
            )
        )
        scores = np.array(all_scores)
        feat_names = self.prepared.X.columns.to_numpy()[candidate_idx]
        # Now remember redundant selection can stop early, so

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best = feat_names.tolist()[best_idx]
        # scores are such that higher is always better (negation already
        # applied to ensure this), and we take the abs of user-supplied
        # redundancy threshold, so we can just blindly subtract here
        selected_idx = scores >= (best_score - self.options.redundant_threshold)
        selected = feat_names[selected_idx]
        scores_selected = scores[selected_idx]
        idx_sort = np.argsort(scores_selected)
        selected = selected[idx_sort].tolist()
        scores_selected = scores_selected[idx_sort].tolist()

        all_selected_idx = np.zeros_like(self.selection_idx, dtype=bool)
        cols = set(self.prepared.X.columns.to_list())
        for feat in selected:
            for i, col in enumerate(cols):
                if feat == col:
                    all_selected_idx[i] = True

        return all_selected_idx, RedundantFeatures(
            best=best,
            best_idx=best_idx.item(),
            best_score=best_score,
            features=selected,
            scores=scores_selected,
            metric=metric.name,
        )

    def _get_best_new_feature(self) -> tuple[int, float]:
        # Return the best new feature to add to the current_mask, i.e. return
        # the best new feature to add (resp. remove) when doing forward
        # selection (resp. backward selection)
        candidate_idx = np.flatnonzero(~self.selection_idx)
        model_enum = self.options.wrapper_model
        is_cls = self.prepared.is_classification
        metric = (
            self.options.htune_cls_metric if is_cls else self.options.htune_reg_metric
        )
        if model_enum is WrapperSelectionModel.LGBM:
            model_cls = LightGBMClassifier if is_cls else LightGBMRegressor
        elif model_enum is WrapperSelectionModel.KNN:
            model_cls = KNNClassifier if is_cls else KNNRegressor
        else:
            model_cls = SGDClassifierSelector if is_cls else ElasticNetRegressor

        scores: list[float] = Parallel(n_jobs=-1)(
            delayed(get_dfanalyze_score)(  # type: ignore
                model_cls=model_cls,
                X=self.prepared.X,
                y=self.prepared.y,
                metric=metric,
                selection_idx=self.selection_idx,
                feature_idx=feature_idx,
                is_forward=self.is_forward,
                test=self.test,
            )
            for feature_idx in tqdm(
                candidate_idx,
                total=len(candidate_idx),
                desc="Getting best new feature",
                position=1,
            )
        )
        # scores_dict = {idx: score for score, idx in scores}
        # feature_idx = max(scores_dict, key=lambda feature_idx: scores_dict[feature_idx])
        # score = scores_dict[feature_idx]
        # return feature_idx, score

        idx = np.argmax(scores)
        selected_idx = candidate_idx[idx]
        score = scores[idx]
        return selected_idx, score


def stepwise_select(
    prep_train: PreparedData,
    options: ProgramOptions,
    test: bool = False,
) -> Optional[tuple[list[str], dict[str, float], list[RedundantFeatures], bool]]:
    if options.wrapper_select is None:
        return
    selector = StepwiseSelector(
        prep_train=prep_train,
        options=options,
        n_features=options.n_feat_wrapper,
        direction=options.wrapper_select.direction(),
        test=test,
    )
    selector.fit()
    # selected_idx = selector.support_
    # selected_feats = prep_train.X.loc[:, selected_idx].columns.to_list()
    cols = prep_train.X.columns.to_numpy()

    scores = {}
    selected_feats = []
    for idx, score in selector.ordered_scores:
        selected_feats.append(cols[idx])
        scores[cols[idx]] = score

    # scores = {}
    # for feat, score in zip(selected_feats, selector.scores):
    #     scores[feat] = score
    return (
        selected_feats,
        scores,
        selector.redundant_results,
        selector.redundant_early_stop,
    )
