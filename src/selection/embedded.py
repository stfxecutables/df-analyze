from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import randint, uniform
from typing import TYPE_CHECKING, Optional, cast

import jsonpickle
import numpy as np
from pandas import DataFrame, Series
from sklearn.feature_selection import SelectFromModel

if TYPE_CHECKING:
    from src.cli.cli import ProgramOptions
from src.enumerables import EmbedSelectionModel
from src.models.lgbm import LightGBMClassifier, LightGBMRegressor
from src.models.linear import SGDClassifierSelector, SGDRegressorSelector
from src.preprocessing.prepare import PreparedData
from src.testing.datasets import TestDataset


@dataclass
class EmbedSelected:
    model: EmbedSelectionModel
    selected: list[str]
    scores: dict[str, float]
    is_classification: bool

    def to_markdown(self) -> str:
        fnames, scores = zip(*self.scores.items())
        scores = DataFrame(
            data=scores, index=Series(name="feature", data=fnames), columns=["score"]
        )
        direction = "Higher" if self.is_classification else "Lower"
        metric = "Accuracy" if self.is_classification else "MAE"
        text = (
            "# Wrapper-Based Feature Selection Summary\n\n"
            f"Wrapper model:  {self.model.name}\n"
            "\n"
            f"## Selected Features\n\n"
            f"{self.selected}\n\n"
            f"## Selection scores ({metric}: {direction} = More important)\n\n"
            f"{scores.to_markdown(floatfmt='0.3e')}"
        )
        return text

    def to_json(self) -> str:
        return str(jsonpickle.encode(self))

    @staticmethod
    def from_json(path: Path) -> EmbedSelected:
        return cast(EmbedSelected, jsonpickle.decode(path.read_text()))

    @staticmethod
    def random(ds: TestDataset) -> EmbedSelected:
        model = EmbedSelectionModel.random()
        df = ds.load()
        x = df.drop(columns="target", errors="ignore")
        cols = x.columns.to_list()
        n_feat = randint(min(10, x.shape[1]), x.shape[1])
        selected = np.random.choice(cols, size=n_feat, replace=False).tolist()
        scores = {s: uniform(0, 1) for s in selected}
        return EmbedSelected(
            model=model,
            selected=selected,
            scores=scores,
            is_classification=ds.is_classification,
        )


def embed_select_features(
    prep_train: PreparedData,
    options: ProgramOptions,
) -> Optional[EmbedSelected]:
    ...
    y = prep_train.y
    X_train = prep_train.X
    is_cls = options.is_classification

    if options.embed_select is None:
        return None

    if options.embed_select is EmbedSelectionModel.Linear:
        model = SGDClassifierSelector() if is_cls else SGDRegressorSelector()

    else:
        model = LightGBMClassifier() if is_cls else LightGBMRegressor()

    model.htune_optuna(X_train=X_train, y_train=y, n_trials=100, n_jobs=-1)
    # `coefs` are floats if Linear, int32 if LGBM
    scores = np.ravel(
        model.tuned_model.coef_  # type: ignore
        if options.embed_select is EmbedSelectionModel.Linear
        else model.tuned_model.feature_importances_  # type: ignore
    )
    fscores = {feature: score for feature, score in zip(X_train.columns, scores)}

    selector = SelectFromModel(
        model.tuned_model,
        prefit=True,
    )
    idx = np.array(selector.get_support()).astype(bool)
    selected = X_train.loc[:, idx].columns.to_list()  # type: ignore

    return EmbedSelected(
        model=options.embed_select,
        selected=selected,
        scores=fscores,
        is_classification=prep_train.is_classification,
    )
