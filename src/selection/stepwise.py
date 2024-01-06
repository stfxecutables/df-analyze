from typing import (
    Optional,
)


from src.cli.cli import ProgramOptions
from src.enumerables import WrapperSelectionModel
from src.models.lgbm import LightGBMClassifier, LightGBMRegressor
from src.models.linear import SGDClassifierSelector, SGDRegressorSelector
from src.preprocessing.prepare import PreparedData
from src.selection.filter import FilterSelected
from src.selection.wrapper import WrapperSelectionModel


def stepwise_select(
    prep_train: PreparedData, filtered: FilterSelected, options: ProgramOptions
) -> Optional[tuple[list[str], dict[str, float]]]:
    selected: list[str] = []  # selected features
    scores: dict[str, float] = {}  # incremental scores
    is_cls = prep_train.is_classification

    if options.wrapper_model is WrapperSelectionModel.Linear:
        model = SGDClassifierSelector() if is_cls else SGDRegressorSelector()

    else:
        model = LightGBMClassifier() if is_cls else LightGBMRegressor()

    if options.wrapper_select is None:
        return None
    results = model.wrapper_select(
        X_train=prep_train.X,
        y_train=prep_train.y,
        n_feat=options.n_feat_wrapper,
        method=options.wrapper_select,
    )
    if results is None:
        return None

    selected, scores = results
    return selected, scores
