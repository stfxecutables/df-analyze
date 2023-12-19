import warnings
from typing import Any, Callable

import numpy as np
from numpy import ndarray
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    multilabel_confusion_matrix,
    r2_score,
    recall_score,
    roc_auc_score,
)


def sensitivity(y_true: ndarray, y_pred: ndarray) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=np.nan))  # type: ignore


def specificity(y_true: ndarray, y_pred: ndarray) -> float:
    mat = multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred)
    tns = mat[:, 0, 0]
    fps = mat[:, 0, 1]
    specs = tns / (tns + fps)
    return specs.mean()


def silent_scorer(f: Callable) -> Callable:
    def silent(*args, **kwargs) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Scoring failed", category=UserWarning)
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                if "Number of classes in y_true not equal" in str(e):
                    return np.nan
                if "Input contains NaN" in str(e):
                    return np.nan

    return silent


accuracy_scorer = make_scorer(accuracy_score)
auc_scorer = make_scorer(
    silent_scorer(roc_auc_score), needs_proba=True, multi_class="ovr", average="macro"
)
sensitivity_scorer = make_scorer(sensitivity)
specificity_scorer = make_scorer(specificity)

mae_scorer = make_scorer(mean_absolute_error)
mse_scorer = make_scorer(mean_squared_error)
mdae_scorer = make_scorer(median_absolute_error)
mape_scorer = make_scorer(mean_absolute_percentage_error)
r2_scorer = make_scorer(r2_score)
expl_var_scorer = make_scorer(explained_variance_score)
