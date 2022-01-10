from numpy import ndarray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)


def sensitivity(y_true: ndarray, y_pred: ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, normalize=None).ravel()
    return tp / (tp + fn)  # type: ignore


def specificity(y_true: ndarray, y_pred: ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, normalize=None).ravel()
    return tn / (tn + fp)  # type: ignore


accuracy_scorer = make_scorer(accuracy_score)
auc_scorer = make_scorer(roc_auc_score)
sensitivity_scorer = make_scorer(sensitivity)
specificity_scorer = make_scorer(specificity)

mae_scorer = make_scorer(mean_absolute_error)
mse_scorer = make_scorer(mean_squared_error)
mdae_scorer = make_scorer(median_absolute_error)
mape_scorer = make_scorer(mean_absolute_percentage_error)
r2_scorer = make_scorer(r2_score)
expl_var_scorer = make_scorer(explained_variance_score)
