from numpy import ndarray
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, roc_auc_score


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
