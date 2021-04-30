from typing import Callable
import optuna
import numpy as np

from optuna import Trial
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.cleaning import get_clean_data
from src.hypertune import train_val_splits
from src.feature_selection import (
    auroc,
    cohens_d,
    correlations,
    remove_weak_features,
    get_pca_features,
    get_kernel_pca_features,
    select_features_by_univariate_rank,
    select_stepwise_features,
)
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


FEATURE_SELECTION_ANALYSES = [
    "pca"
    "kernel_pca",
    "unvariate_d",
    "unvariate_auc",
    "unvariate_pearson",
    "stewise_up",
    "stewise_down",
]

CLASSIFIERS = [
    "svm",
    "rf",
    "dtree",
    "lsq_bag",
    "ann-mlp",
]

def wrapper(X_train: DataFrame, y_train: DataFrame) -> Callable[[Trial], float]:

    def objective(trial: Trial) -> float:
        kernel = trial.suggest_categorical("kernel", choices=["rbf", "linear", "sigmoid"])
        c = trial.suggest_loguniform("C", 1e-10, 1e10)
        svc = SVC(C=c, kernel=kernel)
        return np.mean(cross_val_score(svc, X=X_train, y=y_train, scoring="accuracy", cv=3))

    return objective


if __name__ == "__main__":

    df = get_clean_data()
    selected = remove_weak_features(df)
    X_train, X_test, y_train, y_test = train_val_splits(selected)
    # pca = get_pca_features(df, 100)
    # print(pca)
    # kpca = get_kernel_pca_features(df, 100)
    # print(kpca)

    # print(cohens_d(df))
    # print(auroc(df))
    # print(correlations(df))
    # reduced = select_features_by_univariate_rank(df, "d", 10)
    # reduced = select_stepwise_features(
    #     selected,
    #     SVC(),
    #     n_features=2,
    #     direction="forward"
    # )
    # print(reduced)
    study = optuna.create_study(direction="maximize")
    study.optimize(wrapper(X_train, y_train), n_trials=5)


