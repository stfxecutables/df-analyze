import sys

from typing import Callable
import optuna
import numpy as np

from optuna import Trial
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.cleaning import get_clean_data
from src.hypertune import hypertune_classifier, train_val_splits
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

    # hypertune_classifier("svm", X_train, y_train, X_test, y_test, n_trials=100)
    hypertune_classifier("mlp", X_train, y_train, X_test, y_test, n_trials=200, mlp_args=dict(val_size=0.2))
    sys.exit()





    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(wrapper(X_train, y_train), n_trials=100)
    print("Best params:", study.best_params)
    print("Best 3-Fold Accuracy on Training Set:", study.best_value)
    svc = SVC(**study.best_params)
    test_acc = svc.fit(X_train, y_train).score(X_test, y_test)
    print("Test Accuracy:", test_acc)


