import sys

from typing import Callable
import optuna
import numpy as np

from optuna import Trial
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.analyses import classifier_analysis
from src.cleaning import get_clean_data
from src.hypertune import evaluate_hypertuned, hypertune_classifier, train_val_splits
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
    "pca" "kernel_pca",
    "unvariate_d",
    "unvariate_auc",
    "unvariate_pearson",
    "stewise_up",
    "stewise_down",
]

CLASSIFIERS = ["svm", "rf", "dtree", "lsq_bag", "ann-mlp"]


if __name__ == "__main__":

    classifier_analysis(
        feature_selection="pca",
        n_features=50,
        htune_trials=200,
        htune_validation=10,
        test_validation=10,
        classifier="svm",
    )
    sys.exit()
    # df = get_clean_data()
    # selected = remove_weak_features(df)
    # X_train, X_test, y_train, y_test = train_val_splits(selected)
    # pca = get_pca_features(df, 100)
    # print(pca)
    # kpca = get_kernel_pca_features(df, 100)
    # print(kpca)

    # hypertune_classifier("mlp", X_train, y_train, X_test, y_test, n_trials=200, mlp_args=dict(val_size=0.2))
    # htune = hypertune_classifier(
    #     classifier="svm", X_train=X_train, y_train=y_train, n_trials=20, cv_method=10
    # )
    # evaluate_hypertuned(htune, 5, selected)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(wrapper(X_train, y_train), n_trials=100)
    print("Best params:", study.best_params)
    print("Best 3-Fold Accuracy on Training Set:", study.best_value)
    svc = SVC(**study.best_params)
    test_acc = svc.fit(X_train, y_train).score(X_test, y_test)
    print("Test Accuracy:", test_acc)

