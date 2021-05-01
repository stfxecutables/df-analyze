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
    "pca",
    "kernel_pca",
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
        htune_trials=20,
        htune_validation=0.3,
        test_validation=5,
        classifier="svm",
    )
    sys.exit()

