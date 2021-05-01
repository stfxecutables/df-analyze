import sys

from typing import Callable
import optuna
import numpy as np

from optuna import Trial
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.analyses import classifier_analysis, classifier_analysis_multitest
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
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

ARG_OPTIONS = dict(
    classifier=["svm", "rf", "dtree", "bag", "mlp"],
    feature_selection=["pca", "kpca", "d", "auc"],
    n_features=[20, 50, 100],
    htune_validation=[5, 10, "mc", 0.2],
)
ARGS = list(ParameterGrid(ARG_OPTIONS))


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

    for args in tqdm(ARGS):
        classifier_analysis_multitest(htune_trials=20, **args)
    sys.exit()

