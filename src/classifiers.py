import os
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from warnings import filterwarnings

import numpy as np
import optuna
from numpy import ndarray
from optuna import Trial
from pandas import DataFrame
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import BaseCrossValidator, LeaveOneOut, StratifiedShuffleSplit
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTreeClassifier

from src._constants import SEED, VAL_SIZE
from src._types import Classifier, CVMethod

LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_LAYER_SIZES = [4, 8, 16, 32]
N_SPLITS = 5
TEST_SCORES = ["accuracy", "roc_auc"]


def bagger(**kwargs: Any) -> Callable:
    """Helper for uniform interface only"""
    return BaggingClassifier(base_estimator=LR(solver=LR_SOLVER), **kwargs)  # type: ignore


def get_classifier_constructor(name: Classifier) -> Callable:
    CLASSIFIERS: Dict[str, Callable] = {
        "rf": RF,
        "svm": SVC,
        "dtree": DTreeClassifier,
        "mlp": MLP,
        "bag": bagger,
    }
    constructor = CLASSIFIERS[name]
    return constructor
