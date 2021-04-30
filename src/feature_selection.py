# TODO
# - implement step-up feature selection
# - implement step-down feature selection
# - extract N largest PCA components as features
# - choose features with largest univariate separation in classes (d, AUC)
# - smarter methods
#   - remove correlated features
#   - remove constant features (no variance)

# NOTE: feature selection is PRIOR to hypertuning, but "what features are best" is of course
# contengent on the choice of regressor / classifier
# correct way to frame this is as an overall derivative-free optimization problem where the
# classifier choice is *just another hyperparameter*
import sys

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series

from featuretools.selection import (
    remove_single_value_features,
    remove_highly_correlated_features,
    remove_low_information_features,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectPercentile,
    GenericUnivariateSelect,
    RFECV,
    SelectFromModel,
    SequentialFeatureSelector,
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LassoCV, LinearRegression

from src.cleaning import DATADIR

# see https://scikit-learn.org/stable/modules/feature_selection.html
# for SVM-based feature selection, LASSO based feature selction, and RF-based feature-selection
# using SelectFromModel

UNCORRELATED = DATADIR / "mcic_uncorrelated.json"


def remove_correlated_custom(df: DataFrame, threshold: float = 0.95) -> DataFrame:
    """TODO: implement this to greedily combine highly-correlated features instead of just dropping"""
    corrs = np.corrcoef(df, rowvar=False)
    rows, cols = np.where(corrs > threshold)
    correlated_feature_pairs = [(df.columns[i], df.columns[j]) for i, j in zip(rows, cols) if i < j]
    for pair in correlated_feature_pairs[:10]:
        print(pair)
    print(f"{len(correlated_feature_pairs)} correlated feature pairs total")
    raise NotImplementedError()


def remove_weak_features(df: DataFrame, decorrelate: bool = True) -> DataFrame:
    """Remove constant, low-information, and highly-correlated (> 0.95) features"""
    if UNCORRELATED.exists():
        return pd.read_json(UNCORRELATED)
    print("Starting shape: ", df.shape)
    df_v = remove_single_value_features(df)
    print("Shape after removing constant features: ", df_v.shape)
    df_i = remove_low_information_features(df_v)
    print("Shape after removing low-information features: ", df_i.shape)
    if not decorrelate:
        return df_i
    df_c = remove_highly_correlated_features(df_i)
    print("Shape after removing highly-correlated features: ", df_c.shape)
    df_c.to_json(UNCORRELATED)
    print(f"Saved uncorrelated features to {UNCORRELATED}")

