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

import os
from typing import Union
from warnings import filterwarnings

import numpy as np
import pandas as pd
from featuretools.selection import (
    remove_highly_correlated_features,
    remove_low_information_features,
    remove_single_value_features,
)
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.decomposition import PCA, KernelPCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing_extensions import Literal

from src._sequential import SequentialFeatureSelector
from src.constants import DATADIR, SEED, UNCORRELATED
from src.hypertune import Classifier, get_classifier_constructor

# see https://scikit-learn.org/stable/modules/feature_selection.html
# for SVM-based feature selection, LASSO based feature selction, and RF-based feature-selection
# using SelectFromModel
FlatArray = Union[DataFrame, Series, ndarray]
UnivariateMetric = Literal["d", "auc", "pearson", "spearman"]
CorrMethod = Literal["pearson", "spearman"]


def cohens_d(df: DataFrame) -> Series:
    """For each feature in `df`, compute the absolute Cohen's d values

    Parameters
    ----------
    df: DataFrame
        DataFrame with shape (n_samples, n_features + 1) and target variable in column "target"
    """
    X = df.drop(columns="target")
    y = df["target"].copy()
    x1, x2 = X.loc[y == 0, :], X.loc[y == 1, :]
    n1, n2 = len(x1) - 1, len(x2) - 1
    sd1, sd2 = np.std(x1, ddof=1, axis=0), np.std(x2, ddof=1, axis=0)
    sd_pools = np.sqrt((n1 * sd1 + n2 * sd2) / (n1 + n2))
    m1, m2 = np.mean(x1, axis=0), np.mean(x2, axis=0)
    ds = np.abs(m1 - m2) / sd_pools
    return ds


def auroc(df: DataFrame) -> Series:
    """For each feature in `df` compute rho, the common-language effect size (see Notes) via the
    area-under-the-ROC curve (AUC), and rescale this effect size to allow sorting across features.

    Parameters
    ----------
    df: DataFrame
        DataFrame with shape (n_samples, n_features + 1) and target variable in column "target"

    Notes
    -----
    This is equivalent to calculating U / abs(y1*y2), where `y1` and `y2` are the subgroup sizes,
    and U is the Mann-Whitney U, and is also sometimes referred to as Herrnstein's rho [1] or "f",
    the "common-language effect size" [2].

    Note we also *must* rescale as rho is implicity "signed", with values above 0.5 indicating
    separation in one direction, and values below 0.5 indicating separation in the other.

    [1] Herrnstein, R. J., Loveland, D. H., & Cable, C. (1976). Natural concepts in pigeons.
        Journal of Experimental Psychology: Animal Behavior Processes, 2, 285-302

    [2] McGraw, K.O.; Wong, J.J. (1992). "A common language effect size statistic". Psychological
        Bulletin. 111 (2): 361–365. doi:10.1037/0033-2909.111.2.361.
    """
    X = df.drop(columns="target")
    y = df["target"].copy()

    aucs = Series(data=[roc_auc_score(y, X[col]) for col in X], index=X.columns)
    rescaled = (aucs - 0.5).abs()
    return rescaled


def correlations(df: DataFrame, method: CorrMethod = "pearson") -> Series:
    """For each feature in `df` compute the ABSOLUTE correlation with the target variable.

    Parameters
    ----------
    df: DataFrame
        DataFrame with shape (n_samples, n_features + 1) and target variable in column "target"

    """
    X = df.drop(columns="target")
    y = df["target"].copy()
    return X.corrwith(y, method=method).abs()


def remove_correlated_custom(df: DataFrame, threshold: float = 0.95) -> DataFrame:
    """TODO: implement this to greedily combine highly-correlated features instead of just dropping"""
    # corrs = np.corrcoef(df, rowvar=False)
    # rows, cols = np.where(corrs > threshold)
    # correlated_feature_pairs = [(df.columns[i], df.columns[j]) for i, j in zip(rows, cols) if i < j]
    # for pair in correlated_feature_pairs[:10]:
    #     print(pair)
    # print(f"{len(correlated_feature_pairs)} correlated feature pairs total")
    raise NotImplementedError()


def remove_weak_features(df: DataFrame, decorrelate: bool = True) -> DataFrame:
    """Remove constant, low-information, and highly-correlated (> 0.95) features"""
    if UNCORRELATED.exists():
        cols = pd.read_json(UNCORRELATED, typ="series")
        return df.loc[:, cols].copy()

    X = df.drop(columns="target")
    y = df["target"].copy()

    print("Starting shape: ", X.shape)
    X_v = remove_single_value_features(X)
    print("Shape after removing constant features: ", X_v.shape)
    X_i = remove_low_information_features(X_v)
    print("Shape after removing low-information features: ", X_i.shape)
    if not decorrelate:
        return X_i
    print(
        "Removing highly-correlated features."
        "NOTE: this could take a while when there are 1000+ features."
    )
    X_c = remove_highly_correlated_features(X_i)
    print("Shape after removing highly-correlated features: ", X_c.shape)
    X_c.columns.to_series().to_json(UNCORRELATED)
    print(f"Saved uncorrelated feature names to {UNCORRELATED}")
    ret = X_c.copy()
    ret["target"] = y
    return ret


def select_features_by_univariate_rank(
    df: DataFrame, metric: UnivariateMetric, n_features: int = 10
) -> DataFrame:
    """Naively select features based on their univariate relation with the target variable.

    Parameters
    ----------
    df: DataFrame
        Data with target in column named "target".

    metric: "d" | "auc" | "pearson" | "spearman"
        Metric to compute for each feature

    n_features: int = 10
        How many features to select

    Returns
    -------
    reduced: DataFrame
        Data with reduced feature set.
    """
    X = df.drop(columns="target")
    y = df["target"].to_numpy()
    importances = None
    if metric.lower() == "d":
        importances = cohens_d(df).sort_values(ascending=False)
    elif metric.lower() == "auc":
        importances = auroc(df).sort_values(ascending=False)
    elif metric.lower() in ["pearson", "spearman"]:
        importances = correlations(df, method=metric).sort_values(ascending=False)
    else:
        raise ValueError("Invalid metric")
    strongest = importances[:n_features]
    reduced = df.loc[:, strongest.index]
    reduced["target"] = y
    return reduced


def get_pca_features(df: DataFrame, n_features: int = 10) -> DataFrame:
    """Return a DataFrame that is the original DataFrame projected onto the space described by first
    `n_features` principal components

    Parameters
    ----------
    df: DataFrame
        DataFrame to process

    n_features: int = 10
        Number of final features (components) to use.

    Returns
    -------
    reduced: DataFrame
        Feature-reduced DataFrame
    """
    # if you make this a series assigning to re["target"] later CREATES A FUCKING NAN
    y = df["target"].to_numpy()
    X = df.drop(columns="target")
    pca = PCA(n_features, svd_solver="full", whiten=True)
    reduced = pca.fit_transform(X)
    ret = DataFrame(data=reduced, columns=[f"pca-{i}" for i in range(reduced.shape[1])])
    ret["target"] = y
    return ret


def get_kernel_pca_features(df: DataFrame, n_features: int = 10) -> DataFrame:
    """Return a DataFrame that is reduced via KernelPCA to `n_features` principal components.

    Parameters
    ----------
    df: DataFrame
        DataFrame to process

    n_features: int = 10
        Number of final features (components) to use.

    Returns
    -------
    reduced: DataFrame
        Feature-reduced DataFrame
    """
    y = df["target"].to_numpy()
    X = df.drop(columns="target")
    kpca = KernelPCA(n_features, kernel="rbf", random_state=SEED, n_jobs=-1)
    reduced = kpca.fit_transform(X)
    ret = DataFrame(data=reduced, columns=[f"kpca-{i}" for i in range(reduced.shape[1])])
    ret["target"] = y
    return ret


def preselect_stepwise_features(
    df: DataFrame,
    classifier: Classifier,
    n_features: int = 100,
    direction: Literal["forward", "backward"] = "forward",
) -> DataFrame:
    outfile = DATADIR / f"mcic_{direction}-select{n_features}__{classifier}.json"
    selector = SequentialFeatureSelector(
        estimator=get_classifier_constructor(classifier)(),
        n_features_to_select=n_features,
        direction=direction,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    )
    X_raw = df.drop(columns="target")
    y = df["target"].to_numpy().astype(int)
    X_arr = StandardScaler().fit_transform(X_raw)
    X = DataFrame(data=X_arr, columns=X_raw.columns, index=X_raw.index)
    if classifier == "mlp":
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = "ignore"  # can't kill ConvergenceWarning any other way
    selector.fit(X, y)
    if classifier == "mlp":
        os.environ["PYTHONWARNINGS"] = before
    column_idx = selector.get_support()
    reduced = X.loc[:, column_idx].copy()
    reduced["target"] = y
    reduced.to_json(outfile)
    return reduced


def select_stepwise_features(
    df: DataFrame,
    classifier: Classifier,
    n_features: int = 10,
    direction: Literal["forward", "backward"] = "forward",
) -> DataFrame:
    outfile = DATADIR / f"mcic_{direction}-select{n_features}__{classifier}.json"
    if outfile.exists():
        return pd.read_json(outfile)
    return preselect_stepwise_features(
        df=df, classifier=classifier, n_features=n_features, direction=direction
    )
