# NOTE: feature selection is PRIOR to hypertuning, but "what features are best" is of course
# contengent on the choice of regressor / classifier. The correct way to frame this is as an overall
# derivative-free optimization problem where the classifier choice is *just another hyperparameter*.
# However, this is not possible for stepwise selection methods, which already take 5-20 hours even
# without incorporating any tuning.

import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from featuretools.selection import (
    remove_highly_correlated_features,
    remove_low_information_features,
    remove_single_value_features,
)
from pandas import DataFrame, Series
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing_extensions import Literal

from src._constants import DATADIR, SEED, UNCORRELATED
from src._setup import MEMOIZER
from src._types import (
    Classifier,
    CorrMethod,
    Estimator,
    FeatureSelection,
    Regressor,
    UnivariateMetric,
)
from src.classifiers import get_classifier_constructor
from src.cleaning import get_clean_data
from src.cli import CleaningOptions, ProgramOptions, SelectionOptions
from src.io import FileType, try_save
from src.regressors import get_regressor_constructor
from src.sklearn_pasta._sequential import SequentialFeatureSelector

# FEATURE_CACHE = GLOBALS.JOBLIB_CACHE_DIR / "__features__"
# MEMOIZER = Memory(location=FEATURE_CACHE, backend="local", compress=9)


def cohens_d(df: DataFrame, target: str) -> Series:
    """For each feature in `df`, compute the absolute Cohen's d values.

    Parameters
    ----------
    df: DataFrame
        DataFrame with shape (n_samples, n_features + 1) and target variable in column
        `target`

    Returns
    -------
    ds: Series
        Cohen's d values for each feature, as a Pandas Series.
    """
    X = df.drop(columns=target)
    y = df[target].copy()
    x1, x2 = X.loc[y == 0, :], X.loc[y == 1, :]
    n1, n2 = len(x1) - 1, len(x2) - 1
    sd1, sd2 = np.std(x1, ddof=1, axis=0), np.std(x2, ddof=1, axis=0)
    sd_pools = np.sqrt((n1 * sd1 + n2 * sd2) / (n1 + n2))
    m1, m2 = np.mean(x1, axis=0), np.mean(x2, axis=0)
    ds = np.abs(m1 - m2) / sd_pools
    return ds


def auroc(df: DataFrame, target: str) -> Series:
    """For each feature in `df` compute rho, the common-language effect size (see Notes)
    via the area-under-the-ROC curve (AUC), and rescale this effect size to allow sorting
    across features.

    Parameters
    ----------
    df: DataFrame
        DataFrame with shape (n_samples, n_features + 1) and target variable in column
        `target`

    Returns
    -------
    rescaled_aucs: Series
        The rescaled AUC values (see notes) for each feature.

    Notes
    -----
    This is equivalent to calculating U / abs(y1*y2), where `y1` and `y2` are the subgroup
    sizes, and U is the Mann-Whitney U, and is also sometimes referred to as Herrnstein's
    rho [1] or "f", the "common-language effect size" [2].

    Note we also *must* rescale as rho is implicity "signed", with values above 0.5
    indicating separation in one direction, and values below 0.5 indicating separation in
    the other.

    [1] Herrnstein, R. J., Loveland, D. H., & Cable, C. (1976). Natural concepts in
        pigeons. Journal of Experimental Psychology: Animal Behavior Processes, 2, 285-302

    [2] McGraw, K.O.; Wong, J.J. (1992). "A common language effect size statistic".
        Psychological Bulletin. 111 (2): 361–365. doi:10.1037/0033-2909.111.2.361.
    """
    X = df.drop(columns=target)
    y = df[target].copy()

    aucs = Series(data=[roc_auc_score(y, X[col]) for col in X], index=X.columns)
    rescaled = (aucs - 0.5).abs()
    return rescaled


def correlations(df: DataFrame, target: str, method: CorrMethod = "pearson") -> Series:
    """For each feature in `df` compute the ABSOLUTE correlation with the target variable.

    Parameters
    ----------
    df: DataFrame
        DataFrame with shape (n_samples, n_features + 1) and target variable in column `target`

    method: "pearson" | "spearman" | "kendall"
        How to correlate.

    Returns
    -------
    corrs: Series
        Correlations for each feature.
    """
    X = df.drop(columns=target)
    y = df[target].copy()
    return X.corrwith(y, method=method).abs()


def remove_correlated_custom(df: DataFrame, target: str, threshold: float = 0.95) -> DataFrame:
    """TODO: implement this to greedily combine highly-correlated features instead of just
    dropping"""
    # corrs = np.corrcoef(df, rowvar=False)
    # rows, cols = np.where(corrs > threshold)
    # correlated_pairs = [(df.columns[i], df.columns[j]) for i, j in zip(rows, cols) if i < j]
    # for pair in correlated_pairs[:10]:
    #     print(pair)
    # print(f"{len(correlated_pairs)} correlated feature pairs total")
    raise NotImplementedError()


def remove_weak_features(options: SelectionOptions) -> DataFrame:
    """Remove constant, low-information, and highly-correlated (> 0.95) features using the
    featuretools (https://www.featuretools.com/) Python API. This should be run *first*
    before other feature selection method.

    Parameters
    ----------
    df: DataFrame The DataFrame with all the original features that you desire to perform feature
        selection on. Should have a column named `target` which contains the value to be classified
        / predicted.

    decorrelate: bool = True
        If True, use `featuretools.remove_highly_correlated_features` to eliminate features
        correlated at over 0.95. NOTE: this process can be quite slow.

    Returns
    -------
    df_selected: DataFrame
        Copy of data with selected columns. Also still includes the `target` column.
    """

    df = get_clean_data(options.cleaning_options)
    target = options.cleaning_options.target
    X = df.drop(columns=target)
    y_orig = df[target].to_numpy()

    print("Starting shape: ", X.shape)
    if "constant" in options.cleaning_options.feat_clean:
        X = remove_single_value_features(X)
        print("Shape after removing constant features: ", X.shape)
    if "lowinfo" in options.cleaning_options.feat_clean:
        X = remove_low_information_features(X)
        print("Shape after removing low-information features: ", X.shape)
    if "correlated" in options.cleaning_options.feat_clean:
        print(
            "Removing highly-correlated features."
            "NOTE: this could take a while when there are 1000+ features."
        )
        X = remove_highly_correlated_features(X)
    print("Shape after removing highly-correlated features: ", X.shape)
    X[target] = y_orig
    return X


def select_features_by_univariate_rank(
    df: DataFrame, target: str, metric: UnivariateMetric, n_feat: int
) -> Series:
    """Naively select features based on their univariate relation with the target variable.

    Parameters
    ----------
    df: DataFrame
        Data with target in column named `target`.

    metric: "d" | "auc" | "pearson" | "spearman"
        Metric to compute for each feature, between that feature and the target.

    n_feat: int = 10
        How many features to select

    Returns
    -------
    reduced: DataFrame
        Data with reduced feature set.
    """
    # y = df[target].to_numpy()
    importances = None
    if metric.lower() == "d":
        importances = cohens_d(df, target).sort_values(ascending=False)
    elif metric.lower() == "auc":
        importances = auroc(df, target).sort_values(ascending=False)
    elif metric.lower() in ["pearson", "spearman"]:
        importances = correlations(df, target, method=metric).sort_values(ascending=False)  # noqa # type: ignore
    else:
        raise ValueError("Invalid metric")
    strongest = importances[:n_feat]
    # reduced = df.loc[:, strongest.index]
    # reduced[target] = y
    # return reduced
    return strongest


def pca_reduce(df: DataFrame, target: str, n_features: int = 10) -> DataFrame:
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
    # if you make this a series assigning to re[target] later CREATES A FUCKING NAN
    y = df[target].to_numpy()
    X = df.drop(columns=target)
    pca = PCA(n_features, svd_solver="full", whiten=True)
    reduced = pca.fit_transform(X)
    ret = DataFrame(data=reduced, columns=[f"pca-{i}" for i in range(reduced.shape[1])])
    ret[target] = y
    return ret


def kernel_pca_reduce(df: DataFrame, target: str, n_features: int = 10) -> DataFrame:
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

    Notes
    -----
    This seemed to result in some *very* poor classifier performance so the hyperparams here likely
    ought to be tuned and/or selected much more carefully (TODO).
    """
    y = df[target].to_numpy()
    X = df.drop(columns=target)
    kpca = KernelPCA(n_features, kernel="rbf", random_state=SEED, n_jobs=-1)
    reduced = kpca.fit_transform(X)
    ret = DataFrame(data=reduced, columns=[f"kpca-{i}" for i in range(reduced.shape[1])])
    ret[target] = y
    return ret


@MEMOIZER.cache
def select_stepwise_features(
    df: DataFrame,
    target: str,
    mode: Literal["classify", "regress"],
    estimator: Estimator,
    n_features: int,
    direction: Literal["forward", "backward"] = "forward",
) -> DataFrame:
    """Perform stepwise feature selection (which takes HOURS) and save the selected features in a
    DataFrame so that subsequent runs do not need to do this again."""

    # outfile = DATADIR / f"mcic_{direction}-select{n_features}__{classifier}.json"
    get_model = get_classifier_constructor if mode == "classify" else get_regressor_constructor
    selector = SequentialFeatureSelector(
        estimator=get_model(estimator)(),
        n_features_to_select=n_features,
        direction=direction,
        scoring="accuracy" if mode == "classify" else "neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
    )
    X_raw = df.drop(columns=target)
    y = df[target].to_numpy()
    if mode == "classify":
        y = y.astype(int)
    X_arr = StandardScaler().fit_transform(X_raw)
    X = DataFrame(data=X_arr, columns=X_raw.columns, index=X_raw.index)
    if estimator == "mlp":
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = "ignore"  # can't kill ConvergenceWarning any other way
    selector.fit(X, y)
    if estimator == "mlp":
        os.environ["PYTHONWARNINGS"] = before
    column_idx = selector.get_support()
    # reduced = X.loc[:, column_idx].copy()
    # reduced[target] = y
    # reduced.to_json(outfile)
    # return reduced
    return column_idx


# DO NOT MEMOIZE
def select_features(
    options: ProgramOptions, feature_selection: Optional[FeatureSelection], classifier: Classifier
) -> DataFrame:
    """Dispatch function to handle executing a single feature selection option

    Parameters
    ----------
    df_all: DataFrame
        The DataFrame with all the original features that you desire to perform feature selection
        on. Should have a column named `target` which contains the value to be classified /
        predicted.

    feature_selection: Optional[FeatureSelection]
        How to select features.

    n_features: int
        Number of features (columns) to select.

    classifier: Classifier
        Classifier to use during selection. Only currently relevant if `feature_selection` is
        stepwise.

    Returns
    -------
    df_selected: DataFrame
        Copy of data with selected columns. Also still includes the `target` column.
    """
    df = remove_weak_features(options)
    target = options.cleaning_options.target
    y = df[options.cleaning_options.target].to_numpy()

    n_feat = options.selection_options.n_feat
    method = str(feature_selection).lower()
    if method == "pca":
        df_selected = pca_reduce(df, target, n_feat)
    elif method == "kpca":
        df_selected = kernel_pca_reduce(df, target, n_feat)
    elif method in ["d", "auc", "pearson", "spearman"]:
        features = select_features_by_univariate_rank(df, target, metric=method, n_feat=n_feat)
        df_selected = df.loc[:, features.index]
        df_selected[target] = y
    elif method == "step-down":
        # raise NotImplementedError()
        column_idx = select_stepwise_features(
            df,
            target,
            mode=options.mode,
            estimator=classifier,
            n_features=n_feat,
            direction="backward",
        )
        df_selected = df.drop(columns=target).loc[:, column_idx]
        df_selected[target] = y
    elif method == "step-up":
        column_idx = select_stepwise_features(
            df,
            target,
            mode=options.mode,
            estimator=classifier,
            n_features=n_feat,
            direction="forward",
        )
        df_selected = df.drop(columns=target).loc[:, column_idx]
        df_selected[target] = y
    elif method in ["minimal", "none"]:
        df_selected = df
    else:
        raise ValueError("Invalid feature selection method")

    # For pca and kpca, features are in fact new data, so values must
    # be saved as well. Otherwise, it is sufficient to save the names.
    if method in ["pca", "kpca"]:
        try_save(
            program_dirs=options.program_dirs,
            df=df_selected,
            file_stem=f"{classifier}_features",
            file_type=FileType.Feature,
            selection=method,
            cleaning=options.feat_clean,
        )
    else:
        features = pd.Series(
            name="features", data=df_selected.drop(columns=options.target).columns.to_list()
        )
        try_save(
            program_dirs=options.program_dirs,
            df=features,
            file_stem=f"{classifier}_features",
            file_type=FileType.Feature,
            selection=method,
            cleaning=options.feat_clean,
        )

    return df_selected
