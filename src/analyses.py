import optuna
from typing import Any, Optional, Union, List, Dict
from typing_extensions import Literal

from pandas import DataFrame
from sklearn.model_selection import cross_validate

from src.cleaning import get_clean_data
from src.feature_selection import (
    remove_weak_features,
    get_pca_features,
    get_kernel_pca_features,
    select_features_by_univariate_rank,
    select_stepwise_features,
    UnivariateMetric,
)
from src.hypertune import (
    CVMethod,
    Classifier,
    HtuneResult,
    evaluate_hypertuned,
    get_classifier_constructor,
    get_cv,
    hypertune_classifier,
    train_val_splits,
)


FeatureSelection = Union[
    Literal["minimal", "step-down", "step-up", "pca", "kpca"], UnivariateMetric
]
MultiTestCVMethod = Union[int, Literal["mc"]]


def select_features(
    df_all: DataFrame,
    feature_selection: Optional[FeatureSelection],
    n_features: int,
    classifier: Classifier,
) -> DataFrame:
    df_most = remove_weak_features(df_all, decorrelate=True)
    selection = str(feature_selection).lower()
    if selection == "pca":
        df = get_pca_features(df_most, n_features)
    elif selection == "kpca":
        df = get_kernel_pca_features(df_most, n_features)
    elif selection in ["d", "auc", "pearson", "spearman"]:
        df = select_features_by_univariate_rank(df_most, metric=selection, n_features=n_features)
    elif selection == "step-down":
        df = select_stepwise_features(
            df_most,
            estimator=get_classifier_constructor(classifier)(),
            n_features=n_features,
            direction="backward",
        )
        raise NotImplementedError()
    elif selection == "step-down":
        df = select_stepwise_features(
            df_most,
            estimator=get_classifier_constructor(classifier)(),
            n_features=n_features,
            direction="forward",
        )
        raise NotImplementedError()
    elif selection == "minimal":
        df = df_most
    elif selection == "none":
        df = df_all
    else:
        raise ValueError("Invalid feature selection method")
    return df


def result_to_df(result: Dict[str, Any]) -> DataFrame:
    htuned: HtuneResult = result.htuned

    row = dict(model=htuned.classifier, htune_val=htuned.cv_method)
    pass


def classifier_analysis(
    classifier: Classifier = "svm",
    feature_selection: Optional[FeatureSelection] = "pca",
    n_features: int = 20,
    htune_trials: int = 200,
    htune_validation: CVMethod = 5,
    test_validation: CVMethod = 10,
) -> None:
    """Summary

    Parameters
    ----------
    htune_validation: Union[int, float, Literal["loocv", "mc"]]
        If an `int`, specifies k-fold and the value of `k`.
        If a float in (0, 1), specifies holdout validation with test_size=`htune_validation`.
        If "loocv", specifies LOOCV validation.
        If "mc", specifies Monte-Carlo validation with 20 random 10% holdouts.

    Returns
    -------
    val1: Any
    """
    df_all = get_clean_data()
    print(f"Preparing feature selection with method: {feature_selection}")
    df = select_features(df_all, feature_selection, n_features, classifier)
    if isinstance(test_validation, float):  # set aside data for final test
        if test_validation <= 0 or test_validation >= 1:
            raise ValueError("`test_validation` must be in (0, 1)")
        X_train, X_test, y_train, y_test = train_val_splits(df, test_validation)
    else:
        X_train = df.drop(columns="target")
        X_test = None
        y_train = df["target"].copy().astype(int)
        y_test = None
    htuned = hypertune_classifier(
        classifier=classifier,
        X_train=X_train,
        y_train=y_train,
        n_trials=htune_trials,
        cv_method=htune_validation,
    )
    print(f"\n{' Testing Results ':=^80}\n")
    evaluate_hypertuned(
        htuned,
        cv_method=test_validation,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def classifier_analysis_multitest(
    classifier: Classifier = "svm",
    feature_selection: Optional[FeatureSelection] = "pca",
    n_features: int = 20,
    htune_trials: int = 200,
    htune_validation: CVMethod = 5,
    test_validations: List[MultiTestCVMethod] = [5, 10, "mc"],
    verbosity: int = optuna.logging.ERROR,
) -> None:
    """Summary

    Parameters
    ----------
    htune_validation: Union[int, float, Literal["loocv", "mc"]]
        If an `int`, specifies k-fold and the value of `k`.
        If a float in (0, 1), specifies holdout validation with test_size=`htune_validation`.
        If "loocv", specifies LOOCV validation.
        If "mc", specifies Monte-Carlo validation with 20 random 10% holdouts.

    Returns
    -------
    val1: Any
    """
    log = verbosity != optuna.logging.ERROR
    df_all = get_clean_data()
    if log:
        print(f"Preparing feature selection with method: {feature_selection}")
    df = select_features(df_all, feature_selection, n_features, classifier)
    X_train = df.drop(columns="target")
    y_train = df["target"].to_numpy().astype(int)
    htuned = hypertune_classifier(
        classifier=classifier,
        X_train=X_train,
        y_train=y_train,
        n_trials=htune_trials,
        cv_method=htune_validation,
        verbosity=verbosity,
    )
    if verbosity != optuna.logging.ERROR:
        print(f"\n{' Testing Results ':=^80}\n")
    results = []
    for test_validation in test_validations:
        result = evaluate_hypertuned(
            htuned, cv_method=test_validation, X_train=X_train, y_train=y_train, log=log
        )
        results.append(result_to_df(result))


if __name__ == "__main__":
    pass