from typing import Optional, Union
from typing_extensions import Literal

from pandas import DataFrame

from src.cleaning import get_clean_data
from src.feature_selection import (
    remove_weak_features,
    get_pca_features,
    get_kernel_pca_features,
    select_features_by_univariate_rank,
    select_stepwise_features,
    UnivariateMetric,
)
from src.hypertune import Classifier


FeatureSelection = Union[
    Literal["minimal", "step-down", "step-up", "pca", "kpca"],
    UnivariateMetric,
]
HtuneValidation = Union[int, DataFrame, Literal["loocv", "mc"]]

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
            estimator=None,
            n_features=n_features,
            direction="backward"
        )
        raise NotImplementedError()
    elif selection == "step-down":
        df = select_stepwise_features(
            df_most,
            estimator=None,
            n_features=n_features,
            direction="forward"
        )
        raise NotImplementedError()
    elif selection == "minimal":
        df = df_most
    elif selection == "none":
        df = df_all
    else:
        raise ValueError("Invalid feature selection method")
    return df


def classifier_analysis(
    feature_selection: Optional[FeatureSelection] = "pca",
    n_features: int = 20,
    htune_validation: HtuneValidation = 5,
    htune_trials: int = 200,
    classifier: Classifier = "svm",
) -> None:
    df_all = get_clean_data()
    df = select_features(df_all, feature_selection, n_features, classifier)





if __name__ == "__main__":
    pass
