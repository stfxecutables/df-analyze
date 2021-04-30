from sklearn.model_selection import train_test_split
from src.cleaning import get_clean_data
from src.hypertune import train_val_splits
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

VAL_SIZE = 0.20
SEED = 69

if __name__ == "__main__":

    df = get_clean_data()
    # X_train, X_val, y_train, y_val = train_val_splits(df)
    selected = remove_weak_features(df)
    # pca = get_pca_features(df, 100)
    # print(pca)
    # kpca = get_kernel_pca_features(df, 100)
    # print(kpca)

    # print(cohens_d(df))
    # print(auroc(df))
    # print(correlations(df))
    # reduced = select_features_by_univariate_rank(df, "d", 10)
    reduced = select_stepwise_features(
        selected,
        SVC(),
        n_features=10,
        direction="forward"
    )
    print(reduced)

