from sklearn.model_selection import train_test_split
from src.cleaning import get_clean_data
from src.hypertune import train_val_splits
from src.feature_selection import remove_weak_features

VAL_SIZE = 0.20
SEED = 69

if __name__ == "__main__":

    df = get_clean_data()
    X_train, X_val, y_train, y_val = train_val_splits(df)
    selected = remove_weak_features(df)
    print(selected)

