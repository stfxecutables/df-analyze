from typing import Tuple

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.cleaning import get_clean_data


VAL_SIZE = 0.20
SEED = 69

def train_val_splits(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    train, val = train_test_split(
        df, test_size=VAL_SIZE,
        random_state=SEED, shuffle=True,
        stratify=df.target,
    )
    X_train = train.drop(columns="target")
    X_val = val.drop(columns="target")
    y_train = train.target
    y_val = val.target
    return X_train, X_val, y_train, y_val

# need a validation set for hypertuning and also a training set...
# but given number of samples this is not very viable
# Do subject-level k-fold for hparam tuning? Or keep a final test set to evaluate?
if __name__ == "__main__":

    df = get_clean_data()
    X_train, X_val, y_train, y_val = train_val_splits(df)