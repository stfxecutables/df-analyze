from sklearn.model_selection import train_test_split
from src.cleaning import get_clean_data

VAL_SIZE = 0.20
SEED = 69

if __name__ == "__main__":

    df = get_clean_data()
    train, val = train_test_split(
        df, test_size=VAL_SIZE,
        random_state=SEED, shuffle=True,
        stratify=df.target,
    )
    X_train = train.drop(columns="target")
    X_val = val.drop(columns="target")
    y_train = train.target
    y_val = val.target
    print(X_train)
    print(X_val)
