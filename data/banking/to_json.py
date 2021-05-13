from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    # target is called "y"
    FILE = Path(__file__).resolve().parent / "bank.csv"
    df = pd.read_csv(FILE, sep=";", true_values=["yes"], false_values=["no"], na_values=["unknown"])
    print(df)
    dumbed = pd.get_dummies(df, drop_first=False)
    print(dumbed.dtypes)
    X = dumbed.drop(columns="y")
    y = dumbed["y"].to_numpy()
    X["y"] = y
    print(X)
    X.to_json(FILE.parent / "bank.json")
