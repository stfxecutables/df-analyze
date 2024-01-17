import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostRegressor as ADA
from sklearn.ensemble import GradientBoostingRegressor as GB
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
OUTFILE = ROOT / "data.json"

if __name__ == "__main__":
    X, y = make_regression(
        n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=3
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    cols.append("target")
    df = DataFrame(data=np.concatenate([X, y.reshape(-1, 1)], axis=1), columns=cols)
    print(df)
    print(df.describe())
    df.to_json(OUTFILE)
    print(f"Saved data to {OUTFILE}")
    response = input("Continue to analysis of generated data? [y/N]\n")
    if "y" not in response.lower():
        sys.exit()

    grid = list(
        ParameterGrid(
            dict(
                loss=["huber"],
                learning_rate=[0.5, 0.25, 0.1, 0.05, 0.01],
                n_estimators=[200, 500],
                max_depth=[1, 2, 3],
                alpha=[0.99, 0.9, 1e-1, 1e-2],
            )
        )
    )

    for args in tqdm(grid):
        # args={'alpha': 0.99, 'learning_rate': 0.25, 'loss': 'huber', 'max_depth': 1, 'n_estimators': 500}:
        # [23.554 26.406 24.561 27.707 27.698] (mean = 25.9852)
        result = -np.array(
            cross_val_score(
                GB(**args),
                X,
                y,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                cv=5,
            )
        )
        result = np.round(result, 3)
        print(f"GradientBoosting with args={args}: {result} (mean = {np.mean(result)})")

    # For AdaBoost, we want learning_rate=2, n_estimators=200, loss="linear" to acheive MAEs
    # across folds of [90.031 79.197 76.504 80.566 83.373] (mean = 81.9342)
    result = -np.array(
        cross_val_score(
            ADA(n_estimators=200, learning_rate=2, loss="linear"),
            X,
            y,
            scoring="neg_mean_absolute_error",
            n_jobs=5,
            cv=5,
        )
    )
    result = np.round(result, 3)
    print(
        f"AdaBoost with n_estimators=200, lr=2, loss='linear': {result} (mean = {np.mean(result)})"
    )

    # A quick test shows alpha=0.001, lr=0.01 allows a cross-validated MAE of around 21 for this data
    # MAEs: [21.476 21.616 20.092 19.453 20.805] (mean = 20.6884)
    result = -np.array(
        cross_val_score(
            MLPRegressor((256, 256, 256, 256), alpha=0.001, max_iter=200, learning_rate_init=0.01),
            X,
            y,
            scoring="neg_mean_absolute_error",
            n_jobs=5,
            cv=5,
        )
    )
    result = np.round(result, 3)
    print(f"MLPRegressor lr={0.01}: {result} (mean = {np.mean(result)})")
