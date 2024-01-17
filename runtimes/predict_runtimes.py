from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import ParameterGrid

from src.models.lgbm import LightGBMRegressor

FILES = sorted(Path(__file__).resolve().parent.glob("*.txt"))


def extrapolate() -> None:
    grid = {
        "N_sub": [100, 200, 500, 1000, 2000, 5000],
        "p": [10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
        "n_iter": [10],
    }
    args = [
        DataFrame({"N_sub": arg["N_sub"], "p": arg["p"], "n_iter": arg["n_iter"]}, index=[i])
        for i, arg in enumerate(ParameterGrid(grid))
    ]
    X_test = pd.concat(args, axis=0)
    for file in FILES:
        df = pd.read_csv(file, sep=r" +", engine="python")
        y = df["minutes"]
        X = df.drop(columns=["dsname", "minutes", "N"])
        model = LightGBMRegressor()
        model.htune_optuna(X, y, n_trials=200)
        preds = model.tuned_predict(X)
        mae = np.mean(np.abs(preds - y))
        extrapolate = Series(data=model.tuned_predict(X_test), name="minutes")
        df_ex = pd.concat([X_test, extrapolate], axis=1)
        print(file)
        print(f"MAE: {mae.round(3)}")
        print("Extrapolated:")
        print(
            df_ex.round(3).sort_values(by="minutes", ascending=False).to_markdown(tablefmt="simple")
        )


if __name__ == "__main__":
    # extrapolate()
    dfs = []
    for file in FILES:
        df = pd.read_csv(file, sep=r" +", engine="python")
        fname = file.stem.replace("select_runtime_", "").replace("_estimates", "")
        splits = fname.split("_")
        model, direction = splits[:2]
        extra = "no" if "no_subsample" in fname else "yes"
        df.insert(0, "subsample", extra)
        df.insert(0, "direction", direction)
        df.insert(0, "model", model)
        df["dsname"] = df["dsname"].apply(lambda s: s[:20])
        print(df.round(3).sort_values(by="minutes", ascending=False).to_markdown(tablefmt="simple"))
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    largest = (
        df.groupby(["model", "direction", "subsample"])
        .apply(lambda grp: grp.nlargest(5, "minutes"))
        .reset_index(drop=True)
        .sort_values(
            by=["model", "direction", "subsample", "minutes"], ascending=[True, False, False, False]
        )
    )
    smallest = (
        df.groupby(["model", "direction", "subsample"])
        .apply(lambda grp: grp.nsmallest(5, "minutes"))
        .reset_index(drop=True)
        .sort_values(
            by=["model", "direction", "subsample", "minutes"], ascending=[True, False, False, False]
        )
    )
    idx_keep = df["minutes"] != np.inf

    desc = (
        df.loc[idx_keep]
        .drop(columns=["dsname", "N", "N_sub", "p", "n_iter"])
        .groupby(["model", "direction", "subsample"])
        .describe(percentiles=[0.05, 0.25, 0.75, 0.95])
        .reset_index()
        .drop(columns=[("minutes", "count"), ("minutes", "std")])
        .round(0)
    )
    desc.columns = [
        "model",
        "direction",
        "subsample",
        "mean",
        "min",
        "5%",
        "25%",
        "50%",
        "75%",
        "95%",
        "max",
    ]
    desc = desc.map(lambda x: int(x) if isinstance(x, float) else x)
    print(largest.to_markdown(tablefmt="simple", index=False))
    print(smallest.to_markdown(tablefmt="simple", index=False))
    print(desc.to_markdown(tablefmt="simple", index=False))
    print(desc)
