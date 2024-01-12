from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import ParameterGrid
from typing_extensions import Literal

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
        df["file"] = file.stem.replace("select_runtime_", "")
        y = df["minutes"]
        print(df.round(3).sort_values(by="minutes", ascending=False).to_markdown(tablefmt="simple"))
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    largest = df.groupby("file").apply(lambda grp: grp.nlargest(5, "minutes")).drop(columns="file")
    print(largest.to_markdown(tablefmt="simple"))
