import sys
from time import ctime
from typing import Any, Dict

import optuna
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.analyses import classifier_analysis_multitest

TEST_ARG_OPTIONS = dict(
    classifier=["rf"],
    feature_selection=["pca", "d"],
    n_features=[10, 20],
    htune_validation=[5, "mc", 0.2],
)

# SVM - 20 minutes
# RF - 4-8 hours
# DTREE - 30 minutes
# BAG - 30 minutes
# MLP - 4 hours
ARG_OPTIONS = dict(
    classifier=["svm", "rf", "dtree", "bag", "mlp"],
    # classifier=["mlp"],
    feature_selection=["pca", "kpca", "d", "auc", "pearson"],
    n_features=[20, 50, 100],
    htune_validation=[5, 10],
)
ARGS = list(ParameterGrid(ARG_OPTIONS))
TEST_ARGS = list(ParameterGrid(TEST_ARG_OPTIONS))


FEATURE_SELECTION_ANALYSES = [
    "pca",
    "kernel_pca",
    "unvariate_d",
    "unvariate_auc",
    "unvariate_pearson",
    "stewise_up",
    "stewise_down",
]

CLASSIFIERS = ["svm", "rf", "dtree", "lsq_bag", "ann-mlp"]


def pbar_desc(args: Dict[str, Any]) -> str:
    classifier = args["classifier"]
    selection = args["feature_selection"]
    n_feat = args["n_features"]
    htune_val = args["htune_validation"]
    if isinstance(htune_val, int):
        hv = f"{htune_val}-fold"
    elif isinstance(htune_val, float):
        hv = f"{int(100*htune_val)}%-holdout"
    elif htune_val == "mc":
        hv = "mc"
    else:
        hv = "none"
    return f"{classifier}|{selection}|{n_feat} features|htune_val={hv}"


if __name__ == "__main__":

    # classifier_analysis(
    #     "bag",
    #     feature_selection="pca",
    #     n_features=20,
    #     htune_trials=20,
    #     htune_validation=5,
    #     test_validation=10,
    # )
    # sys.exit()

    # ARGS = TEST_ARGS

    results = []
    pbar = tqdm(total=len(ARGS))
    for args in ARGS:
        pbar.set_description(pbar_desc(args))
        results.append(
            classifier_analysis_multitest(htune_trials=100, verbosity=optuna.logging.ERROR, **args)
        )
        pbar.update()
    df = pd.concat(results, axis=0, ignore_index=True)
    timestamp = ctime().replace(":", "-").replace("  ", " ").replace(" ", "_")
    try:
        df.to_json(f"results__{timestamp}.json")
    except Exception:
        pass
    df.to_csv(f"results__{timestamp}.csv")
    print(df)
    sys.exit()
