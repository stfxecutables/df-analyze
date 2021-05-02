import sys
from argparse import ArgumentParser
from pathlib import Path
from time import ctime
from typing import Any, Dict

import optuna
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.analyses import classifier_analysis_multitest
from src.hypertune import Classifier

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


def get_classifier() -> Classifier:
    CLASSIFIERS = ["svm", "rf", "dtree", "bag", "mlp"]
    parser = ArgumentParser()
    parser.add_argument("--classifier", choices=CLASSIFIERS, default="svm")
    args = parser.parse_args()
    return args.classifier


if __name__ == "__main__":
    # SVM - 20 minutes
    # RF - 4-8 hours
    # DTREE - 30 minutes
    # BAG - 30 minutes
    # MLP - 4 hours
    classifier = get_classifier()
    ARG_OPTIONS = dict(
        classifier=[classifier],
        # classifier=["mlp"],
        feature_selection=["pca", "kpca", "d", "auc", "pearson"],
        n_features=[20, 50, 100],
        htune_validation=[5, 10],
    )
    ARGS = list(ParameterGrid(ARG_OPTIONS))

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
    df.sort_values(by="acc", ascending=False, inplace=True)
    timestamp = ctime().replace(":", "-").replace("  ", " ").replace(" ", "_")
    json = Path(__file__).parent / f"results__{classifier}__{timestamp}.json"
    csv = Path(__file__).parent / f"results__{classifier}__{timestamp}.csv"
    try:
        df.to_json(json)
    except Exception:
        pass
    df.to_csv(csv)
    print(df.sort_values(by="acc", ascending=False).to_markdown(tablefmt="simple", floatfmt="0.3f"))
    sys.exit()
