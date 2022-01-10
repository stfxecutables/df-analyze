import os
from pathlib import Path
from time import ctime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pytest
from numpy import ndarray
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Series
from tqdm import tqdm
from typing_extensions import Literal
from sklearn.model_selection import ParameterGrid

from src._types import Classifier
from src.analyses import classifier_analysis_multitest
from src.options import get_options

FloatArray = NDArray[np.floating]


def pbar_desc(args: Dict[str, Any]) -> str:
    classifier = args["classifier"] if
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


def run_analysis(args: List[Dict], classifier: Classifier, step: bool = False) -> pd.DataFrame:
    results = []
    pbar = tqdm(total=len(args))
    for arg in args:
        pbar.set_description(pbar_desc(arg))
        results.append(
            classifier_analysis_multitest(htune_trials=100, verbosity=optuna.logging.ERROR, **arg)
        )
        pbar.update()
    df = pd.concat(results, axis=0, ignore_index=True)
    df.sort_values(by="acc", ascending=False, inplace=True)
    timestamp = ctime().replace(":", "-").replace("  ", " ").replace(" ", "_")
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        os.makedirs(results_dir, exist_ok=True)
    file_info = f"results__{classifier}{'_step-up' if step else ''}__{timestamp}"
    json = results_dir / f"{file_info}.json"
    csv = results_dir / f"{file_info}.csv"
    try:
        df.to_json(json)
    except Exception:
        pass
    df.to_csv(csv)
    print(df.sort_values(by="acc", ascending=False).to_markdown(tablefmt="simple", floatfmt="0.3f"))
    return df


if __name__ == "__main__":
    # TODO:
    # get arguments
    # run analyses based on args
    options = get_options()
    classifiers = options.classifiers

    stepup = "step-up" in options.selection_options.feat_select

    n_features = [10, 50, 100]
    if stepup and options.classifiers == "mlp":
        n_features = [10, 50]  # 100 features will probably take about 30 hours

    ARG_OPTIONS = dict(
        classifier=[classifier],
        feature_selection=["step-up"] if stepup else SELECTIONS,
        n_features=n_features,
        htune_validation=[5],
    )
    ARGS = list(ParameterGrid(ARG_OPTIONS))
    run_analysis(ARGS, classifier, stepup)
