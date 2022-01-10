import os
from argparse import ArgumentParser
from pathlib import Path
from time import ctime
from typing import Any, Dict, List, Tuple

import optuna
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src._constants import CLASSIFIERS
from src.analyses import full_estimator_analysis
from src.hypertune import Classifier
from src.options import get_options

IN_CCANADA = os.environ.get("CC_CLUSTER") is not None
IN_CC_JOB = os.environ.get("SLURM_TMPDIR") is not None
DISABLE_PBAR = IN_CCANADA and IN_CC_JOB

SELECTIONS = ["pca", "kpca", "d", "auc", "pearson"]


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


# def get_options() -> Tuple[Classifier, bool]:
#     parser = ArgumentParser()
#     parser.add_argument("--classifier", choices=CLASSIFIERS, default="svm")
#     parser.add_argument("--step-up", action="store_true")
#     args = parser.parse_args()
#     return args.classifier, args.step_up


def run_analysis(args: List[Dict], classifier: Classifier, step: bool = False) -> pd.DataFrame:
    results = []
    pbar = tqdm(total=len(args))
    for arg in args:
        pbar.set_description(pbar_desc(arg))
        results.append(
            full_estimator_analysis(htune_trials=100, verbosity=optuna.logging.ERROR, **arg)
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
    """
    Runtime over-estimates for Compute Canada
    SVM - 20 minutes
    RF - 4-8 hours
    DTREE - 30 minutes
    BAG - 30 minutes
    MLP - 4 hours


    Empirical runtimes for step-up selection:

    __________  __________   _______  ______
    Classifier  n_features    Time    Factor
    __________  __________   _______  ______
           svm          10      5:00      x1   (10/5)**1.55 = 2.92
           svm          50     35:00      x7   (50/5)**1.55 = 35.5
           svm         100   2:30:00     x15  (100/5)**1.55 = 103.9
           mlp          10     ~2-3h      x1
           mlp       10,50   7:30:00      x3
           mlp         100  15:00:00      x8
            rf          10     ~2-3h      x1
            rf         100       ???     x15
            rf   10,50,100  12:00:00
           bag          10     20:00
           bag   10,50,100   6:30:00
         dtree          10      5:00      x1
         dtree          50     40:00      x7
         dtree   10,50,100   2:00:00
    """

    classifier, stepup = get_options()
    n_features = [10, 50, 100]
    if stepup and classifier == "mlp":
        n_features = [10, 50]  # 100 features will probably take about 30 hours

    ARG_OPTIONS = dict(
        classifier=[classifier],
        feature_selection=["step-up"] if stepup else SELECTIONS,
        n_features=n_features,
        htune_validation=[5],
    )
    ARGS = list(ParameterGrid(ARG_OPTIONS))
    run_analysis(ARGS, classifier, stepup)
