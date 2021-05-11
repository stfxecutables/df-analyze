"""
File for defining all options passed to `df-analyze.py`.
"""
from argparse import ArgumentParser, Namespace, ArgumentError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series

from src._constants import CLASSIFIERS, FEATURE_CLEANINGS, FEATURE_SELECTIONS, HTUNE_VAL_METHODS
from src._types import FeatureSelection
from src.hypertune import CVMethod, Classifier

DF_HELP_STR = """
The dataframe to analyze.

    Currently only Pandas `DataFrame` objects saved as either `.json` or `.csv`,
    or NumPy `ndarray`s saved as "<filename>.npy" are supported, but a Pandas
    `DataFrame` is recommended.

    If your data is saved as a Pandas `DataFrame`, it must have shape
    `(n_samples, n_features)` or `(n_samples, n_features + 1)`. The name of the
    column holding the target variable (or feature) can be specified by the
    `--target` / `-y` argument, but is "target" by default if such a column name
    exists, or the last column if it does not.

    If your data is in a NumPy array, the array must have the shape
    `(n_samples, n_features + 1)` where the last column is the target for either
    classification or prediction.
"""
DFNAME_HELP_STR = """
A unique identifier for your DataFrame to use when saving outputs. If unspecified,
a name will be generated based on the filename passed to `--df`.
"""

Y_HELP_STR = """
The location of the target variable for either regression or classification.

    If a string, then `--df` must be a Pandas `DataFrame` and the string passed
    in here specifies the name of the column holding the targer variable.

    If an integer, and `--df` is a NumPy array only, specifies the column index.

    (default: %(default))
"""

HTUNEVAL_HELP_STR = """
If an
"""

class ProgramOptions:
    def __init__(self, cli_args: Namespace) -> None:
        self.datapath: Path
        self.target: str
        self.classifiers: Set[Classifier]
        self.feat_select: Set[FeatureSelection]
        self.feat_clean:
        self.drop_nan:
        self.n_feat:
        self.htune:
        self.htune_val:
        self.htune_val_size:
        self.htune_trials:
        self.test_val:
        self.test_val_size:
        self.outdir:
        pass

def resolved_path(p: str) -> Path:
    return Path(p).resolve()


def cv_size(cv_str: str) -> float:
    try:
        cv = float(cv_str)
    except Exception as e:
        raise ArgumentError("Could not convert `--htune-val-size` argument to float") from e
    if cv <= 0:
        raise ArgumentError("`--htune-val-size` must be positive")
    if 0 < cv < 1:
        return cv
    if cv == 1:
        raise ArgumentError("`--htune-val-size=1` is invalid.")
    if cv > 10:
        warn("`--htune-val-size` greater than 10 is not recommended.", category=UserWarning)
    return cv


def get_cli_args() -> Namespace:
    """parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument("--df", action="store", type=resolved_path, required=True, help=DF_HELP_STR)
    # just use existing pathname instead
    # parser.add_argument("--df-name", action="store", type=str, default="", help=DFNAME_HELP_STR)
    parser.add_argument(
        "--target", "-y", action="store", type=str, default="target", help=Y_HELP_STR
    )
    # NOTE: `nargs="+"` allows repeats, must be removed after
    parser.add_argument("--classifiers", "-C", nargs="+", type=str, choices=CLASSIFIERS)
    parser.add_argument("--feat-select", "-F", nargs="+", type=str, choices=FEATURE_SELECTIONS)
    parser.add_argument("--feat-clean", "-f", nargs="+", type=str, choices=FEATURE_CLEANINGS)
    parser.add_argument("--drop-nan", "-d", choices=["all", "rows", "cols"], default="all")
    parser.add_argument("--n-feat", type=int, default=-1)
    parser.add_argument("--htune", action="store_true")
    parser.add_argument("--htune-val", "-H", type=str, choices=HTUNE_VAL_METHODS, default="none")
    parser.add_argument("--htune-val-size", type=cv_size, default=0)
    parser.add_argument("--htune-trials", type=int, default=100)
    parser.add_argument("--test-val", "-T", type=str, choices=HTUNE_VAL_METHODS, default="kfold")
    parser.add_argument("--test-val-size", type=cv_size, default=5)
    parser.add_argument(
        "--outdir", type=resolved_path, default=Path.cwd().resolve() / "df-analyze_results"
    )
    return parser.parse_args()
