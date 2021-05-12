"""
File for defining all options passed to `df-analyze.py`.
"""
import os

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
from src._types import Classifier, FeatureSelection, FeatureCleaning, DropNan, ValMethod

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
    """Just a container for handling CLI options and default logic (while also providing better
    typing than just using the `Namespace` from the `ArgumentParser`).
    """

    def __init__(self, cli_args: Namespace) -> None:
        self.datapath: Path
        self.target: str
        self.classifiers: Set[Classifier]
        self.feat_select: Set[FeatureSelection]
        self.feat_clean: Set[FeatureCleaning]
        self.drop_nan: DropNan
        self.n_feat: int
        self.htune: bool
        self.htune_val: ValMethod
        self.htune_val_size: float
        self.htune_trials: int
        self.test_val: ValMethod
        self.test_val_size: float
        self.outdir: Path

        self.target = cli_args.target
        self.classifiers = set(cli_args.classifiers)
        self.feat_select = set(cli_args.feat_select)
        self.drop_nan = cli_args.drop_nan
        self.n_feat = cli_args.n_feat
        self.htune = cli_args.htune
        self.htune_val = cli_args.htune_val
        self.htune_val_size = cli_args.htune_val_size
        self.htune_trials = cli_args.htune_trials
        self.test_val = cli_args.test_val
        self.test_val_size = cli_args.test_val_size
        self.datapath = self.validate_datapath(cli_args)
        self.outdir = self.ensure_outdir(cli_args)

    def validate_datapath(self, cli_args: Namespace) -> Path:
        datapath = cli_args.df
        if not datapath.exists():
            raise FileNotFoundError(f"The specified file {datapath} does not exist.")
        if not datapath.is_file():
            raise FileNotFoundError(f"The object at {datapath} is not a file.")
        return Path(datapath).resolve()

    def ensure_outdir(self, cli_args: Namespace) -> Path:
        if cli_args.outdir is None:
            out = f"df-analyze-results__{self.datapath.stem}"
            outdir = self.datapath.parent / out
        else:
            outdir = cli_args.outdir
        if outdir.exists():
            if not outdir.is_dir():
                raise FileExistsError(
                    f"The specified output directory {outdir}"
                    "already exists and is not a directory."
                )
        else:
            os.makedirs(outdir, exist_ok=True)
        return outdir


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


def get_options() -> ProgramOptions:
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
    parser.add_argument("--drop-nan", "-d", choices=["all", "rows", "cols"], default="rows")
    parser.add_argument("--n-feat", type=int, default=-1)
    parser.add_argument("--htune", action="store_true")
    parser.add_argument("--htune-val", "-H", type=str, choices=HTUNE_VAL_METHODS, default="none")
    parser.add_argument("--htune-val-size", type=cv_size, default=0)
    parser.add_argument("--htune-trials", type=int, default=100)
    parser.add_argument("--test-val", "-T", type=str, choices=HTUNE_VAL_METHODS, default="kfold")
    parser.add_argument("--test-val-size", type=cv_size, default=5)
    parser.add_argument("--outdir", type=resolved_path, default=None)
    cli_args = parser.parse_args()
    return ProgramOptions(cli_args)
