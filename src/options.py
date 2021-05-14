"""
File for defining all options passed to `df-analyze.py`.
"""
import os
from argparse import ArgumentError, ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast, no_type_check
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from src._constants import CLASSIFIERS, FEATURE_CLEANINGS, FEATURE_SELECTIONS, HTUNE_VAL_METHODS
from src._types import Classifier, DropNan, FeatureCleaning, FeatureSelection, ValMethod

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


class Debug:
    """Printing mixin a la https://doc.rust-lang.org/std/fmt/trait.Debug.html"""

    def __repr__(self) -> str:
        return "".join(pformat(self.__dict__, indent=2, width=80, compact=False))


@dataclass
class CleaningOptions(Debug):
    """Container for HASHABLE arguments used to check whether a memoized cleaning
    function needs to be re-computed or not. Because a change in the source file
    results in a change in the results, that file path must be duplicated here.
    """

    datapath: Path
    target: str
    feat_clean: Tuple[FeatureCleaning, ...]
    drop_nan: DropNan


@dataclass
class SelectionOptions(Debug):
    """Container for HASHABLE arguments used to check whether a memoized feature selection
    function needs to be re-computed or not. Because a change in the source file results
    in a change in the results, that file path must be duplicated here.

    Also, since feature selection depends on the cleaning choices, those must be included
    here as well. Note that *nesting does work* with immutable dataclasses and
    `joblib.Memory`.

    However, the reason we have separate classes from ProgramOptions is also that we don't
    want to e.g. recompute an expensive feature cleaning step (like removing correlated
    features), just because some set of arguments *later* in the pipeline changed.
    """

    cleaning_options: CleaningOptions
    classifiers: Tuple[Classifier, ...]
    feat_select: Tuple[FeatureSelection, ...]
    n_feat: int


class ProgramOptions(Debug):
    """Just a container for handling CLI options and default logic (while also providing better
    typing than just using the `Namespace` from the `ArgumentParser`).

    Notes
    -----
    For `joblib.Memory` to cache properly, we need all arguments to be hashable. This means
    immutable (among other things) so we use `Tuple` types for arguments or options where there are
    multiple steps to go through, e.g. feature selection.
    """

    def __init__(self, cli_args: Namespace) -> None:
        # memoization-related
        self.cleaning_options: CleaningOptions
        self.selection_options: SelectionOptions
        # other
        self.datapath: Path
        self.target: str
        self.classifiers: Tuple[Classifier, ...]
        self.htune: bool
        self.htune_val: ValMethod
        self.htune_val_size: float
        self.htune_trials: int
        self.test_val: ValMethod
        self.test_val_size: float
        self.outdir: Path

        self.datapath = self.validate_datapath(cli_args.df)
        self.outdir = self.ensure_outdir(self.datapath, cli_args.outdir)
        self.target = cli_args.target
        self.classifiers = tuple(sorted(set(cli_args.classifiers)))

        self.cleaning_options = CleaningOptions(
            datapath=self.datapath,
            target=self.target,
            feat_clean=tuple(sorted(set(cli_args.feat_clean))),
            drop_nan=cli_args.drop_nan,
        )
        self.selection_options = SelectionOptions(
            cleaning_options=self.cleaning_options,
            classifiers=self.classifiers,
            feat_select=tuple(sorted(set(cli_args.feat_select))),
            n_feat=cli_args.n_feat,
        )

        self.htune = cli_args.htune
        self.htune_val = cli_args.htune_val
        self.htune_val_size = cli_args.htune_val_size
        self.htune_trials = cli_args.htune_trials
        self.test_val = cli_args.test_val
        self.test_val_size = cli_args.test_val_size

    @staticmethod
    def validate_datapath(df_path: Path) -> Path:
        datapath = df_path
        if not datapath.exists():
            raise FileNotFoundError(f"The specified file {datapath} does not exist.")
        if not datapath.is_file():
            raise FileNotFoundError(f"The object at {datapath} is not a file.")
        return Path(datapath).resolve()

    @staticmethod
    def ensure_outdir(datapath: Path, outdir: Optional[Path]) -> Path:
        if outdir is None:
            out = f"df-analyze-results__{datapath.stem}"
            outdir = datapath.parent / out
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


def get_options(args: str = None) -> ProgramOptions:
    """parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument("--df", action="store", type=resolved_path, required=True, help=DF_HELP_STR)
    # just use existing pathname instead
    # parser.add_argument("--df-name", action="store", type=str, default="", help=DFNAME_HELP_STR)
    parser.add_argument(
        "--target", "-y", action="store", type=str, default="target", help=Y_HELP_STR
    )
    # NOTE: `nargs="+"` allows repeats, must be removed after
    parser.add_argument(
        "--classifiers", "-C", nargs="+", type=str, choices=CLASSIFIERS, default=["svm"]
    )
    parser.add_argument(
        "--feat-select", "-F", nargs="+", type=str, choices=FEATURE_SELECTIONS, default=["pca"]
    )
    parser.add_argument(
        "--feat-clean", "-f", nargs="+", type=str, choices=FEATURE_CLEANINGS, default=["constant"]
    )
    parser.add_argument("--drop-nan", "-d", choices=["all", "rows", "cols", "none"], default="none")
    parser.add_argument("--n-feat", type=int, default=10)
    parser.add_argument("--htune", action="store_true")
    parser.add_argument("--htune-val", "-H", type=str, choices=HTUNE_VAL_METHODS, default="none")
    parser.add_argument("--htune-val-size", type=cv_size, default=0)
    parser.add_argument("--htune-trials", type=int, default=100)
    parser.add_argument("--test-val", "-T", type=str, choices=HTUNE_VAL_METHODS, default="kfold")
    parser.add_argument("--test-val-size", type=cv_size, default=5)
    parser.add_argument("--outdir", type=resolved_path, default=None)
    cli_args = parser.parse_args() if args is None else parser.parse_args(args.split())
    return ProgramOptions(cli_args)
