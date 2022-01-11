"""
File for defining all options passed to `df-analyze.py`.
"""
import os
from argparse import ArgumentError, ArgumentParser, Namespace, RawTextHelpFormatter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast, no_type_check
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from src._constants import (
    CLASSIFIERS,
    FEATURE_CLEANINGS,
    FEATURE_SELECTIONS,
    HTUNE_VAL_METHODS,
    REGRESSORS,
)
from src._types import (
    Classifier,
    DropNan,
    EstimationMode,
    FeatureCleaning,
    FeatureSelection,
    Regressor,
    ValMethod,
)
from src.io import ProgramDirs, setup_io
from src.utils import Debug

DF_HELP_STR = """
The dataframe to analyze.

Currently only Pandas `DataFrame` objects saved as either `.json` or `.csv`, or
NumPy `ndarray`s saved as "<filename>.npy" are supported, but a Pandas
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

TARGET_HELP_STR = """
The location of the target variable for either regression or classification.

If a string, then `--df` must be a Pandas `DataFrame` and the string passed in
here specifies the name of the column holding the targer variable.

If an integer, and `--df` is a NumPy array only, specifies the column index.

"""

MODE_HELP_STR = """
If "classify", do classification. If "regress", do regression.

"""

# DEPRECATE
DFNAME_HELP_STR = """
A unique identifier for your DataFrame to use when saving outputs. If
unspecified, a name will be generated based on the filename passed to `--df`.

"""


CLS_HELP_STR = f"""
The list of classifiers to use when comparing classification performance.
Can be a list of elements from {sorted(CLASSIFIERS)}.

"""

REG_HELP_STR = f"""
The list of regressors to use when comparing regression model performance.
Can be a list of elements from {sorted(REGRESSORS)}.

"""

FEAT_SELECT_HELP = """
The feature selection methods to use. Available options are:

  auc:        Select features with largest AUC values relative to the two
              classes (classification only).

  d:          Select features with largest Cohen's d values relative to the two
              classes (classification only).

  kpca:       Generate features by using largest components of kernel PCA.

  pca:        Generate features by using largest components from a PCA.

  pearson:    Select features with largest Pearson correlations with target.

  step-up:    Use step-up features selection. Costly.

"""

FEAT_CLEAN_HELP = """
If specified, which feature cleaning methods to use prior to feature selection.
Makes use of the featuretools library (featuretools.com). Options are:

  correlated: remove highly correlated features using featuretools.
  constant:   remove constant (zero-variance) features. Default.
  lowinfo:    remove "low information" features via featuretools.

"""

NAN_HELP = """
How to drop NaN values. Uses Pandas options.

  none:       Do not remove. Will cause errors for most algorithms. Default.
  all:        Remove the sample and feature both (row and column) for any NaN.
  rows:       Drop samples (rows) that contain one or more NaN values.
  cols:       Drop features (columns) that contain one or more NaN values.

"""

N_FEAT_HELP = """
Number of features to select using method specified by --feat-select. NOTE:
specifying values greater than e.g. 10-50 with --feat-select=step-up and slower
algorithms can easily result in compute times of many hours.

"""

HTUNE_HELP = """

"""

HTUNEVAL_HELP_STR = """
If an

"""

DESC = f"""
{DF_HELP_STR}
{TARGET_HELP_STR}
{CLS_HELP_STR}
"""

USAGE_EXAMPLES = """
USAGE EXAMPLE (assumes you have run `poetry shell`):

    python df-analyze.py \\
        --df="weather_data.json" \\
        --target='temperature' \\
        --mode=regress \\
        --regressors=svm linear \\
        --drop-nan=rows \\
        --feat-clean=constant \\
        --feat-select=pca pearson \\
        --n-feat=5 \\
        --htune \\
        --test-val=kfold \\
        --test-val-size=5 \\
        --outdir='./results'

"""


class Verbosity(Enum):
    """
    Properties
    ----------
    ERROR
        Only log errors.

    INFO
        Log results of each full hyperparameter tuning and other interim progress bars.

    DEBUG
        Maximum level of logging.
    """

    ERROR = 0
    INFO = 1
    DEBUG = 2


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
    mode: EstimationMode
    classifiers: Tuple[Classifier, ...]
    regressors: Tuple[Regressor, ...]
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
        self.mode: EstimationMode
        self.classifiers: Tuple[Classifier, ...]
        self.regressors: Tuple[Regressor, ...]
        self.htune: bool
        self.htune_val: ValMethod
        self.htune_val_size: float
        self.htune_trials: int
        self.test_val: ValMethod
        self.test_val_sizes: Tuple[float, ...]
        self.outdir: Path
        self.program_dirs: ProgramDirs
        self.verbosity: Verbosity = cli_args.verbosity

        self.datapath = self.validate_datapath(cli_args.df)
        self.outdir = self.ensure_outdir(self.datapath, cli_args.outdir)
        self.program_dirs = setup_io(self.outdir)
        self.target = cli_args.target
        self.mode = cli_args.mode
        # remove duplicates
        self.classifiers = tuple(sorted(set(cli_args.classifiers)))
        self.regressors = tuple(sorted(set(cli_args.regressors)))
        self.feat_select = tuple(sorted(set(cli_args.feat_select)))
        self.feat_clean = tuple(sorted(set(cli_args.feat_clean)))

        self.cleaning_options = CleaningOptions(
            datapath=self.datapath,
            target=self.target,
            feat_clean=self.feat_clean,
            drop_nan=cli_args.drop_nan,
        )
        self.selection_options = SelectionOptions(
            cleaning_options=self.cleaning_options,
            mode=self.mode,
            classifiers=self.classifiers,
            regressors=self.regressors,
            feat_select=self.feat_select,
            n_feat=cli_args.n_feat,
        )

        self.htune = cli_args.htune
        self.htune_val = cli_args.htune_val
        self.htune_val_size = cli_args.htune_val_size
        self.htune_trials = cli_args.htune_trials
        self.test_val = cli_args.test_val
        self.test_val_sizes = tuple(sorted(set(cli_args.test_val_sizes)))

        # errors
        if self.mode == "regress":
            if ("d" in self.feat_select) or ("auc" in self.feat_select):
                args = " ".join(self.feat_select)
                raise ValueError(
                    "Feature selection with Cohen's d or AUC values not supported "
                    "for regression data. Do not pass arguments `d` or `auc` to "
                    f"`--feat-select` CLI option. [Got arguments: {args}]"
                )

        # warnings
        if self.verbosity is Verbosity.ERROR:
            return  # don't warn user

        if self.htune_trials < 100:
            warn(
                "Without pruning, Optuna generally only shows clear superiority\n"
                "to random search at roughly 50-100 trials. See e.g.\n"
                "    Akiba et al. (2019)\n"
                "    Optuna: A Next-generation Hyperparameter Optimization Framework \n"
                "    https://arxiv.org/pdf/1907.10902.pdf\n"
                "For deep learners, e.g. if using `mlp` as either a classifer\n"
                "or regressor, experience suggests more like 100-200 trials (with\n"
                "pruning) are needed when exploring new architectures. For the\n"
                "current MLP architecture, probably 100 trials is sufficient.\n"
            )

        if ("step-up" in self.feat_select) or ("step-down" in self.feat_select):
            warn(
                "Step-up and step-down feature selection have very high time-complexity.\n"
                "It is strongly recommended to run these selection procedures in isolation,\n"
                "and not in the same process as all other feature selection procedures.\n"
            )
        if "step-down" in self.feat_select:
            warn(
                "Step-down feature selection in particular will usually be intractable\n"
                "even on small (100 features, 1000 samples) datasets and when selecting\n"
                "only 10-20 features, unless using a very fast estimator (linear\n"
                "regression, logistic regression) or sometimes, a support-vector machine.\n"
            )

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


def cv_size(cv_str: str) -> Union[float, int]:
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
    if cv != round(cv):
        raise ArgumentError(
            "`--htune-val-size` must be an integer if greater than 1, as it specified the `k` in k-fold"
        )
    if cv > 10:
        warn("`--htune-val-size` greater than 10 is not recommended.", category=UserWarning)
    if cv > 1:
        return int(cv)
    return cv


def get_options(args: str = None) -> ProgramOptions:
    """parse command line arguments"""
    # parser = ArgumentParser(description=DESC)
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter, epilog=USAGE_EXAMPLES)
    parser.add_argument("--df", action="store", type=resolved_path, required=True, help=DF_HELP_STR)
    # just use existing pathname instead
    # parser.add_argument("--df-name", action="store", type=str, default="", help=DFNAME_HELP_STR)
    parser.add_argument(
        "--target", "-y", action="store", type=str, default="target", help=TARGET_HELP_STR
    )
    parser.add_argument(
        "--mode",
        "-m",
        action="store",
        choices=["classify", "regress"],
        default="classify",
        help=MODE_HELP_STR,
    )
    # NOTE: `nargs="+"` allows repeats, must be removed after
    parser.add_argument(
        "--classifiers",
        "-C",
        nargs="+",
        type=str,
        choices=CLASSIFIERS,
        default=["svm"],
        help=CLS_HELP_STR,
    )
    parser.add_argument(
        "--regressors",
        "-R",
        nargs="+",
        type=str,
        choices=REGRESSORS,
        default=["linear"],
        help=REG_HELP_STR,
    )
    parser.add_argument(
        "--feat-select",
        "-F",
        nargs="+",
        type=str,
        choices=FEATURE_SELECTIONS,
        default=["pca"],
        help=FEAT_SELECT_HELP,
    )
    parser.add_argument(
        "--feat-clean",
        "-f",
        nargs="+",
        type=str,
        choices=FEATURE_CLEANINGS,
        default=["constant"],
        help=FEAT_CLEAN_HELP,
    )
    parser.add_argument(
        "--drop-nan", "-d", choices=["all", "rows", "cols", "none"], default="none", help=NAN_HELP
    )
    parser.add_argument("--n-feat", type=int, default=10, help=N_FEAT_HELP)
    parser.add_argument("--htune", action="store_true", help=HTUNE_HELP)
    parser.add_argument(
        "--htune-val",
        "-H",
        type=str,
        choices=HTUNE_VAL_METHODS,
        default=3,
        help=HTUNEVAL_HELP_STR,
    )
    parser.add_argument("--htune-val-size", type=cv_size, default=0)
    parser.add_argument("--htune-trials", type=int, default=100)
    parser.add_argument("--test-val", "-T", type=str, choices=HTUNE_VAL_METHODS, default="kfold")
    parser.add_argument("--test-val-sizes", nargs="+", type=cv_size, default=5)
    parser.add_argument("--outdir", type=resolved_path, default=None)
    parser.add_argument("--verbosity", type=Verbosity, default=Verbosity(1))
    cli_args = parser.parse_args() if args is None else parser.parse_args(args.split())
    return ProgramOptions(cli_args)
