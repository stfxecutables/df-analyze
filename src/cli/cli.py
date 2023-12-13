"""
File for defining all options passed to `df-analyze.py`.
"""
import os
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    Union,
)
from warnings import warn

from src._constants import (
    CLASSIFIERS,
    DEFAULT_OUTDIR,
    FEATURE_CLEANINGS,
    FEATURE_SELECTIONS,
    HTUNE_VAL_METHODS,
    REGRESSORS,
    SENTINEL,
)
from src._types import (
    Classifier,
    EstimationMode,
    FeatureCleaning,
    FeatureSelection,
    Regressor,
    ValMethod,
)
from src.cli.parsing import cv_size, resolved_path, separator
from src.cli.text import (
    CATEGORICAL_HELP_STR,
    CLS_HELP_STR,
    DF_HELP_STR,
    DROP_HELP_STR,
    EXPLODE_HELP,
    FEAT_CLEAN_HELP,
    FEAT_SELECT_HELP,
    HTUNE_HELP,
    HTUNE_TRIALS_HELP,
    HTUNE_VALSIZE_HELP,
    HTUNEVAL_HELP_STR,
    MC_REPEATS_HELP,
    MODE_HELP_STR,
    N_FEAT_HELP,
    NAN_HELP,
    ORDINAL_HELP_STR,
    OUTDIR_HELP,
    REG_HELP_STR,
    SEP_HELP_STR,
    SHEET_HELP_STR,
    TARGET_HELP_STR,
    TEST_VAL_HELP,
    TEST_VALSIZES_HELP,
    USAGE_EXAMPLES,
    USAGE_STRING,
    VERBOSITY_HELP,
)
from src.enumerables import NanHandling
from src.loading import load_spreadsheet
from src.saving import ProgramDirs, setup_io
from src.utils import Debug

Size = Union[float, int]


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
    categoricals: list[str]
    ordinals: list[str]
    drops: list[str]
    feat_clean: Tuple[FeatureCleaning, ...]
    nan_handling: NanHandling


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
    """Just a container for handling CLI options and default logic (while also
    providing better typing than just using the `Namespace` from the
    `ArgumentParser`).

    Notes
    -----
    For `joblib.Memory` to cache properly, we need all arguments to be
    hashable. This means immutable (among other things) so we use `Tuple` types
    for arguments or options where there are multiple steps to go through, e.g.
    feature selection.
    """

    def __init__(
        self,
        datapath: Path,
        target: str,
        categoricals: list[str],
        ordinals: list[str],
        drops: list[str],
        nan_handling: NanHandling,
        feat_clean: Tuple[FeatureCleaning, ...],
        feat_select: Tuple[FeatureSelection, ...],
        n_feat: int,
        mode: EstimationMode,
        classifiers: Tuple[Classifier, ...],
        regressors: Tuple[Regressor, ...],
        htune: bool,
        htune_val: ValMethod,
        htune_val_size: Size,
        htune_trials: int,
        test_val: ValMethod,
        test_val_sizes: Tuple[Size, ...],
        mc_repeats: int,
        outdir: Path,
        is_spreadsheet: bool,
        separator: str,
        verbosity: Verbosity,
        no_warn_explosion: bool,
    ) -> None:
        # memoization-related
        self.cleaning_options: CleaningOptions
        self.selection_options: SelectionOptions
        # other
        self.datapath: Path = self.validate_datapath(datapath)
        self.target: str = target
        self.categoricals: list[str] = categoricals
        self.ordinals: list[str] = ordinals
        self.drops: list[str] = drops
        self.nan_handling: NanHandling = nan_handling
        self.feat_clean: Tuple[FeatureCleaning, ...] = tuple(sorted(set(feat_clean)))
        self.feat_select: Tuple[FeatureSelection, ...] = tuple(sorted(set(feat_select)))
        self.n_feat: int = n_feat
        self.mode: EstimationMode = mode
        self.classifiers: Tuple[Classifier, ...] = tuple(sorted(set(classifiers)))
        self.regressors: Tuple[Regressor, ...] = tuple(sorted(set(regressors)))
        self.htune: bool = htune
        self.htune_val: ValMethod = htune_val
        self.htune_val_size: Size = htune_val_size
        self.htune_trials: int = htune_trials
        self.test_val: ValMethod = test_val
        self.test_val_sizes: Tuple[Size, ...]
        self.mc_repeats: int = mc_repeats
        self.outdir: Path = self.ensure_outdir(self.datapath, outdir)
        self.program_dirs: ProgramDirs = setup_io(self.outdir)
        self.is_spreadsheet: bool = is_spreadsheet
        self.separator: str = separator
        self.verbosity: Verbosity = verbosity
        self.no_warn_explosion: bool = no_warn_explosion

        if isinstance(test_val_sizes, (int, float)):
            self.test_val_sizes = (test_val_sizes,)
        else:
            self.test_val_sizes = tuple(sorted(set(test_val_sizes)))

        self.cleaning_options = CleaningOptions(
            datapath=self.datapath,
            target=self.target,
            categoricals=self.categoricals,
            ordinals=self.ordinals,
            drops=self.drops,
            feat_clean=self.feat_clean,
            nan_handling=self.nan_handling,
        )
        self.selection_options = SelectionOptions(
            cleaning_options=self.cleaning_options,
            mode=self.mode,
            classifiers=self.classifiers,
            regressors=self.regressors,
            feat_select=self.feat_select,
            n_feat=self.n_feat,
        )

        # errors
        if self.mode == "regress":
            if ("d" in self.feat_select) or ("auc" in self.feat_select):
                args = " ".join(self.feat_select)
                raise ValueError(
                    "Feature selection with Cohen's d or AUC values is not supported "
                    "for regression tasks. Do not pass arguments `d` or `auc` to "
                    f"`--feat-select` CLI option. [Got arguments: {args}]"
                )
        self.spam_warnings()

    def spam_warnings(self) -> None:
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
                "Step-up and step-down feature selection can have very high time-complexity.\n"
                "It is strongly recommended to run these selection procedures in isolation,\n"
                "and not in the same process as all other feature selection procedures.\n"
                "See also the relevant notes on runtime complexity of these techniques:\n"
                "https://scikit-learn.org/stable/modules/feature_selection.html#sequential-feature-selection"
            )
        if "step-down" in self.feat_select:
            warn(
                "Step-down feature selection in particular will usually be intractable\n"
                "even on small (100 features, 1000 samples) datasets and when selecting\n"
                "a much smaller number of features (10-20), unless using a very fast\n"
                "estimator (linear regression, logistic regression, maybe svm)."
            )
        print("To silence these warnings, use `--verbosity=0`.")

    @staticmethod
    def validate_datapath(df_path: Path) -> Path:
        datapath = resolved_path(df_path)
        if not datapath.exists():
            raise FileNotFoundError(f"The specified file {datapath} does not exist.")
        if not datapath.is_file():
            raise FileNotFoundError(f"{datapath} is not a file.")
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


def parse_and_merge_args(parser: ArgumentParser, args: Optional[str] = None) -> Namespace:
    # CLI args supersede when non default and also specified in sheet
    # see https://stackoverflow.com/a/76230387 for a similar problem
    cli_parser = deepcopy(parser)
    sheet_parser = deepcopy(parser)
    sentinel_parser = deepcopy(parser)

    if args is None:
        sentinels = {key: SENTINEL for key in parser.parse_args().__dict__}
    else:
        sentinels = {key: SENTINEL for key in parser.parse_args(args.split()).__dict__}
    sentinel_parser.set_defaults(**sentinels)

    cli_args = cli_parser.parse_args() if args is None else cli_parser.parse_args(args.split())
    if cli_args.spreadsheet is None and cli_args.df is None:
        raise ValueError("Must specify one of either `--spreadsheet [file]` or `--df [file]`.")

    sentinel_cli_args = (
        sentinel_parser.parse_args() if args is None else sentinel_parser.parse_args(args.split())
    )
    explicit_cli_args = {
        key: val for key, val in sentinel_cli_args.__dict__.items() if val is not SENTINEL
    }

    spreadsheet = cli_args.spreadsheet
    if spreadsheet is not None:
        options = load_spreadsheet(spreadsheet)[1]
    else:
        options = ""

    sheet_args = sheet_parser.parse_args(options.split()).__dict__
    # sentinel_sheet_args = sentinel_parser.parse_args(options.split())
    # explicit_sheet_args = Namespace(
    #     **{key: val for key, val in sentinel_sheet_args.__dict__.items() if val is not SENTINEL}
    # )

    cli_args = Namespace(**{**sheet_args, **explicit_cli_args})
    return cli_args


def get_options(args: Optional[str] = None) -> ProgramOptions:
    """parse command line arguments"""
    # parser = ArgumentParser(description=DESC)
    parser = ArgumentParser(
        prog="df-analyze",
        usage=USAGE_STRING,
        formatter_class=RawTextHelpFormatter,
        epilog=USAGE_EXAMPLES,
    )
    parser.add_argument(
        "--spreadsheet",
        type=resolved_path,
        required=False,
        default=None,
        help=SHEET_HELP_STR,
    )
    parser.add_argument(
        "--df",
        action="store",
        type=resolved_path,
        required=False,
        default=None,
        help=DF_HELP_STR,
    )
    parser.add_argument(
        "--separator",
        type=separator,
        required=False,
        default=",",
        help=SEP_HELP_STR,
    )
    parser.add_argument(
        "--target",
        action="store",
        type=str,
        default="target",
        help=TARGET_HELP_STR,
    )
    parser.add_argument(
        "--categoricals",
        nargs="+",
        action="store",
        type=str,
        default=[],
        help=CATEGORICAL_HELP_STR,
    )
    parser.add_argument(
        "--ordinals",
        nargs="+",
        action="store",
        type=str,
        default=[],
        help=ORDINAL_HELP_STR,
    )
    parser.add_argument(
        "--drops",
        nargs="+",
        action="store",
        type=str,
        default=[],
        help=DROP_HELP_STR,
    )
    parser.add_argument(
        "--mode",
        action="store",
        choices=["classify", "regress"],
        default="classify",
        help=MODE_HELP_STR,
    )
    # NOTE: `nargs="+"` allows repeats, must be removed after
    parser.add_argument(
        "--classifiers",
        nargs="+",
        type=str,
        choices=CLASSIFIERS,
        default=["svm"],
        help=CLS_HELP_STR,
    )
    parser.add_argument(
        "--regressors",
        nargs="+",
        type=str,
        choices=REGRESSORS,
        default=["linear"],
        help=REG_HELP_STR,
    )
    parser.add_argument(
        "--feat-select",
        nargs="+",
        type=str,
        choices=FEATURE_SELECTIONS,
        default=["pca"],
        help=FEAT_SELECT_HELP,
    )
    parser.add_argument(
        "--feat-clean",
        nargs="+",
        type=str,
        choices=FEATURE_CLEANINGS,
        default=["constant"],
        help=FEAT_CLEAN_HELP,
    )
    parser.add_argument(
        "--nan",
        choices=[*NanHandling],
        default=NanHandling.Mean,
        type=NanHandling,
        help=NAN_HELP,
    )
    parser.add_argument(
        "--n-feat",
        type=int,
        default=10,
        help=N_FEAT_HELP,
    )
    parser.add_argument(
        "--htune",
        action="store_true",
        help=HTUNE_HELP,
    )
    parser.add_argument(
        "--htune-val",
        type=str,
        choices=HTUNE_VAL_METHODS,
        default=3,
        help=HTUNEVAL_HELP_STR,
    )
    parser.add_argument(
        "--htune-val-size",
        type=cv_size,
        default=3,
        help=HTUNE_VALSIZE_HELP,
    )
    parser.add_argument(
        "--htune-trials",
        type=int,
        default=100,
        help=HTUNE_TRIALS_HELP,
    )
    parser.add_argument(
        "--mc-repeats",
        type=int,
        default=10,
        help=MC_REPEATS_HELP,
    )
    parser.add_argument(
        "--test-val",
        type=str,
        choices=HTUNE_VAL_METHODS,
        default="kfold",
        help=TEST_VAL_HELP,
    )
    parser.add_argument(
        "--test-val-sizes",
        nargs="+",
        type=cv_size,
        default=5,
        help=TEST_VALSIZES_HELP,
    )
    parser.add_argument(
        "--outdir",
        type=resolved_path,
        required=False,
        default=DEFAULT_OUTDIR,
        help=OUTDIR_HELP,
    )
    parser.add_argument(
        "--verbosity",
        type=lambda a: Verbosity(int(a)),
        default=Verbosity(1),
        help=VERBOSITY_HELP,
    )
    parser.add_argument(
        "--no-warn-explosion",
        action="store_true",
        help=EXPLODE_HELP,
    )

    cli_args = parse_and_merge_args(parser, args)
    cats = set(cli_args.categoricals)
    ords = set(cli_args.ordinals)
    for cat in cats:
        if cat in ords:
            warn(
                f"Feature '{cat}' is present in both `--categoricals` and "
                f"`--ordinals` options. Feature '{cat}' will be treated as "
                "ordinal to reduce compute times"
            )
            ords.remove(cat)

    return ProgramOptions(
        datapath=cli_args.spreadsheet if cli_args.df is None else cli_args.df,
        target=cli_args.target,
        categoricals=sorted(cats),
        ordinals=sorted(ords),
        drops=cli_args.drops,
        nan_handling=cli_args.nan,
        feat_clean=cli_args.feat_clean,
        feat_select=cli_args.feat_select,
        n_feat=cli_args.n_feat,
        mode=cli_args.mode,
        classifiers=cli_args.classifiers,
        regressors=cli_args.regressors,
        htune=cli_args.htune,
        htune_val=cli_args.htune_val,
        htune_val_size=cli_args.htune_val_size,
        htune_trials=cli_args.htune_trials,
        test_val=cli_args.test_val,
        test_val_sizes=cli_args.test_val_sizes,
        mc_repeats=cli_args.mc_repeats,
        outdir=cli_args.outdir,
        is_spreadsheet=cli_args.spreadsheet is not None,
        separator=cli_args.separator,
        verbosity=cli_args.verbosity,
        no_warn_explosion=cli_args.no_warn_explosion,
    )
