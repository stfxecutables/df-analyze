from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


"""
File for defining all options passed to `df-analyze.py`.
"""
import os
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from copy import deepcopy
from enum import Enum
from pathlib import Path
from random import choice, randint, uniform
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
from warnings import warn

import jsonpickle
import numpy as np
import pandas as pd
from pandas import DataFrame

from src._constants import (
    FULL_RESULTS,
    N_WRAPPER_DEFAULT,
    P_FILTER_CAT_DEFAULT,
    P_FILTER_CONT_DEFAULT,
    P_FILTER_TOTAL_DEFAULT,
    SENTINEL,
)
from src.analysis.univariate.associate import (
    CatClsStats,
    CatRegStats,
    ContClsStats,
    ContRegStats,
)
from src.cli.parsing import (
    column_parser,
    int_or_percent_parser,
    resolved_path,
    separator,
)
from src.cli.text import (
    ASSOC_SELECT_CAT_CLS_STATS,
    ASSOC_SELECT_CAT_REG_STATS,
    ASSOC_SELECT_CONT_CLS_STATS,
    ASSOC_SELECT_CONT_REG_STATS,
    CATEGORICAL_HELP_STR,
    CLS_HELP_STR,
    CLS_TUNE_METRIC,
    DF_HELP_STR,
    DROP_HELP_STR,
    EMBED_SELECT_MODEL_HELP,
    EXPLODE_HELP,
    FEAT_SELECT_HELP,
    FILTER_METHOD_HELP,
    HTUNE_TRIALS_HELP,
    MODE_HELP_STR,
    N_FEAT_CAT_FILTER_HELP,
    N_FEAT_CONT_FILTER_HELP,
    N_FEAT_TOTAL_FILTER_HELP,
    N_FEAT_WRAPPER_HELP,
    NAN_HELP,
    NORM_HELP,
    ORDINAL_HELP_STR,
    OUTDIR_HELP,
    PRED_SELECT_CLS_SCORE,
    PRED_SELECT_REG_SCORE,
    REG_HELP_STR,
    REG_TUNE_METRIC,
    SEP_HELP_STR,
    SHEET_HELP_STR,
    TARGET_HELP_STR,
    TEST_VALSIZES_HELP,
    USAGE_EXAMPLES,
    USAGE_STRING,
    VERBOSITY_HELP,
    WRAP_SELECT_HELP,
    WRAP_SELECT_MODEL_HELP,
)
from src.enumerables import (
    ClassifierScorer,
    ClsScore,
    DfAnalyzeClassifier,
    DfAnalyzeRegressor,
    EmbedSelectionModel,
    FeatureSelection,
    FilterSelection,
    NanHandling,
    Normalization,
    RegressorScorer,
    RegScore,
    WrapperSelection,
    WrapperSelectionModel,
)
from src.loading import load_spreadsheet

if TYPE_CHECKING:
    from src.models.base import DfAnalyzeModel
    from src.testing.datasets import TestDataset
from src.saving import ProgramDirs, get_hash
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
        norm: Normalization,
        # feat_clean: Tuple[FeatureCleaning, ...],
        feat_select: Tuple[FeatureSelection, ...],
        embed_select: Optional[tuple[EmbedSelectionModel, ...]],
        wrapper_select: Optional[WrapperSelection],
        wrapper_model: WrapperSelectionModel,
        # n_feat: int,
        n_filter_cont: Union[int, float],
        n_filter_cat: Union[int, float],
        n_feat_filter: Union[int, float],
        n_feat_wrapper: Union[int, float, None],
        filter_assoc_cont_cls: ContClsStats,
        filter_assoc_cat_cls: CatClsStats,
        filter_assoc_cont_reg: ContRegStats,
        filter_assoc_cat_reg: CatRegStats,
        filter_pred_cls_score: ClsScore,
        filter_pred_reg_score: RegScore,
        is_classification: bool,
        classifiers: Tuple[DfAnalyzeClassifier, ...],
        regressors: Tuple[DfAnalyzeRegressor, ...],
        # htune: bool,
        # htune_val: ValMethod,
        # htune_val_size: Size,
        htune_trials: int,
        htune_cls_metric: ClassifierScorer,
        htune_reg_metric: RegressorScorer,
        # test_val: ValMethod,
        test_val_size: Size,
        # mc_repeats: int,
        outdir: Path,
        is_spreadsheet: bool,
        separator: str,
        verbosity: Verbosity,
        no_warn_explosion: bool,
    ) -> None:
        # memoization-related
        # other
        self.datapath: Path = self.validate_datapath(datapath)
        self.target: str = target
        self.categoricals: list[str] = categoricals
        self.ordinals: list[str] = ordinals
        self.drops: list[str] = drops
        self.nan_handling: NanHandling = nan_handling
        self.norm: Normalization = norm
        # self.feat_clean: Tuple[FeatureCleaning, ...] = tuple(sorted(set(feat_clean)))
        self.feat_select: Tuple[FeatureSelection, ...] = tuple(sorted(set(feat_select)))
        self.embed_select: Optional[tuple[EmbedSelectionModel, ...]] = embed_select
        self.wrapper_select: Optional[WrapperSelection] = wrapper_select
        self.wrapper_model: WrapperSelectionModel = wrapper_model
        # self.n_feat: int = n_feat
        self.n_filter_cont: Union[int, float] = n_filter_cont
        self.n_filter_cat: Union[int, float] = n_filter_cat
        self.n_feat_filter: Union[int, float] = n_feat_filter
        self.n_feat_wrapper: Union[int, float, None] = n_feat_wrapper
        self.filter_assoc_cont_cls: ContClsStats = filter_assoc_cont_cls
        self.filter_assoc_cat_cls: CatClsStats = filter_assoc_cat_cls
        self.filter_assoc_cont_reg: ContRegStats = filter_assoc_cont_reg
        self.filter_assoc_cat_reg: CatRegStats = filter_assoc_cat_reg
        self.filter_pred_cls_score: ClsScore = filter_pred_cls_score
        self.filter_pred_reg_score: RegScore = filter_pred_reg_score
        self.is_classification: bool = is_classification
        self.classifiers: Tuple[DfAnalyzeClassifier, ...] = tuple(
            sorted(set(classifiers))
        )
        self.regressors: Tuple[DfAnalyzeRegressor, ...] = tuple(sorted(set(regressors)))
        # self.htune: bool = htune
        # self.htune_val: ValMethod = htune_val
        # self.htune_val_size: Size = htune_val_size
        self.htune_trials: int = htune_trials
        self.htune_cls_metric: ClassifierScorer = htune_cls_metric
        self.htune_reg_metric: RegressorScorer = htune_reg_metric
        # self.test_val: ValMethod = test_val
        self.test_val_size: Size = test_val_size
        # self.mc_repeats: int = mc_repeats
        # TODO: fix below
        self.outdir: Optional[Path] = self.get_outdir(outdir, self.datapath)
        self.is_spreadsheet: bool = is_spreadsheet
        self.separator: str = separator
        self.verbosity: Verbosity = verbosity
        self.no_warn_explosion: bool = no_warn_explosion
        self.program_dirs: ProgramDirs = ProgramDirs.new(self.outdir, self.hash())

        # cleanup
        if DfAnalyzeClassifier.Dummy not in self.classifiers:
            self.classifiers = (DfAnalyzeClassifier.Dummy, *self.classifiers)
        if DfAnalyzeRegressor.Dummy not in self.regressors:
            self.regressors = (DfAnalyzeRegressor.Dummy, *self.regressors)

        is_cls = self.is_classification
        self.comparables = {
            "model": self.classifiers if is_cls else self.regressors,
            "selection": self.feat_select,
            "norm": self.norm,
        }

        self.spam_warnings()

    @property
    def models(self) -> list[Type[DfAnalyzeModel]]:
        sources = self.classifiers if self.is_classification else self.regressors
        clses = [source.get_model() for source in sources]
        return clses

    @staticmethod
    def random(ds: TestDataset, outdir: Optional[Path] = None) -> ProgramOptions:
        n_samples, n_feats = ds.shape
        # feat_clean = FeatureCleaning.random_n()
        feat_select = FeatureSelection.random_n()
        wrap_select = WrapperSelection.random()
        wrap_model = WrapperSelectionModel.random()
        embed_select = (EmbedSelectionModel.random_none(),)
        # n_feat = n_feats
        n_feat_filter = uniform(0.5, 0.95)
        n_feat_wrapper = choice([uniform(0.2, 0.5), None])
        n_filter_cont = uniform(0.1, 0.25)
        n_filter_cat = n_feat_filter - n_filter_cont
        filter_assoc_cont_cls = ContClsStats.random()
        filter_assoc_cat_cls = CatClsStats.random()
        filter_assoc_cont_reg = ContRegStats.random()
        filter_assoc_cat_reg = CatRegStats.random()
        filter_pred_cls_score = ClsScore.random()
        filter_pred_reg_score = RegScore.random()
        is_classification = ds.is_classification
        classifiers = DfAnalyzeClassifier.random_n()
        regressors = DfAnalyzeRegressor.random_n()
        # htune: bool = choice([True, False])
        # htune_val: ValMethod = "kfold"
        # htune_val_size: Size = 5
        htune_trials: int = randint(20, 100)
        htune_cls_metric = ClassifierScorer.random()
        htune_reg_metric = RegressorScorer.random()
        # test_val: ValMethod = "kfold"
        test_val_size: Size = 0.4
        # mc_repeats: int = 0
        sub = "classification" if ds.is_classification else "regression"
        outdir = outdir or ((FULL_RESULTS / sub) / ds.dsname)
        is_spreadsheet: bool = False
        separator: str = ","
        verbosity: Verbosity = Verbosity.ERROR
        no_warn_explosion: bool = False
        return ProgramOptions(
            datapath=ds.datapath,
            target="target",
            categoricals=ds.categoricals,
            ordinals=[],
            drops=[],
            nan_handling=NanHandling.random(),
            norm=Normalization.random(),
            # feat_clean=feat_clean,
            feat_select=feat_select,
            embed_select=embed_select,  # type: ignore
            wrapper_select=wrap_select,
            wrapper_model=wrap_model,
            # n_feat=n_feat,
            n_filter_cat=n_filter_cat,
            n_filter_cont=n_filter_cont,
            n_feat_filter=n_feat_filter,
            n_feat_wrapper=n_feat_wrapper,
            filter_assoc_cont_cls=filter_assoc_cont_cls,
            filter_assoc_cat_cls=filter_assoc_cat_cls,
            filter_assoc_cont_reg=filter_assoc_cont_reg,
            filter_assoc_cat_reg=filter_assoc_cat_reg,
            filter_pred_cls_score=filter_pred_cls_score,
            filter_pred_reg_score=filter_pred_reg_score,
            is_classification=is_classification,
            classifiers=classifiers,
            regressors=regressors,
            # htune=htune,
            # htune_val=htune_val,
            # htune_val_size=htune_val_size,
            htune_trials=htune_trials,
            htune_cls_metric=htune_cls_metric,
            htune_reg_metric=htune_reg_metric,
            # test_val=test_val,
            test_val_size=test_val_size,
            outdir=outdir,
            is_spreadsheet=is_spreadsheet,
            separator=separator,
            verbosity=verbosity,
            no_warn_explosion=no_warn_explosion,
        )

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
    def get_outdir(outdir: Optional[Path], datapath: Path) -> Optional[Path]:
        if outdir is None:
            return None
        name = datapath.stem
        outdir = outdir / name
        os.makedirs(outdir, exist_ok=True)
        return outdir

    def hash(self) -> str:
        return get_hash(
            self.__dict__,
            ignores=[
                "cleaning_options",
                "selection_options",
                "comparables",
                "program_dirs",
            ],
        )

    def to_json(self) -> None:
        path = self.program_dirs.options
        if path is None:
            return
        path.write_text(str(jsonpickle.encode(self)))

    @staticmethod
    def from_json(root: Path) -> ProgramOptions:
        options = root / "options.json"
        obj = jsonpickle.decode(options.read_text())
        return cast(ProgramOptions, obj)

    @staticmethod
    def from_jsonfile(file: Path) -> ProgramOptions:
        obj = jsonpickle.decode(file.read_text())
        return cast(ProgramOptions, obj)

    def load_df(self) -> DataFrame:
        if self.is_spreadsheet:
            return load_spreadsheet(self.datapath, self.separator)[0]
        path = self.datapath
        if path.name.endswith("parquet"):
            return pd.read_parquet(path)
        if path.name.endswith("csv"):
            return pd.read_csv(path, sep=self.separator)
        if path.name.endswith("json"):
            return pd.read_json(path)
        raise ValueError(f"Unrecognized filetype: '{path.suffix}'")


def parse_and_merge_args(parser: ArgumentParser, args: Optional[str] = None) -> Namespace:
    # CLI args supersede when non default and also specified in sheet
    # see https://stackoverflow.com/a/76230387 for a similar problem
    cli_parser = deepcopy(parser)
    sheet_parser = deepcopy(parser)
    sentinel_parser = deepcopy(parser)

    if args is None:
        sentinels = {key: SENTINEL for key in parser.parse_known_args()[0].__dict__}
    else:
        sentinels = {
            key: SENTINEL for key in parser.parse_known_args(args.split())[0].__dict__
        }
    sentinel_parser.set_defaults(**sentinels)

    cli_args = (
        cli_parser.parse_known_args()[0]
        if args is None
        else cli_parser.parse_known_args(args.split())[0]
    )
    if cli_args.spreadsheet is None and cli_args.df is None:
        raise ValueError(
            "Must specify one of either `--spreadsheet [file]` or `--df [file]`."
        )

    sentinel_cli_args = (
        sentinel_parser.parse_known_args()[0]
        if args is None
        else sentinel_parser.parse_known_args(args.split())[0]
    )
    explicit_cli_args = {
        key: val for key, val in sentinel_cli_args.__dict__.items() if val is not SENTINEL
    }

    spreadsheet = cli_args.spreadsheet
    if spreadsheet is not None:
        options = load_spreadsheet(spreadsheet)[1]
    else:
        options = ""

    sheet_args = sheet_parser.parse_known_args(options.split())[0].__dict__
    # sentinel_sheet_args = sentinel_parser.parse_args(options.split())
    # explicit_sheet_args = Namespace(
    #     **{key: val for key, val in sentinel_sheet_args.__dict__.items() if val is not SENTINEL}
    # )

    cli_args = Namespace(**{**sheet_args, **explicit_cli_args})
    return cli_args


def make_parser() -> ArgumentParser:
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
        nargs="+",  # allow spaces: https://stackoverflow.com/a/26990349,
        type=str,
        default="target",
        help=TARGET_HELP_STR,
    )
    parser.add_argument(
        "--categoricals",
        # nargs="+",
        action="store",
        type=column_parser,
        default=[],
        help=CATEGORICAL_HELP_STR,
    )
    parser.add_argument(
        "--ordinals",
        # nargs="+",
        action="store",
        type=column_parser,
        default=[],
        help=ORDINAL_HELP_STR,
    )
    parser.add_argument(
        "--drops",
        # nargs="+",
        action="store",
        type=column_parser,
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
        # type=DfAnalyzeClassifier.parse,
        type=str,
        choices=DfAnalyzeClassifier.choices(),
        default=DfAnalyzeClassifier.defaults(),
        metavar="",
        help=CLS_HELP_STR,
    )
    parser.add_argument(
        "--regressors",
        nargs="+",
        type=str,
        choices=DfAnalyzeRegressor.choices(),
        default=DfAnalyzeRegressor.defaults(),
        help=REG_HELP_STR,
    )
    parser.add_argument(
        "--feat-select",
        nargs="+",
        type=FeatureSelection.parseN,  # applied to each spaced element in arg
        choices=FeatureSelection.choicesN(),
        default=(FeatureSelection.Filter,),
        metavar="",  # silences ugly options spam related to enums
        help=FEAT_SELECT_HELP,
    )
    # parser.add_argument(
    #     "--model-select",
    #     nargs="+",
    #     type=ModelFeatureSelection.parseN,  # applied to each spaced element in arg
    #     choices=ModelFeatureSelection.choicesN(),
    #     default=(ModelFeatureSelection.Embedded,),
    #     metavar="",  # silences ugly options spam related to enums
    #     help=MODEL_SELECT_HELP,
    # )
    parser.add_argument(
        "--embed-select",
        nargs="+",
        # type=enum_or_none_parser(EmbedSelectionModel),
        type=EmbedSelectionModel.parseN,
        # choices=[m.value for m in EmbedSelectionModel]
        # + ["none"]
        # + [EmbedSelectionModel.LGBM, EmbedSelectionModel.Linear, None],
        choices=EmbedSelectionModel.choicesN(),
        default=(EmbedSelectionModel.Linear,),
        help=EMBED_SELECT_MODEL_HELP,
        metavar="",
    )
    parser.add_argument(
        "--wrapper-select",
        type=WrapperSelection.parseN,
        choices=WrapperSelection.choicesN(),
        default=None,
        metavar="",
        help=WRAP_SELECT_HELP,
    )
    parser.add_argument(
        "--wrapper-model",
        type=WrapperSelectionModel.parseN,
        choices=WrapperSelectionModel.choicesN(),
        default=WrapperSelectionModel.Linear,
        metavar="",
        help=WRAP_SELECT_MODEL_HELP,
    )
    # parser.add_argument(  # TODO UNIMPLEMENTED!
    #     "--n-selection-tune-rounds",
    #     type=int,
    #     default=0,
    #     help=SELECT_TUNE_ROUNDS_HELP,
    # )
    # parser.add_argument(
    #     "--feat-clean",
    #     nargs="+",
    #     type=str,
    #     choices=FEATURE_CLEANINGS,
    #     default=["constant"],
    #     help=FEAT_CLEAN_HELP,
    # )
    parser.add_argument(
        "--norm",
        type=Normalization.parse,
        choices=Normalization.choices(),
        default=Normalization.Robust.value,
        help=NORM_HELP,
    )
    parser.add_argument(
        "--nan",
        type=NanHandling.parse,
        choices=NanHandling.choices(),
        default=NanHandling.Mean.value,
        help=NAN_HELP,
    )
    # parser.add_argument(
    #     "--n-feat-embed",
    #     type=int_or_percent_or_none_parser(default=None),
    #     default=None,
    #     help=N_FEAT_EMBED_HELP,
    # )
    parser.add_argument(
        "--n-feat-filter",
        type=int_or_percent_parser(default=P_FILTER_TOTAL_DEFAULT),
        default=P_FILTER_TOTAL_DEFAULT,
        help=N_FEAT_TOTAL_FILTER_HELP,
    )
    parser.add_argument(
        "--n-feat-wrapper",
        type=int_or_percent_parser(default=N_WRAPPER_DEFAULT),
        default=N_WRAPPER_DEFAULT,
        help=N_FEAT_WRAPPER_HELP,
    )
    parser.add_argument(
        "--n-filter-cont",
        type=int_or_percent_parser(default=P_FILTER_CONT_DEFAULT),
        default=P_FILTER_CONT_DEFAULT,
        help=N_FEAT_CONT_FILTER_HELP,
    )
    parser.add_argument(
        "--n-filter-cat",
        type=int_or_percent_parser(default=P_FILTER_CAT_DEFAULT),
        default=P_FILTER_CAT_DEFAULT,
        help=N_FEAT_CAT_FILTER_HELP,
    )
    parser.add_argument(
        "--filter-method",
        type=FilterSelection.parse,
        choices=FilterSelection.choices(),
        default=FilterSelection.Association,
        metavar="",
        help=FILTER_METHOD_HELP,
    )
    parser.add_argument(
        "--filter-assoc-cont-classify",
        choices=ContClsStats.choices(),
        type=ContClsStats.parse,
        default=ContClsStats.default(),
        metavar="",
        help=ASSOC_SELECT_CONT_CLS_STATS,
    )
    parser.add_argument(
        "--filter-assoc-cat-classify",
        choices=CatClsStats.choices(),
        type=CatClsStats.parse,
        default=CatClsStats.default(),
        metavar="",
        help=ASSOC_SELECT_CAT_CLS_STATS,
    )
    parser.add_argument(
        "--filter-assoc-cont-regress",
        choices=ContRegStats.choices(),
        type=ContRegStats.parse,
        default=ContRegStats.default(),
        metavar="",
        help=ASSOC_SELECT_CONT_REG_STATS,
    )
    parser.add_argument(
        "--filter-assoc-cat-regress",
        choices=CatRegStats.choices(),
        type=CatRegStats.parse,
        default=CatRegStats.default(),
        metavar="",
        help=ASSOC_SELECT_CAT_REG_STATS,
    )
    parser.add_argument(
        "--filter-pred-regress",
        choices=RegScore.choices(),
        type=RegScore.parse,
        default=RegScore.default(),
        help=PRED_SELECT_REG_SCORE,
    )
    parser.add_argument(
        "--filter-pred-classify",
        choices=ClsScore.choices(),
        type=ClsScore.parse,
        default=ClsScore.default(),
        help=PRED_SELECT_CLS_SCORE,
    )
    # parser.add_argument(
    #     "--htune",
    #     action="store_true",
    #     help=HTUNE_HELP,
    # )
    # parser.add_argument(
    #     "--htune-val",
    #     type=str,
    #     choices=HTUNE_VAL_METHODS,
    #     default=3,
    #     help=HTUNEVAL_HELP_STR,
    # )
    # parser.add_argument(
    #     "--htune-val-size",
    #     type=cv_size,
    #     default=3,
    #     help=HTUNE_VALSIZE_HELP,
    # )
    parser.add_argument(
        "--htune-trials",
        type=int,
        default=100,
        help=HTUNE_TRIALS_HELP,
    )
    parser.add_argument(
        "--htune-cls-metric",
        choices=ClassifierScorer.choices(),
        type=ClassifierScorer.parse,
        default=ClassifierScorer.default(),
        help=CLS_TUNE_METRIC,
    )
    parser.add_argument(
        "--htune-reg-metric",
        choices=RegressorScorer.choices(),
        type=RegressorScorer.parse,
        default=RegressorScorer.default(),
        help=REG_TUNE_METRIC,
    )
    # parser.add_argument(
    #     "--mc-repeats",
    #     type=int,
    #     default=10,
    #     help=MC_REPEATS_HELP,
    # )
    # parser.add_argument(
    #     "--test-val",
    #     type=str,
    #     choices=HTUNE_VAL_METHODS,
    #     default="kfold",
    #     help=TEST_VAL_HELP,
    # )
    parser.add_argument(
        "--test-val-size",
        type=int_or_percent_parser(default=0.4),
        default=0.4,
        help=TEST_VALSIZES_HELP,
    )
    parser.add_argument(
        "--outdir",
        type=resolved_path,
        required=False,
        default=None,
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
    return parser


class ArgKind(Enum):
    Choice = "choice"
    ChoiceN = "choice-n"
    Flag = "flag"
    IntOrPercent = "int-percent"
    Int = "int"
    Path = "path"
    Separator = "seperator"
    String = "string"
    StringList = "string-list"


class RandKind(Enum):
    ChooseN = "choose-n"
    ChooseNoneOrN = "choose-none-or-n"
    ChooseOne = "one"
    Columns = "columns"
    Custom = "custom"
    Flag = "flag"
    Int = "int"
    IntOrPercent = "int-or-percent"
    Path = "path"
    Target = "target"


ArgsDict = dict[str, tuple[RandKind, Optional[list[str]]]]


def get_parser_dict() -> ArgsDict:
    args_dict: ArgsDict = {
        "--df": (RandKind.Path, None),
        "--spreadsheet": (RandKind.Path, None),
        "--separator": (RandKind.Custom, None),
        "--target": (RandKind.Target, None),
        "--categoricals": (RandKind.Columns, None),
        "--ordinals": (RandKind.Columns, None),
        "--drops": (RandKind.Columns, None),
        "--mode": (RandKind.ChooseOne, ["classify", "regress"]),
        "--classifiers": (RandKind.ChooseN, DfAnalyzeClassifier.choices()),
        "--regressors": (RandKind.ChooseN, DfAnalyzeRegressor.choices()),
        "--feat-select": (RandKind.ChooseNoneOrN, FeatureSelection.choices()),
        "--embed-select": (RandKind.ChooseNoneOrN, EmbedSelectionModel.choices()),
        "--wrapper-select": (RandKind.ChooseNoneOrN, WrapperSelection.choices()),
        "--wrapper-model": (RandKind.ChooseOne, WrapperSelectionModel.choices()),
        "--norm": (RandKind.ChooseOne, Normalization.choices()),
        "--nan": (RandKind.ChooseOne, NanHandling.choices()),
        "--n-feat-filter": (RandKind.IntOrPercent, None),
        "--n-feat-wrapper": (RandKind.IntOrPercent, None),
        "--n-filter-cont": (RandKind.IntOrPercent, None),
        "--n-filter-cat": (RandKind.IntOrPercent, None),
        "--filter-method": (RandKind.ChooseOne, FilterSelection.choices()),
        "--filter-assoc-cont-classify": (RandKind.ChooseOne, ContClsStats.choices()),
        "--filter-assoc-cat-classify": (RandKind.ChooseOne, CatClsStats.choices()),
        "--filter-assoc-cont-regress": (RandKind.ChooseOne, ContRegStats.choices()),
        "--filter-assoc-cat-regress": (RandKind.ChooseOne, CatRegStats.choices()),
        "--filter-pred-regress": (RandKind.ChooseOne, RegScore.choices()),
        "--filter-pred-classify": (RandKind.ChooseOne, ClsScore.choices()),
        "--htune-trials": (RandKind.ChooseOne, ["5", "10", "20"]),
        "--htune-cls-metric": (RandKind.ChooseOne, ClassifierScorer.choices()),
        "--htune-reg-metric": (RandKind.ChooseOne, RegressorScorer.choices()),
        "--test-val-size": (RandKind.IntOrPercent, None),
        "--outdir": (RandKind.Path, None),
        "--verbosity": (RandKind.ChooseOne, ["0", "1", "2"]),
        "--no-warn-explosion": (RandKind.Flag, None),
    }
    return args_dict


def randip(
    imin: int, imax: int, pmin: float = 0.1, pmax: float = 0.5
) -> Union[float, int]:
    use_ints = bool(np.random.binomial(n=1, p=0.5, size=1).item())
    if use_ints:
        return np.random.randint(imin, imax)
    return np.random.uniform(pmin, pmax)


def random_cli_args(
    ds: TestDataset, tempdir: Path, spreadsheet: bool = True
) -> tuple[list[str], Path]:
    args_dict = get_parser_dict()
    n, p = ds.shape[0], ds.shape[1] - 1
    p_max = min(15, p)
    p_min = 1
    pp_max = p_max / p
    pp_min = 1 / p

    n_max = n // 2
    n_min = min(100, n)
    tmin = 0.1
    tmax = 0.5

    int_or_percents = {
        "--n-feat-filter": randip(p_min, p_max, pp_min, pp_max),
        "--n-feat-wrapper": randip(p_min, p_max, pp_min, pp_max),
        "--n-filter-cont": randip(p_min, p_max, pp_min, pp_max),
        "--n-filter-cat": randip(p_min, p_max, pp_min, pp_max),
        "--test-val-size": randip(n_min, n_max, tmin, tmax),
    }

    cols = set(ds.load().columns.to_list())
    target = choice(list(cols))
    cols.remove(target)
    # if " " in target:
    #     target = f"'{target}'"

    n_drop = np.random.randint(0, 3)
    drops = np.random.choice(list(cols), n_drop).tolist()
    cols.difference_update(drops)
    # drops = " ".join([f"'{d}'" if " " in d else d for d in drops])
    # drops = f"'{' '.join(drops)}'"
    drops = " ".join([f'"{x}"' if " " in x else x for x in drops])

    n_cats = np.random.randint(len(cols) // 2)
    cats = np.random.choice(list(cols), n_cats).tolist()
    cols.difference_update(cats)
    # cats = " ".join([f"'{c}'" if " " in c else c for c in cats])
    # cats = f"'{' '.join(cats)}'"
    cats = " ".join([f'"{x}"' if " " in x else x for x in cats])

    n_ords = np.random.randint(len(cols) // 2)
    ords = np.random.choice(list(cols), n_ords).tolist()
    cols.difference_update(ords)
    # ords = " ".join([f"'{o}'" if " " in o else o for o in ords])
    # ords = f"'{' '.join(ords)}'"
    ords = " ".join([f'"{x}"' if " " in x else x for x in ords])

    datafile = tempdir / f"{ds.dsname}.csv" if spreadsheet else ds.datapath

    arg_options = []
    for argstr, (kind, choices) in args_dict.items():
        # handle ds-related randoms first
        if kind is RandKind.Path:
            if argstr == "--df":
                if not spreadsheet:
                    arg_options.append(f"{argstr} {datafile}")
            elif argstr == "--spreadsheet":
                if spreadsheet:
                    arg_options.append(f"{argstr} {datafile}")
            elif argstr == "--outdir":
                arg_options.append(f"{argstr} {tempdir}")
            else:
                raise KeyError(
                    f"Unhandled argument: {argstr}, kind={kind}, choices={choices}"
                )
        elif (kind is RandKind.Custom) and (argstr == "--separator"):
            continue
        elif kind is RandKind.IntOrPercent:
            numeric = round(int_or_percents[argstr], 2)
            arg_options.append(f"{argstr} {numeric}")
        elif kind is RandKind.Target:
            arg_options.append(f"{argstr} {target}")
        elif kind is RandKind.Columns:
            if argstr == "--categoricals":
                if n_cats > 0:
                    arg_options.append(f"{argstr} {cats}")
            elif argstr == "--ordinals":
                if n_ords > 0:
                    arg_options.append(f"{argstr} {ords}")
            elif argstr == "--drops":
                if n_drop > 0:
                    arg_options.append(f"{argstr} {drops}")
            else:
                raise KeyError(
                    f"Unhandled argument: {argstr}, kind={kind}, choices={choices}"
                )
        elif kind is RandKind.Flag:
            do_flag = bool(np.random.binomial(n=1, p=0.5, size=1).item())
            if do_flag:
                arg_options.append(f"{argstr}")
        elif kind in [RandKind.ChooseN, RandKind.ChooseNoneOrN, RandKind.ChooseOne]:
            if choices is None:
                raise ValueError("RandKind cannot be a choice with `choices=None`")
            if kind is RandKind.ChooseOne:
                chosen = choice(choices)
                arg_options.append(f"{argstr} {chosen}")
            elif kind is RandKind.ChooseN:
                n_choose = np.random.randint(1, len(choices))
                chosen = " ".join(np.random.choice(choices, n_choose).tolist())
            elif kind is RandKind.ChooseNoneOrN:
                is_none = bool(np.random.binomial(n=1, p=0.5, size=1).item())
                if is_none:
                    arg_options.append(f"{argstr} none")
                else:
                    n_choose = np.random.randint(1, len(choices))
                    chosen = " ".join(np.random.choice(choices, n_choose).tolist())
                    arg_options.append(f"{argstr} {chosen}")
        else:
            raise KeyError(
                f"Unhandled argument: {argstr}, kind={kind}, choices={choices}"
            )

    return arg_options, datafile


def get_options(args: Optional[str] = None) -> ProgramOptions:
    """parse command line arguments"""
    # parser = ArgumentParser(description=DESC)
    parser = make_parser()
    cli_args = parse_and_merge_args(parser, args)
    mode = str(cli_args.mode).lower()
    is_cls = True if "class" in mode else False
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

    classifiers = set(DfAnalyzeClassifier.from_args(cli_args.classifiers))
    regressors = set(DfAnalyzeRegressor.from_args(cli_args.regressors))
    if DfAnalyzeClassifier.SVM in classifiers:
        warn(
            "Found `svm` classifier as a specified model. SVMs are at the "
            "moment too computationally-expensive on most datasets. This "
            "model has been disabled and performance metrics for this model "
            "will not appear in final results."
        )

        classifiers.discard(DfAnalyzeClassifier.SVM)
    if DfAnalyzeRegressor.SVM in regressors:
        warn(
            "Found `svm` regressor as a specified model. SVMs are at the "
            "moment too computationally-expensive on most datasets. This "
            "model has been disabled and performance metrics for this model "
            "will not appear in final results."
        )

        regressors.discard(DfAnalyzeRegressor.SVM)

    if DfAnalyzeClassifier.Dummy not in classifiers:
        classifiers.add(DfAnalyzeClassifier.Dummy)
    if DfAnalyzeRegressor.Dummy not in regressors:
        regressors.add(DfAnalyzeRegressor.Dummy)

    classifiers = tuple(sorted(classifiers))
    regressors = tuple(sorted(regressors))

    return ProgramOptions(
        datapath=cli_args.spreadsheet if cli_args.df is None else cli_args.df,
        target=" ".join(cli_args.target),  # https://stackoverflow.com/a/26990349,
        categoricals=sorted(cats),
        ordinals=sorted(ords),
        drops=cli_args.drops,
        norm=Normalization.from_arg(cli_args.norm),
        nan_handling=NanHandling.from_arg(cli_args.nan),
        feat_select=FeatureSelection.from_args(cli_args.feat_select),
        embed_select=EmbedSelectionModel.from_args(cli_args.embed_select),
        wrapper_select=WrapperSelection.from_argN(cli_args.wrapper_select),
        wrapper_model=WrapperSelectionModel.from_arg(cli_args.wrapper_model),
        # n_feat=cli_args.n_feat,
        n_feat_wrapper=cli_args.n_feat_wrapper,
        n_feat_filter=cli_args.n_feat_filter,
        n_filter_cont=cli_args.n_filter_cont,
        n_filter_cat=cli_args.n_filter_cat,
        filter_assoc_cont_cls=ContClsStats.from_arg(cli_args.filter_assoc_cont_classify),
        filter_assoc_cat_cls=CatClsStats.from_arg(cli_args.filter_assoc_cat_classify),
        filter_assoc_cont_reg=ContRegStats.from_arg(cli_args.filter_assoc_cont_regress),
        filter_assoc_cat_reg=CatRegStats.from_arg(cli_args.filter_assoc_cat_regress),
        filter_pred_cls_score=ClsScore.from_arg(cli_args.filter_pred_classify),
        filter_pred_reg_score=RegScore.from_arg(cli_args.filter_pred_regress),
        is_classification=is_cls,
        classifiers=classifiers,
        regressors=regressors,
        # htune=cli_args.htune,
        # htune_val=cli_args.htune_val,
        # htune_val_size=cli_args.htune_val_size,
        htune_trials=cli_args.htune_trials,
        htune_cls_metric=ClassifierScorer.from_arg(cli_args.htune_cls_metric),
        htune_reg_metric=RegressorScorer.from_arg(cli_args.htune_reg_metric),
        # test_val=cli_args.test_val,
        test_val_size=cli_args.test_val_size,
        # mc_repeats=cli_args.mc_repeats,
        outdir=cli_args.outdir,
        is_spreadsheet=cli_args.spreadsheet is not None,
        separator=cli_args.separator,
        verbosity=cli_args.verbosity,
        no_warn_explosion=cli_args.no_warn_explosion,
    )


if __name__ == "__main__":
    parser = make_parser()
    args = get_parser_dict()
    for arg, info in args.items():
        print(f"{arg}: {info}")
