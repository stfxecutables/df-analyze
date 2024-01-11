from __future__ import annotations

"""
File for defining all options passed to `df-analyze.py`.
"""
import os
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import choice, randint, uniform
from typing import (
    TYPE_CHECKING,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
from warnings import warn

import jsonpickle
import pandas as pd
from pandas import DataFrame

from src._constants import (
    CLASSIFIERS,
    FULL_RESULTS,
    HTUNE_VAL_METHODS,
    N_WRAPPER_DEFAULT,
    P_FILTER_CAT_DEFAULT,
    P_FILTER_CONT_DEFAULT,
    P_FILTER_TOTAL_DEFAULT,
    REGRESSORS,
    SENTINEL,
)
from src._types import (
    ValMethod,
)
from src.analysis.univariate.associate import CatClsStats, CatRegStats, ContClsStats, ContRegStats
from src.cli.parsing import (
    cv_size,
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
    DF_HELP_STR,
    DROP_HELP_STR,
    EMBED_SELECT_MODEL_HELP,
    EXPLODE_HELP,
    FEAT_SELECT_HELP,
    FILTER_METHOD_HELP,
    HTUNE_HELP,
    HTUNE_TRIALS_HELP,
    HTUNE_VALSIZE_HELP,
    HTUNEVAL_HELP_STR,
    MC_REPEATS_HELP,
    MODE_HELP_STR,
    MODEL_SELECT_HELP,
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
    SELECT_TUNE_ROUNDS_HELP,
    SEP_HELP_STR,
    SHEET_HELP_STR,
    TARGET_HELP_STR,
    TEST_VAL_HELP,
    TEST_VALSIZES_HELP,
    USAGE_EXAMPLES,
    USAGE_STRING,
    VERBOSITY_HELP,
    WRAP_SELECT_HELP,
    WRAP_SELECT_MODEL_HELP,
)
from src.enumerables import (
    ClsScore,
    DfAnalyzeClassifier,
    DfAnalyzeRegressor,
    EmbedSelectionModel,
    FeatureSelection,
    FilterSelection,
    ModelFeatureSelection,
    NanHandling,
    Normalization,
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
    # feat_clean: Tuple[FeatureCleaning, ...]
    nan_handling: NanHandling
    norm: Normalization


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
    is_classification: bool
    classifiers: Tuple[DfAnalyzeClassifier, ...]
    regressors: Tuple[DfAnalyzeRegressor, ...]
    feat_select: Tuple[FeatureSelection, ...]
    model_select: Optional[DfAnalyzeModel]
    embed_select: Optional[EmbedSelectionModel]
    wrapper_select: Optional[WrapperSelection]
    wrapper_model: Optional[WrapperSelectionModel]
    n_feat: int
    n_filter_cont: Union[int, float]
    n_filter_cat: Union[int, float]
    n_feat_filter: Union[int, float]
    n_feat_wrapper: Union[int, float, None]
    filter_assoc_cont_cls: ContClsStats
    filter_assoc_cat_cls: CatClsStats
    filter_assoc_cont_reg: ContRegStats
    filter_assoc_cat_reg: CatRegStats
    filter_pred_cls_score: ClsScore
    filter_pred_reg_score: RegScore


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
        model_select: Optional[DfAnalyzeModel],
        embed_select: Optional[EmbedSelectionModel],
        wrapper_select: Optional[WrapperSelection],
        wrapper_model: WrapperSelectionModel,
        n_feat: int,
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
        self.cleaning_options: CleaningOptions
        self.selection_options: SelectionOptions
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
        self.model_select: Optional[DfAnalyzeModel] = model_select
        self.embed_select: Optional[EmbedSelectionModel] = embed_select
        self.wrapper_select: Optional[WrapperSelection] = wrapper_select
        self.wrapper_model: WrapperSelectionModel = wrapper_model
        self.n_feat: int = n_feat
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
        self.classifiers: Tuple[DfAnalyzeClassifier, ...] = tuple(sorted(set(classifiers)))
        self.regressors: Tuple[DfAnalyzeRegressor, ...] = tuple(sorted(set(regressors)))
        # self.htune: bool = htune
        # self.htune_val: ValMethod = htune_val
        # self.htune_val_size: Size = htune_val_size
        self.htune_trials: int = htune_trials
        # self.test_val: ValMethod = test_val
        self.test_val_size: Size = test_val_size
        # self.mc_repeats: int = mc_repeats
        # TODO: fix below
        self.outdir: Optional[Path] = self.get_outdir(outdir, self.datapath)
        self.program_dirs: ProgramDirs = ProgramDirs.new(self.outdir)
        self.is_spreadsheet: bool = is_spreadsheet
        self.separator: str = separator
        self.verbosity: Verbosity = verbosity
        self.no_warn_explosion: bool = no_warn_explosion

        # cleanup
        if DfAnalyzeClassifier.Dummy not in self.classifiers:
            self.classifiers = (DfAnalyzeClassifier.Dummy, *self.classifiers)
        if DfAnalyzeRegressor.Dummy not in self.regressors:
            self.regressors = (DfAnalyzeRegressor.Dummy, *self.regressors)

        self.cleaning_options = CleaningOptions(
            datapath=self.datapath,
            target=self.target,
            categoricals=self.categoricals,
            ordinals=self.ordinals,
            drops=self.drops,
            # feat_clean=self.feat_clean,
            nan_handling=self.nan_handling,
            norm=self.norm,
        )
        self.selection_options = SelectionOptions(
            cleaning_options=self.cleaning_options,
            is_classification=self.is_classification,
            classifiers=self.classifiers,
            regressors=self.regressors,
            feat_select=self.feat_select,
            model_select=self.model_select,
            embed_select=self.embed_select,
            wrapper_select=self.wrapper_select,
            wrapper_model=self.wrapper_model,
            n_feat=self.n_feat,
            n_filter_cont=self.n_filter_cont,
            n_filter_cat=self.n_filter_cat,
            n_feat_filter=self.n_feat_filter,
            n_feat_wrapper=self.n_feat_wrapper,
            filter_assoc_cont_cls=self.filter_assoc_cont_cls,
            filter_assoc_cat_cls=self.filter_assoc_cat_cls,
            filter_assoc_cont_reg=self.filter_assoc_cont_reg,
            filter_assoc_cat_reg=self.filter_assoc_cat_reg,
            filter_pred_cls_score=self.filter_pred_cls_score,
            filter_pred_reg_score=self.filter_pred_reg_score,
        )

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
    def random(ds: TestDataset) -> ProgramOptions:
        n_samples, n_feats = ds.shape
        # feat_clean = FeatureCleaning.random_n()
        feat_select = FeatureSelection.random_n()
        wrap_select = WrapperSelection.random()
        wrap_model = WrapperSelectionModel.random()
        model_select = ModelFeatureSelection.random()
        embed_select = EmbedSelectionModel.random_none()
        n_feat = n_feats
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
        # test_val: ValMethod = "kfold"
        test_val_size: Size = 0.4
        # mc_repeats: int = 0
        sub = "classification" if ds.is_classification else "regression"
        outdir: Path = (FULL_RESULTS / sub) / ds.dsname
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
            model_select=model_select,
            embed_select=embed_select,
            wrapper_select=wrap_select,
            wrapper_model=wrap_model,
            n_feat=n_feat,
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
            self.__dict__, ignores=["cleaning_options", "selection_options", "comparables"]
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
        sentinels = {key: SENTINEL for key in parser.parse_known_args(args.split())[0].__dict__}
    sentinel_parser.set_defaults(**sentinels)

    cli_args = (
        cli_parser.parse_known_args()[0]
        if args is None
        else cli_parser.parse_known_args(args.split())[0]
    )
    if cli_args.spreadsheet is None and cli_args.df is None:
        raise ValueError("Must specify one of either `--spreadsheet [file]` or `--df [file]`.")

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
        choices=DfAnalyzeClassifier.choices(),
        default=DfAnalyzeClassifier.defaults(),
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
    # parser.add_argument(
    #     "--feat-select",
    #     nargs="+",
    #     type=FeatureSelection,  # applied to each spaced element in arg
    #     choices=[f.value for f in FeatureSelection],
    #     default=(FeatureSelection.Filter,),
    #     help=FEAT_SELECT_HELP,
    # )
    parser.add_argument(
        "--model-select",
        nargs="+",
        type=ModelFeatureSelection,  # applied to each spaced element in arg
        choices=[m.value for m in ModelFeatureSelection],
        default=(ModelFeatureSelection.Embedded,),
        help=MODEL_SELECT_HELP,
    )
    parser.add_argument(
        "--embed-select",
        nargs="+",
        type=lambda x: None if "none" in x.lower() else EmbedSelectionModel(x),
        choices=[m.value for m in EmbedSelectionModel] + ["none"],
        default=(EmbedSelectionModel.LGBM,),
        help=EMBED_SELECT_MODEL_HELP,
    )
    parser.add_argument(
        "--wrapper-select",
        type=WrapperSelection,
        choices=[w.value for w in WrapperSelection],
        default=WrapperSelection.StepUp,
        help=WRAP_SELECT_HELP,
    )
    parser.add_argument(
        "--wrapper-model",
        type=WrapperSelectionModel,
        choices=[w.value for w in WrapperSelectionModel],
        default=WrapperSelectionModel.Linear,
        help=WRAP_SELECT_MODEL_HELP,
    )
    parser.add_argument(  # TODO UNIMPLEMENTED!
        "--embed-model",
        type=EmbedSelectionModel,
        choices=[s.value for s in EmbedSelectionModel],
        default=(EmbedSelectionModel.LGBM,),
        help=EMBED_SELECT_MODEL_HELP,
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
        choices=[n.value for n in Normalization],
        default=Normalization.Robust,
        type=Normalization,
        help=NORM_HELP,
    )
    parser.add_argument(
        "--nan",
        choices=[n.value for n in NanHandling],
        default=NanHandling.Mean,
        type=NanHandling,
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
        type=FilterSelection,
        default=FilterSelection.Relief,
        help=FILTER_METHOD_HELP,
    )
    parser.add_argument(
        "--filter-assoc-cont-classify",
        choices=[c.value for c in ContClsStats],
        type=ContClsStats,
        default=ContClsStats.default(),
        help=ASSOC_SELECT_CONT_CLS_STATS,
    )
    parser.add_argument(
        "--filter-assoc-cat-classify",
        choices=[c.value for c in CatClsStats],
        type=CatClsStats,
        default=CatClsStats.default(),
        help=ASSOC_SELECT_CAT_CLS_STATS,
    )
    parser.add_argument(
        "--filter-assoc-cont-regress",
        choices=[c.value for c in ContRegStats],
        type=ContRegStats,
        default=ContRegStats.default(),
        help=ASSOC_SELECT_CONT_REG_STATS,
    )
    parser.add_argument(
        "--filter-assoc-cat-regress",
        choices=[c.value for c in CatRegStats],
        type=CatRegStats,
        default=CatRegStats.default(),
        help=ASSOC_SELECT_CAT_REG_STATS,
    )
    parser.add_argument(
        "--filter-pred-regress",
        choices=[s.value for s in RegScore],
        type=RegScore,
        default=RegScore.default(),
        help=PRED_SELECT_REG_SCORE,
    )
    parser.add_argument(
        "--filter-pred-classify",
        choices=[s.value for s in ClsScore],
        type=ClsScore,
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
        norm=cli_args.norm,
        nan_handling=cli_args.nan,
        feat_select=cli_args.feat_select,
        model_select=cli_args.model_select,
        embed_select=cli_args.embed_select,
        wrapper_select=cli_args.wrapper_select,
        wrapper_model=cli_args.wrapper_model,
        n_feat=cli_args.n_feat,
        n_feat_wrapper=cli_args.n_feat_wrapper,
        n_feat_filter=cli_args.n_feat_filter,
        n_filter_cont=cli_args.n_filter_cont,
        n_filter_cat=cli_args.n_filter_cat,
        filter_assoc_cont_cls=cli_args.filter_assoc_cont_classify,
        filter_assoc_cat_cls=cli_args.filter_assoc_cat_classify,
        filter_assoc_cont_reg=cli_args.filter_assoc_cont_regress,
        filter_assoc_cat_reg=cli_args.filter_assoc_cat_regress,
        filter_pred_cls_score=cli_args.filter_pred_cls_score,
        filter_pred_reg_score=cli_args.filter_pred_reg_score,
        is_classification=cli_args.mode,
        classifiers=cli_args.classifiers,
        regressors=cli_args.regressors,
        # htune=cli_args.htune,
        # htune_val=cli_args.htune_val,
        # htune_val_size=cli_args.htune_val_size,
        htune_trials=cli_args.htune_trials,
        # test_val=cli_args.test_val,
        test_val_size=cli_args.test_val_size,
        # mc_repeats=cli_args.mc_repeats,
        outdir=cli_args.outdir,
        is_spreadsheet=cli_args.spreadsheet is not None,
        separator=cli_args.separator,
        verbosity=cli_args.verbosity,
        no_warn_explosion=cli_args.no_warn_explosion,
    )
