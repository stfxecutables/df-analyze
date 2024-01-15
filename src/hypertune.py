from __future__ import annotations

import os
import re
import traceback
import warnings
from dataclasses import dataclass, field
from pprint import pprint
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)
from warnings import warn

import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Index, Series
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
)

from legacy.src.objectives import (
    bagging_classifier_objective,
    dtree_classifier_objective,
    mlp_classifier_objective,
    rf_classifier_objective,
    svm_classifier_objective,
)
from src._constants import SEED, VAL_SIZE
from src._types import Classifier, CVMethod, EstimationMode, Estimator
from src.cli.cli import ProgramOptions

if TYPE_CHECKING:
    from src.models.base import DfAnalyzeModel
import jsonpickle

from src.enumerables import ClassifierScorer, RegressorScorer
from src.models.mlp import MLPEstimator
from src.preprocessing.prepare import PreparedData
from src.scoring import (
    sensitivity,
    specificity,
)
from src.selection.embedded import EmbedSelected
from src.selection.filter import FilterSelected
from src.selection.models import ModelSelected
from src.selection.wrapper import WrapperSelected

Splits = Iterable[Tuple[ndarray, ndarray]]

LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_LAYER_SIZES = [4, 8, 16, 32]
N_SPLITS = 5


@dataclass
class HtuneResult:
    selection: str
    model_cls: Type[DfAnalyzeModel]
    model: DfAnalyzeModel
    params: dict[str, Any]
    score: float

    def to_row(self) -> DataFrame:
        return DataFrame(
            {
                "selection": self.selection,
                "model": self.model.shortname,
                "params": str(jsonpickle.encode(self.params)),
                "score": self.score,
            },
            index=[0],
        )


@dataclass
class EvaluationResults:
    df: DataFrame
    results: list[HtuneResult]
    is_classification: bool

    def hp_table(self) -> DataFrame:
        dfs = []
        for res in self.results:
            dfs.append(res.to_row())
        df = pd.concat(dfs, axis=0, ignore_index=True)
        return df

    def wide_table(
        self, valset: Literal["5-fold", "trainset", "holdout"] = "5-fold"
    ) -> DataFrame:
        cols = ["trainset", "holdout", "5-fold"]
        cols.remove(valset)
        col = valset
        df = (
            self.df.drop(columns=cols)
            .pivot(columns="metric", values=col, index=["model", "selection"])
            .reset_index()
        )
        sorter = "acc" if self.is_classification else "mae"
        ascending = not self.is_classification
        return df.sort_values(by=sorter, ascending=ascending)

    def to_markdown(self) -> str:
        df_train = self.wide_table("trainset")
        df_hold = self.wide_table("holdout")
        df_fold = self.wide_table("5-fold")
        tab_train = df_train.to_markdown(tablefmt="simple", floatfmt="0.3f", index=False)
        tab_hold = df_hold.to_markdown(tablefmt="simple", floatfmt="0.3f", index=False)
        tab_fold = df_fold.to_markdown(tablefmt="simple", floatfmt="0.3f", index=False)
        text = (
            "# Final Model Performances\n\n"
            "## Training set performance\n\n"
            f"{tab_train}\n\n"
            "## Holdout set performance\n\n"
            f"{tab_hold}\n\n"
            "## 5-fold performance on holdout set\n\n"
            f"{tab_fold}\n\n"
        )
        return text

    def to_json(self) -> str:
        return str(jsonpickle.encode(self))


def _get_cols(
    selection: Literal["none", "assoc", "pred", "embed", "wrap"],
    selected: Optional[list[str]],
) -> Union[list[str], slice]:
    if selection == "none" or (selected is None):
        return slice(None)

    if selection == "assoc":
        # dealing with level-specific metrics
        selected = sorted(set([re.sub("__.*", "", str(c)) for c in selected]))
    return selected


def _get_splits(
    prep_train: PreparedData,
    prep_test: PreparedData,
    selection: str,
    cols: Union[list[str], slice],
) -> tuple[DataFrame, DataFrame]:
    if selection == "none" or isinstance(cols, slice):
        X_train = prep_train.X
        X_test = prep_test.X
    else:
        # clunky way to deal with renames and indicators
        try:
            X_train = prep_train.X.filter(regex="|".join(cols))
            X_test = prep_test.X.filter(regex="|".join(cols))
        except Exception:
            to_drop = set()
            for col in cols:  # may be e.g. colname__target.1
                for c in prep_train.X.columns:
                    cname = os.path.commonprefix([col, c])
                    if cname == "":
                        continue
                    if cname in [col, c]:  # handle e.g. car1, car2
                        to_drop.add(c)
            X_train = prep_train.X.drop(list(to_drop))
            X_test = prep_test.X.drop(list(to_drop))

    return X_train, X_test


def evaluate_tuned(
    prepared: PreparedData,
    prep_train: PreparedData,
    prep_test: PreparedData,
    assoc_filtered: FilterSelected,
    pred_filtered: FilterSelected,
    model_selected: ModelSelected,
    options: ProgramOptions,
) -> EvaluationResults:
    model_cls: Union[Type[DfAnalyzeModel], Type[MLPEstimator]]
    results: list[HtuneResult]
    dfs: list[DataFrame]

    selections: dict[
        Literal["none", "assoc", "pred", "embed", "wrap"], Optional[list[str]]
    ] = {
        "none": None,
        "assoc": assoc_filtered.selected,
        "pred": pred_filtered.selected,
    }
    if model_selected.embed_selected is not None:
        selections["embed"] = model_selected.embed_selected.selected
    if model_selected.wrap_selected is not None:
        selections["wrap"] = model_selected.wrap_selected.selected

    dfs, results = [], []
    for model_cls in options.models:
        for selection, cols in selections.items():
            selected_cols = _get_cols(selection=selection, selected=cols)
            X_train, X_test = _get_splits(prep_train, prep_test, selection, selected_cols)
            if X_train.empty or X_test.empty:
                raise ValueError(
                    f"Error when subsetting features for model '{model_cls.shortname}'. Got:\n"
                    f"cols: {cols}\n"
                    f"selected_cols: {selected_cols}\n"
                )
            if X_train.isna().any().any() or X_test.isna().any().any():
                raise ValueError(
                    f"Got NaNs when subsetting features for model '{model_cls.shortname}'. Got:\n"
                    f"cols: {cols}\n"
                    f"selected_cols: {selected_cols}\n"
                )

            if model_cls is MLPEstimator:
                model = model_cls(num_classes=prepared.num_classes)  # type: ignore
            else:
                model = model_cls()

            print(f"Tuning {model.longname} for selection={selection}")
            try:
                study = model.htune_optuna(
                    X_train=X_train,
                    y_train=prep_train.y,
                    n_trials=100,
                    n_jobs=-1,
                    verbosity=optuna.logging.ERROR,
                )
                df = model.htune_eval(
                    X_train=X_train,
                    y_train=prep_train.y,
                    X_test=X_test,
                    y_test=prep_test.y,
                )
                result = HtuneResult(
                    selection=selection,
                    model_cls=model_cls,
                    model=model,
                    params=study.best_params,
                    score=study.best_value,
                )
                results.append(result)
            except Exception as e:
                if prepared.is_classification:
                    nulls = Series(ClassifierScorer.null_scores(), name="metric")
                else:
                    nulls = Series(RegressorScorer.null_scores(), name="metric")
                warn(
                    f"Got exception when trying to tune and evaluate {model.shortname}:\n{e}\n"
                    f"{traceback.format_exc()}"
                )
                df = DataFrame(
                    {"trainset": nulls, "holdout": nulls, "5-fold": nulls},
                    index=Index(data=nulls.values, name=nulls.name),
                )
                result = HtuneResult(
                    selection=selection,
                    model_cls=model_cls,
                    model=model,
                    params={},
                    score=np.nan,
                )
                results.append(result)
            df["model"] = model.shortname
            df["selection"] = selection
            dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return EvaluationResults(
        df=df, results=results, is_classification=prepared.is_classification
    )


@dataclass(init=True, repr=True, eq=True, frozen=True)
class HtuneResultLegacy:
    estimator: Estimator
    mode: EstimationMode
    n_trials: int
    cv_method: CVMethod
    val_acc: float = np.nan
    best_params: Dict = field(default_factory=dict)


def train_val_splits(
    df: DataFrame, is_classification: bool, val_size: float = VAL_SIZE
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Wrapper around `sklearn.model_selection.train_test_split` to return splits as `DataFrame`s
    instead of numpy arrays.

    Parameters
    ----------
    df: DataFrame
        Data with target in column named "target".

    mode: Literal["classify", "regress"]
        What kind of model we are running.

    val_size: float = 0.2
        Percent of data to reserve for validation

    Returns
    -------
    splits: [X_train, X_val, y_train, y_val]
    """
    train, val = train_test_split(
        df, test_size=val_size, random_state=SEED, shuffle=True, stratify=df.target
    )
    X_train = train.drop(columns="target")
    X_val = val.drop(columns="target")
    y_train = train.target
    y_val = val.target
    if is_classification:
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
    return X_train, X_val, y_train, y_val


def cv_desc(cv_method: CVMethod, n_folds: Optional[int] = None) -> str:
    """Helper for logging a readable description of the CVMethod to stdout"""
    if isinstance(cv_method, int):
        return f"stratified {cv_method}-fold"
    if isinstance(cv_method, float):  # stratified holdout
        if cv_method <= 0 or cv_method >= 1:
            raise ValueError("Holdout test_size must be in (0, 1)")
        perc = int(100 * cv_method)
        return f"stratified {perc}% holdout"
    cv_method = str(cv_method).lower()  # type: ignore

    if n_folds is None:
        n_folds = 5
    if cv_method == "loocv":
        return "LOOCV"
    if cv_method in ["kfold", "k-fold"]:
        return f"{n_folds}-fold"
    if cv_method == "mc":
        return "stratified Monte-Carlo (20 random 20%-sized test sets)"
    raise ValueError(f"Invalid `cv_method`: {cv_method}")


def package_classifier_cv_scores(
    scores: Dict[str, ndarray],
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    result = dict(
        htuned=htuned,
        cv_method=htuned.cv_method,
        acc=np.mean(scores["test_acc"]),
        auc=np.mean(scores["test_auroc"]),
        sens=np.mean(scores["test_sens"]),
        spec=np.mean(scores["test_spec"]),
        acc_sd=np.std(scores["test_acc"], ddof=1),
        auc_sd=np.std(scores["test_auroc"], ddof=1),
        sens_sd=np.std(scores["test_sens"], ddof=1),
        spec_sd=np.std(scores["test_spec"], ddof=1),
    )
    if not log:
        return result

    acc_mean = float(np.mean(scores["test_acc"]))
    auc_mean = float(np.mean(scores["test_auroc"]))
    sens_mean = float(np.mean(scores["test_sens"]))
    spec_mean = float(np.mean(scores["test_spec"]))
    acc_sd = float(np.std(scores["test_acc"], ddof=1))
    auc_sd = float(np.std(scores["test_auroc"], ddof=1))
    sens_sd = float(np.std(scores["test_sens"], ddof=1))
    spec_sd = float(np.std(scores["test_spec"], ddof=1))
    desc = cv_desc(cv_method)
    # fmt: off
    print(f"Testing validation: {desc}")
    print(f"Accuracy:           μ = {np.round(acc_mean, 3):0.3f} (sd = {np.round(acc_sd, 4):0.4f})")  # noqa
    print(f"AUC:                μ = {np.round(auc_mean, 3):0.3f} (sd = {np.round(auc_sd, 4):0.4f})")  # noqa
    print(f"Sensitivity:        μ = {np.round(sens_mean, 3):0.3f} (sd = {np.round(sens_sd, 4):0.4f})")  # noqa
    print(f"Specificity:        μ = {np.round(spec_mean, 3):0.3f} (sd = {np.round(spec_sd, 4):0.4f})")  # noqa
    # fmt: on
    return result


def package_classifier_scores(
    y_test: ndarray,
    y_pred: ndarray,
    y_score: ndarray,
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    sens = sensitivity(y_test, y_pred)
    spec = specificity(y_test, y_pred)
    percent = int(100 * float(cv_method))
    scores = dict(
        test_accuracy=np.array([acc]).ravel(), test_roc_auc=np.array([auc]).ravel()
    )
    result = dict(
        htuned=htuned,
        cv_method=cv_method,
        acc=np.mean(scores["test_acc"]),
        auc=np.mean(scores["test_auroc"]),
        sens=sens,
        spec=spec,
        acc_sd=np.nan,
        auc_sd=np.nan,
        sens_sd=np.nan,
        spec_sd=np.nan,
    )
    if not log:
        return result
    print(f"Testing validation: {percent}% holdout")
    print(
        f"          Accuracy: μ = {np.round(acc, 3):0.3f} (sd = {np.round(acc, 4):0.4f})"
    )
    print(
        f"               AUC: μ = {np.round(auc, 3):0.3f} (sd = {np.round(auc, 4):0.4f})"
    )
    return result


def package_regressor_cv_scores(
    scores: Dict[str, ndarray],
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    result = dict(
        htuned=htuned,
        cv_method=htuned.cv_method,
        mae=np.mean(scores["test_mae"]),
        msqe=np.mean(scores["test_msqe"]),
        mdae=np.mean(scores["test_mdae"]),
        mape=np.mean(scores["test_mape"]),
        r2=np.mean(scores["test_r2"]),
        var_exp=np.mean(scores["test_var-exp"]),
        mae_sd=np.std(scores["test_mae"], ddof=1),
        msqe_sd=np.std(scores["test_msqe"], ddof=1),
        mdae_sd=np.std(scores["test_mdae"], ddof=1),
        mape_sd=np.std(scores["test_mape"], ddof=1),
        r2_sd=np.std(scores["test_r2"], ddof=1),
        var_exp_sd=np.std(scores["test_var-exp"], ddof=1),
    )
    if not log:
        return result

    mae = np.mean(scores["test_mae"])
    msqe = np.mean(scores["test_msqe"])
    mdae = np.mean(scores["test_mdae"])
    # mape = np.mean(scores["test_mape"])
    r2 = np.mean(scores["test_r2"])
    var_exp = np.mean(scores["test_var-exp"])
    mae_sd = np.std(scores["test_mae"], ddof=1)
    msqe_sd = np.std(scores["test_msqe"], ddof=1)
    mdae_sd = np.std(scores["test_mdae"], ddof=1)
    # mape_sd = np.std(scores["test_mape"], ddof=1)
    r2_sd = np.std(scores["test_r2"], ddof=1)
    var_exp_sd = np.std(scores["test_var exp"], ddof=1)

    desc = cv_desc(cv_method)
    # fmt: off
    print(f"Testing validation: {desc}")
    print(f"MAE             μ = {np.round(mae, 3):0.3f} (sd = {np.round(mae_sd, 4):0.4f})")  # noqa
    print(f"MSqE:           μ = {np.round(msqe, 3):0.3f} (sd = {np.round(msqe_sd, 4):0.4f})")  # noqa
    print(f"Median Abs Err: μ = {np.round(mdae, 3):0.3f} (sd = {np.round(mdae_sd, 4):0.4f})")  # noqa
    # print(f"MAPE:           μ = {np.round(mape, 3):0.3f} (sd = {np.round(mape_sd, 4):0.4f})")  # noqa
    print(f"R-squared:      μ = {np.round(r2, 3):0.3f} (sd = {np.round(r2_sd, 4):0.4f})")  # noqa
    print(f"Var explained:  μ = {np.round(var_exp, 3):0.3f} (sd = {np.round(var_exp_sd, 4):0.4f})")  # noqa
    # fmt: on
    return result


def package_regressor_scores(
    y_test: ndarray,
    y_pred: ndarray,
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    scores = {
        "test_MAE": mean_absolute_error(y_test, y_pred),
        "test_MSqE": mean_squared_error(y_test, y_pred),
        "test_MdAE": median_absolute_error(y_test, y_pred),
        "test_MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "test_R2": r2_score(y_test, y_pred),
        "test_Var exp": explained_variance_score(y_test, y_pred),
    }
    if log:
        percent = int(100 * float(cv_method))
        print(f"Testing validation: {percent}% holdout")
    with warnings.catch_warnings():
        # suppress warnings for taking std of length 1 ndarray:
        # RuntimeWarning: Degrees of freedom <= 0 for slice
        # RuntimeWarning: invalid value encountered in double_scalars
        warnings.simplefilter("ignore", RuntimeWarning)
        return package_regressor_cv_scores(scores, htuned, cv_method, log)


"""See Optuna docs (https://optuna.org/#code_ScikitLearn) for the motivation behond the closures
below. Currently I am using closures, but this might be a BAD IDEA in parallel contexts. In any
case, they do seem to suggest this is OK https://optuna.readthedocs.io/en/stable/faq.html
#how-to-define-objective-functions-that-have-own-arguments, albeit by using classes or lambdas. """


def hypertune_classifier(
    classifier: Classifier,
    X_train: DataFrame,
    y_train: DataFrame,
    n_trials: int = 200,
    cv_method: CVMethod = 5,
    verbosity: int = optuna.logging.ERROR,
) -> HtuneResultLegacy:
    """Core function. Uses Optuna base TPESampler (Tree-Parzen Estimator Sampler) to perform
    Bayesian hyperparameter optimization via Gaussian processes on the classifier specified in
    `classifier`.

    Parameters
    ----------
    classifier: Classifier
        Classifier to tune.

    X_train: DataFrame
        DataFrame with no target value (features only). Shape (n_samples, n_features)

    y_train: DataFrame
        Target values. Shape (n_samples,).

    n_trials: int = 200
        Number of trials to use with Optuna.

    cv_method: CVMethod = 5
        How to evaluate accuracy during tuning.

    verbosity: int = optuna.logging.ERROR
        See https://optuna.readthedocs.io/en/stable/reference/logging.html. Most useful other option
        is `optuna.logging.INFO`.

    Returns
    -------
    htuned: HtuneResultLegacy
        See top of this file.
    """
    OBJECTIVES: Dict[str, Callable] = {
        "rf": rf_classifier_objective(X_train, y_train, cv_method),
        "svm": svm_classifier_objective(X_train, y_train, cv_method),
        "dtree": dtree_classifier_objective(X_train, y_train, cv_method),
        "mlp": mlp_classifier_objective(X_train, y_train, cv_method),
        "bag": bagging_classifier_objective(X_train, y_train, cv_method),
    }
    # HYPERTUNING
    objective = OBJECTIVES[classifier]
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )
    optuna.logging.set_verbosity(verbosity)
    if classifier == "mlp":
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ[
            "PYTHONWARNINGS"
        ] = "ignore"  # can't kill ConvergenceWarning any other way
        study.optimize(objective, n_trials=n_trials)
        os.environ["PYTHONWARNINGS"] = before
    else:
        study.optimize(objective, n_trials=n_trials)

    val_method = cv_desc(cv_method)
    acc = np.round(study.best_value, 3)
    if verbosity != optuna.logging.ERROR:
        print(f"\n{' Tuning Results ':=^80}")
        print("Best params:")
        pprint(study.best_params, indent=4, width=80)
        print(f"\nTuning validation: {val_method}")
        print(f"Best accuracy:      μ = {acc:0.3f}")
        # print("=" * 80, end="\n")

    return HtuneResultLegacy(
        estimator=classifier,
        mode="classify",
        n_trials=n_trials,
        cv_method=cv_method,
        val_acc=study.best_value,
        best_params=study.best_params,
    )
