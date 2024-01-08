from __future__ import annotations

import os
import re
import traceback
import warnings
from dataclasses import dataclass, field
from enum import Enum
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
)
from warnings import warn

import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Index, Series
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import (
    train_test_split,
)

from legacy.src.classifiers import get_classifier_constructor
from legacy.src.objectives import (
    adaboost_regressor_objective,
    bagging_classifier_objective,
    dtree_classifier_objective,
    get_cv,
    gradboost_regressor_objective,
    knn_regressor_objective,
    linear_regressor_objective,
    mlp_args_from_params,
    mlp_classifier_objective,
    mlp_regressor_objective,
    rf_classifier_objective,
    rf_regressor_objective,
    svm_classifier_objective,
    svm_regressor_objective,
)
from legacy.src.regressors import get_regressor_constructor
from src._constants import SEED, VAL_SIZE
from src._types import Classifier, CVMethod, EstimationMode, Estimator, Regressor
from src.cli.cli import ProgramOptions

if TYPE_CHECKING:
    from src.models.base import DfAnalyzeModel
import jsonpickle

from src.enumerables import ClassifierScorer, RegressorScorer
from src.models.mlp import MLPEstimator
from src.preprocessing.prepare import PreparedData
from src.scoring import (
    CLASSIFIER_TEST_SCORERS,
    REGRESSION_TEST_SCORERS,
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


@dataclass
class EvaluationResults:
    df: DataFrame
    results: list[HtuneResult]
    is_classification: bool

    def wide_table(self, valset: Literal["5-fold", "trainset", "holdout"] = "5-fold") -> DataFrame:
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


def _get_cols(selection: str, cols: Any) -> list[str]:
    if selection == "assoc":
        # dealing with level-specific metrics
        cols = sorted(set([re.sub("__.*", "", str(c)) for c in cols]))

    if isinstance(cols, (EmbedSelected, WrapperSelected)):
        cols = cols.selected
    return cols


def _get_splits(
    prep_train: PreparedData, prep_test: PreparedData, selection: str, cols: list[str]
) -> tuple[DataFrame, DataFrame]:
    if selection == "none":
        X_train = prep_train.X
        X_test = prep_test.X
    else:
        # clunky way to deal with renames and indicators
        X_train = prep_train.X.filter(regex="|".join(cols))
        X_test = prep_test.X.filter(regex="|".join(cols))
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

    selections = {
        "none": [],
        "assoc": assoc_filtered.selected,
        "pred": pred_filtered.selected,
        "embed": model_selected.embed_selected,
        "wrap": model_selected.wrap_selected,
    }
    if selections["embed"] is None:
        selections.pop("embed")
    if selections["wrap"] is None:
        selections.pop("wrap")

    dfs, results = [], []
    for model_cls in options.models:
        for selection, cols in selections.items():
            cols = _get_cols(selection=selection, cols=cols)
            X_train, X_test = _get_splits(prep_train, prep_test, selection, cols)
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
                    f"{traceback.print_exc()}"
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
    return EvaluationResults(df=df, results=results, is_classification=prepared.is_classification)


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
    scores = dict(test_accuracy=np.array([acc]).ravel(), test_roc_auc=np.array([auc]).ravel())
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
    print(f"          Accuracy: μ = {np.round(acc, 3):0.3f} (sd = {np.round(acc, 4):0.4f})")
    print(f"               AUC: μ = {np.round(auc, 3):0.3f} (sd = {np.round(auc, 4):0.4f})")
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
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    optuna.logging.set_verbosity(verbosity)
    if classifier == "mlp":
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = "ignore"  # can't kill ConvergenceWarning any other way

    study.optimize(objective, n_trials=n_trials)

    if classifier == "mlp":
        os.environ["PYTHONWARNINGS"] = before

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


def hypertune_regressor(
    regressor: Regressor,
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
    regressor: Regressor
        Regressor to tune.

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
        "linear": linear_regressor_objective(X_train, y_train, cv_method),
        "rf": rf_regressor_objective(X_train, y_train, cv_method),
        "adaboost": adaboost_regressor_objective(X_train, y_train, cv_method),
        "gboost": gradboost_regressor_objective(X_train, y_train, cv_method),
        "svm": svm_regressor_objective(X_train, y_train, cv_method),
        "knn": knn_regressor_objective(X_train, y_train, cv_method),
        "mlp": mlp_regressor_objective(X_train, y_train, cv_method),
    }
    # HYPERTUNING
    objective = OBJECTIVES[regressor]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    optuna.logging.set_verbosity(verbosity)
    if regressor == "mlp":
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = "ignore"  # can't kill ConvergenceWarning any other way

    study.optimize(objective, n_trials=n_trials)

    if regressor == "mlp":
        os.environ["PYTHONWARNINGS"] = before

    val_method = cv_desc(cv_method)
    try:
        best_val = study.best_value
        best_params = study.best_params
    except ValueError:
        traceback.print_exc()
        warn(
            f"All Optuna trials for regressor {regressor} likely either failed "
            "or produced NaN values (full stack trace should be above). Likely "
            "this is due to an inappropriate feature selection method for the "
            "data (e.g. kpca) or a convergence issue. Setting best metric to "
            "NaN for now. "
        )
        best_val = np.nan
        best_params = {}
    mae = -np.round(best_val, 3)

    if verbosity != optuna.logging.ERROR:
        print(f"\n{' Tuning Results ':=^80}")
        print("Best params:")
        pprint(study.best_params, indent=4, width=80)
        print(f"\nTuning validation: {val_method}")
        print(f"Best MAE:       μ = {mae:0.3f}")
        # print("=" * 80, end="\n")

    return HtuneResultLegacy(
        mode="regress",
        estimator=regressor,
        n_trials=n_trials,
        cv_method=cv_method,
        val_acc=best_val,
        best_params=best_params,
    )


def evaluate_hypertuned(
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: Optional[DataFrame] = None,
    y_test: Optional[DataFrame] = None,
    n_folds: Optional[int] = None,
    log: bool = True,
) -> Dict[str, Any]:
    """Core function. Given the result of hypertuning, evaluate the final parameters.

    Parameters
    ----------
    htuned: HtuneResultLegacy
        Results from `src.hypertune.hypertune_classifier`.

    cv_method: CVMethod = 5
        How to evaluate accuracy during tuning.

    X_train: DataFrame
        DataFrame with no target value (features only). Shape (n_samples, n_features)

    y_train: DataFrame
        Target values. Shape (n_samples,).

    X_test: DataFrame = None
        DataFrame with no target value (features only). Shape (n_test_samples, n_features), if using
        a test sample held out during hyperparameter tuning.

    y_test: DataFrame = None
        Target values. Shape (n_test_samples,), corresponding to X_test.

    log: bool = True
        If True, print results to console.

    Returns
    -------
    result: Dict[str, Any]
        A dict with structure:

            {
                htuned: HtuneResultLegacy,
                cv_method: CVMethod,  # The method used during hypertuning
                acc: float  # mean accuracy across folds
                auc: float  # mean AUC across folds
                acc_sd: float  # sd of accuracy across folds
                auc_sd: float  # sd of AUC across folds
            }
    """
    model = htuned.estimator
    params = htuned.best_params
    args = mlp_args_from_params(params) if model == "mlp" else params
    if htuned.mode == "classify":
        estimator = get_classifier_constructor(model)(**args)
    else:
        estimator = get_regressor_constructor(model)(**args)
    SCORERS = CLASSIFIER_TEST_SCORERS if htuned.mode == "classify" else REGRESSION_TEST_SCORERS
    if (X_test is None) and (y_test is None):
        _cv = get_cv(y_train, cv_method, n_folds=n_folds)
        scores = cv(estimator, X=X_train, y=y_train, scoring=SCORERS, cv=_cv)
        if htuned.mode == "classify":
            return package_classifier_cv_scores(scores, htuned, cv_method, log)
        else:
            return package_regressor_cv_scores(scores, htuned, cv_method, log)
    elif (X_test is not None) and (y_test is not None):
        y_pred = estimator.fit(X_train, y_train).predict(X_test)
        if htuned.mode == "classify":
            y_score = (
                estimator.decision_function(X_test)
                if model != "mlp"
                else estimator.predict_proba(X_test)
            )
            return package_classifier_scores(
                y_test=y_test,
                y_pred=y_pred,
                y_score=y_score,
                htuned=htuned,
                cv_method=cv_method,
                log=log,
            )
        else:
            return package_regressor_scores(y_test, y_pred, htuned, cv_method, log)
    else:
        raise ValueError("Invalid test data: only one of `X_test` or `y_test` was None.")
