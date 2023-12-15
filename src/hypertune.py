import os
import traceback
import warnings
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from warnings import warn

import numpy as np
import optuna
from numpy import ndarray
from pandas import DataFrame
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
from src.scoring import (
    accuracy_scorer,
    auc_scorer,
    expl_var_scorer,
    mae_scorer,
    mape_scorer,
    mdae_scorer,
    mse_scorer,
    r2_scorer,
    sensitivity,
    sensitivity_scorer,
    specificity,
    specificity_scorer,
)

Splits = Iterable[Tuple[ndarray, ndarray]]

LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_LAYER_SIZES = [4, 8, 16, 32]
N_SPLITS = 5
CLASSIFIER_TEST_SCORERS = dict(
    acc=accuracy_scorer,
    auroc=auc_scorer,
    sens=sensitivity_scorer,
    spec=specificity_scorer,
)
REGRESSION_TEST_SCORERS = {
    "MAE": mae_scorer,
    "MSqE": mse_scorer,
    "MdAE": mdae_scorer,
    # "MAPE": mape_scorer,
    "R2": r2_scorer,
    "Var exp": expl_var_scorer,
}


@dataclass(init=True, repr=True, eq=True, frozen=True)
class HtuneResult:
    estimator: Estimator
    mode: EstimationMode
    n_trials: int
    cv_method: CVMethod
    val_acc: float = np.nan
    best_params: Dict = field(default_factory=dict)


def train_val_splits(
    df: DataFrame, mode: EstimationMode, val_size: float = VAL_SIZE
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
    if mode == "classify":
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
    htuned: HtuneResult,
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
    htuned: HtuneResult,
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
    htuned: HtuneResult,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    result = dict(
        htuned=htuned,
        cv_method=htuned.cv_method,
        mae=np.mean(scores["test_MAE"]),
        msqe=np.mean(scores["test_MSqE"]),
        mdae=np.mean(scores["test_MdAE"]),
        mape=np.mean(scores["test_MAPE"]),
        r2=np.mean(scores["test_R2"]),
        var_exp=np.mean(scores["test_Var exp"]),
        mae_sd=np.std(scores["test_MAE"], ddof=1),
        msqe_sd=np.std(scores["test_MSqE"], ddof=1),
        mdae_sd=np.std(scores["test_MdAE"], ddof=1),
        mape_sd=np.std(scores["test_MAPE"], ddof=1),
        r2_sd=np.std(scores["test_R2"], ddof=1),
        var_exp_sd=np.std(scores["test_Var exp"], ddof=1),
    )
    if not log:
        return result

    mae = np.mean(scores["test_MAE"])
    msqe = np.mean(scores["test_MSqE"])
    mdae = np.mean(scores["test_MdAE"])
    mape = np.mean(scores["test_MAPE"])
    r2 = np.mean(scores["test_R2"])
    var_exp = np.mean(scores["test_Var exp"])
    mae_sd = np.std(scores["test_MAE"], ddof=1)
    msqe_sd = np.std(scores["test_MSqE"], ddof=1)
    mdae_sd = np.std(scores["test_MdAE"], ddof=1)
    mape_sd = np.std(scores["test_MAPE"], ddof=1)
    r2_sd = np.std(scores["test_R2"], ddof=1)
    var_exp_sd = np.std(scores["test_Var exp"], ddof=1)

    desc = cv_desc(cv_method)
    # fmt: off
    print(f"Testing validation: {desc}")
    print(f"MAE             μ = {np.round(mae, 3):0.3f} (sd = {np.round(mae_sd, 4):0.4f})")  # noqa
    print(f"MSqE:           μ = {np.round(msqe, 3):0.3f} (sd = {np.round(msqe_sd, 4):0.4f})")  # noqa
    print(f"Median Abs Err: μ = {np.round(mdae, 3):0.3f} (sd = {np.round(mdae_sd, 4):0.4f})")  # noqa
    print(f"MAPE:           μ = {np.round(mape, 3):0.3f} (sd = {np.round(mape_sd, 4):0.4f})")  # noqa
    print(f"R-squared:      μ = {np.round(r2, 3):0.3f} (sd = {np.round(r2_sd, 4):0.4f})")  # noqa
    print(f"Var explained:  μ = {np.round(var_exp, 3):0.3f} (sd = {np.round(var_exp_sd, 4):0.4f})")  # noqa
    # fmt: on
    return result


def package_regressor_scores(
    y_test: ndarray,
    y_pred: ndarray,
    htuned: HtuneResult,
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
) -> HtuneResult:
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
    htuned: HtuneResult
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

    return HtuneResult(
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
) -> HtuneResult:
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
    htuned: HtuneResult
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

    return HtuneResult(
        mode="regress",
        estimator=regressor,
        n_trials=n_trials,
        cv_method=cv_method,
        val_acc=best_val,
        best_params=best_params,
    )


def evaluate_hypertuned(
    htuned: HtuneResult,
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
    htuned: HtuneResult
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
                htuned: HtuneResult,
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
