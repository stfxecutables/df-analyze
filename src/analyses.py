from typing import Any, Dict, List, Optional, Union

import optuna
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from src._types import (
    Classifier,
    CVMethod,
    Estimator,
    FeatureSelection,
    MultiTestCVMethod,
    Regressor,
)
from src.cleaning import get_clean_data
from src.cli import ProgramOptions, Verbosity
from src.feature_selection import select_features
from src.hypertune import (
    HtuneResult,
    evaluate_hypertuned,
    hypertune_classifier,
    hypertune_regressor,
    train_val_splits,
)
from src.io import FileType, try_save


def val_method_short(method: CVMethod, test_val_size: int) -> str:
    """Helper for shortening CVMethod for labeling purposes"""
    if isinstance(method, int):
        return f"{method}-fold"
    elif isinstance(method, float):
        return f"{int(100*method)}%-holdout"
    elif str(method).lower() in ["kfold", "k-fold"]:
        return f"{test_val_size}-fold"
    elif str(method).lower() == "mc":
        return "m-carlo"
    elif str(method).lower() == "loocv":
        return "loocv"
    elif str(method).lower() == "holdout":
        return f"{int(100*test_val_size)}%-holdout"
    else:
        return "none"


def results_df(
    options: ProgramOptions,
    feature_selection: FeatureSelection,
    test_val_size: int,
    result: Dict[str, Any],
) -> DataFrame:
    """Package the results of hypertuning into a convenient DataFrame summary.

    Parameters
    ----------
    feature_selection: FeatureSelection
        How features were selected.

    n_features: int
        Number of features (columns) that were selected.

    trials: int
        Number of trials used in Optuna optimization.

    test_validation: CVMethod
        Method used for final validation following hypertuning.

    result: Dict[str, Any]
        The dict returned from `src.hypertune.evaluate_hypertuned`.

    Returns
    -------
    df: DataFrame
        Summary dataframe.
    """
    htuned: HtuneResult = result.pop("htuned")
    cv_method = result.pop("cv_method")
    test_validation = options.test_val
    test_val = val_method_short(test_validation, test_val_size)
    htune_val = val_method_short(cv_method, test_val_size)
    row = {
        "model": htuned.estimator,
        "feat_select": feature_selection,
        "n_feat": options.selection_options.n_feat,
        "test_val": test_val,
        **result,
        "htune_val": htune_val,
        "htune_trials": options.htune_trials,
    }
    return DataFrame([row])


def classifier_analysis(
    options: ProgramOptions,
    classifier: Classifier,
    feature_selection: Optional[FeatureSelection] = "pca",
    verbosity: int = optuna.logging.ERROR,
) -> None:
    """Run a full analysis of a classifier and print results to stdout. No results are saved.

    Parameters
    ----------
    classifier: Classifier = "svm"
        The classifier to evaluate.

    feature_selection: Optional[FeatureSelection] = "pca"
        If "step-up" or "step-down", perform forward or backward stepwise feature selection with
        `sklearn.feature_selection.SequentialFeatureSelector`.
        If "pca", select the first `n_features` principal components.
        If "kpca", select the first `n_features` principal components via
        `sklearn.decomposition.KernelPCA`.
        If "d", "auc", "pearson", or "spearman", use Cohen's d, the AUC, or correlations to select
        the features with the strongest univariate relationship to the target.
        If None or "minimal", perform only basic cleaning and no feature selection.

    n_features: int = 20
        Number of features (columns) to select.

    htune_trials: int = 100
        Number of trials to run with Optuna to optimize hyperparameters.

    htune_validation: Union[int, float, Literal["loocv", "mc"]] = 5
        How to validate / compute scores during hyperparameter optimization.
        If an `int`, specifies k-fold and the value of `k`.
        If a float in (0, 1), specifies holdout validation with test_size=`htune_validation`.
        If "loocv", specifies LOOCV validation.
        If "mc", specifies Monte-Carlo validation with 20 random 10% holdouts.

    test_validation: Union[int, float, Literal["loocv", "mc"]] = 5
        How to validate / compute scores after hyperparameter optimization.
        If an `int`, specifies k-fold and the value of `k`.
        If a float in (0, 1), specifies holdout validation with test_size=`htune_validation`.
        If "loocv", specifies LOOCV validation.
        If "mc", specifies Monte-Carlo validation with 20 random 10% holdouts.

    verbosity: int = optuna.logging.ERROR
        Controls the amount of console spam produced by Optuna, as well as tqdm progress bars.
    """
    # df_all = get_clean_data()
    selection_options = options.selection_options
    test_val = options.test_val
    htune_trials = options.htune_trials
    htune_val = options.htune_val
    print(f"Preparing feature selection with method: {feature_selection}")
    df = select_features(options, feature_selection, classifier)
    if isinstance(test_val, float):  # set aside data for final test
        if test_val <= 0 or test_val >= 1:
            raise ValueError("`--test-val` must be in (0, 1)")
        X_train, X_test, y_train, y_test = train_val_splits(df, options.mode, test_val)
    else:
        X_train = df.drop(columns=options.target)
        X_test = None
        y_train = df[options.target].copy().astype(int)
        y_test = None
    htuned = hypertune_classifier(
        classifier=classifier,
        X_train=X_train,
        y_train=y_train,
        n_trials=htune_trials,
        cv_method=htune_val,
        verbosity=verbosity,
    )
    print(f"\n{' Testing Results ':=^80}\n")
    evaluate_hypertuned(
        htuned,
        cv_method=test_val,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def full_estimator_analysis(
    options: ProgramOptions,
    estimator: Estimator = "svm",
    feature_selection: Optional[FeatureSelection] = "pca",
) -> DataFrame:
    """Run a full analysis of a classifier or regressor and return a summary of the
    results. All listed options in `test_validations` (see below) will be performed
    efficiently without requiring hypertuning each time.

    Parameters
    ----------
    options: ProgramOptions
        See `src.cli.ProgramOptions` for details. Options are created by the
        CLI options, documented by running `python df-analyze.py --help`.

    estimator: Estimator = "svm"
        The classifier or regressor to evaluate. Default is either `sklearn.svm.svc`
        or `sklearn.svm.svr`.

    feature_selection: Optional[FeatureSelection] = "pca"
        If "step-up" or "step-down", perform forward or backward stepwise feature selection with
        `sklearn.feature_selection.SequentialFeatureSelector`.
        If "pca", select the first `n_features` principal components.
        If "kpca", select the first `n_features` principal components via
        `sklearn.decomposition.KernelPCA`.
        If "d", "auc", "pearson", or "spearman", use Cohen's d, the AUC, or correlations to select
        the features with the strongest univariate relationship to the target.
        If None or "minimal", perform only basic cleaning and no feature selection.

    Returns
    -------
    results: DataFrame
        A summary DataFrame of the test results and methods used.
    """
    verbosity = options.verbosity.value
    log = verbosity != Verbosity.ERROR
    optuna_verbosity = optuna.logging.ERROR if verbosity == 0 else optuna.logging.INFO
    if log:
        print(f"Preparing feature selection with method: {feature_selection}")
    selection_options = options.selection_options
    test_val_sizes = options.test_val_sizes
    htune_trials = options.htune_trials
    htune_val = options.htune_val

    df = select_features(options, feature_selection, estimator)
    targname = options.target or "target"
    X_raw = df.drop(columns=targname)
    X_train = StandardScaler().fit_transform(X_raw)
    y_train = df[targname].to_numpy()
    hypertune_estimator = hypertune_regressor
    if options.mode == "classify":
        y_train = y_train.astype(int)
        hypertune_estimator = hypertune_classifier
    htuned = hypertune_estimator(
        estimator,
        X_train=X_train,
        y_train=y_train,
        n_trials=htune_trials,
        cv_method=htune_val,
        verbosity=optuna_verbosity,
    )
    # TODO: save params
    try_save(
        program_dirs=options.program_dirs,
        df=Series(htuned.best_params).to_frame(),
        file_stem="params",
        file_type=FileType.Params,
        selection=feature_selection,
        cleaning=options.feat_clean,
    )
    if verbosity != optuna.logging.ERROR:
        print(f"\n{' Testing Results ':=^80}\n")
    results = []
    for test_val_size in test_val_sizes:
        result = evaluate_hypertuned(
            htuned, cv_method=test_val_size, X_train=X_train, y_train=y_train, log=log
        )
        results.append(
            results_df(
                options=options,
                feature_selection=feature_selection,
                test_val_size=test_val_size,
                result=result,
            )
        )
    all_results = pd.concat(results, axis=0, ignore_index=True)
    return all_results
