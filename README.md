[![DOI](https://zenodo.org/badge/364694785.svg)](https://zenodo.org/badge/latestdoi/364694785)

- [Setup and Requirements](#setup-and-requirements)
- [Running the Script](#running-the-script)
- [Usage](#usage)
- [Basic Example](#basic-example)

# Setup and Requirements

This project is currently only distributed as source code, and uses Poetry to manage
dependencies, so if you want to reproduce the results you should first [install
Poetry](https://python-poetry.org/docs/).

Then you should clone or download this repository to some location (e.g.
`~/df-analyze.py`), and `cd` to that directory. Then you can install dependencies and
activate the venv with:

```shell
poetry install  # setup environment
poetry shell  # activate venv
```

If you have git and are on a Unix-based system, just:

```sh
cd <wherever>
git clone https://github.com/stfxecutables/df-analyze.git
cd df-analyze
poetry install
poetry shell
```

# Running the Script

To see how to use the tool, and after having gone through the setup, do:

```shell
python df-analyze.py --help
```

This will print detailed usage information, which is duplicated below in the [Usage section](#usage).

# Usage

```
Currently, `classifier` can be one of `svm`, `rf`, `mlp`, `bag` or `dtree`. If the `--step-up`
argument is omitted, instead analyses for other feature selection methods are used.usage: df-analyze.py [-h] --df DF [--target TARGET]
                     [--mode {classify,regress}]
                     [--classifiers {rf,svm,dtree,mlp,bag} [{rf,svm,dtree,mlp,bag} ...]]
                     [--regressors {linear,rf,svm,adaboost,gboost,mlp,knn} [{linear,rf,svm,adaboost,gboost,mlp,knn} ...]]
                     [--feat-select {step-up,step-down,pca,kpca,d,auc,pearson,none} [{step-up,step-down,pca,kpca,d,auc,pearson,none} ...]]
                     [--feat-clean {correlated,constant,lowinfo} [{correlated,constant,lowinfo} ...]]
                     [--drop-nan {all,rows,cols,none}] [--n-feat N_FEAT]
                     [--htune]
                     [--htune-val {holdout,kfold,k-fold,loocv,mc,none}]
                     [--htune-val-size HTUNE_VAL_SIZE]
                     [--htune-trials HTUNE_TRIALS]
                     [--test-val {holdout,kfold,k-fold,loocv,mc,none}]
                     [--test-val-sizes TEST_VAL_SIZES [TEST_VAL_SIZES ...]]
                     --outdir OUTDIR [--verbosity VERBOSITY]

optional arguments:
  -h, --help            show this help message and exit
  --df DF
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

  --target TARGET, -y TARGET

                        The location of the target variable for either regression or classification.

                        If a string, then `--df` must be a Pandas `DataFrame` and the string passed in
                        here specifies the name of the column holding the targer variable.

                        If an integer, and `--df` is a NumPy array only, specifies the column index.

  --mode {classify,regress}, -m {classify,regress}

                        If "classify", do classification. If "regress", do regression.

  --classifiers {rf,svm,dtree,mlp,bag} [{rf,svm,dtree,mlp,bag} ...], -C {rf,svm,dtree,mlp,bag} [{rf,svm,dtree,mlp,bag} ...]

                        The list of classifiers to use when comparing classification performance.
                        Can be a list of elements from: bag dtree mlp rf svm.

  --regressors {linear,rf,svm,adaboost,gboost,mlp,knn} [{linear,rf,svm,adaboost,gboost,mlp,knn} ...], -R {linear,rf,svm,adaboost,gboost,mlp,knn} [{linear,rf,svm,adaboost,gboost,mlp,knn} ...]

                        The list of regressors to use when comparing regression model performance.
                        Can be a list of elements from: adaboost gboost knn linear mlp rf svm.

  --feat-select {step-up,step-down,pca,kpca,d,auc,pearson,none} [{step-up,step-down,pca,kpca,d,auc,pearson,none} ...], -F {step-up,step-down,pca,kpca,d,auc,pearson,none} [{step-up,step-down,pca,kpca,d,auc,pearson,none} ...]

                        The feature selection methods to use. Available options are:

                          auc:        Select features with largest AUC values relative to the two
                                      classes (classification only).

                          d:          Select features with largest Cohen's d values relative to the two
                                      classes (classification only).

                          kpca:       Generate features by using largest components of kernel PCA.

                          pca:        Generate features by using largest components from a PCA.

                          pearson:    Select features with largest Pearson correlations with target.

                          step-up:    Use step-up (forward) feature selection. Costly.

                          step-up:    Use step-down (backward) feature selection. Also costly.

                        NOTE: Feature selection currently uses the full data provided in the `--df`
                        argument to `df-analyze.py`. Thus, if you take the final reported test results
                        following feature selection and hyperparamter tuning as truly cross-validated
                        or heldout test results, you are in fact double-dipping and reporting biased
                        performance. To properly test the discovered features and optimal estimators,
                        you should have held-out test data that never gets passed to `df-analyze`.

  --feat-clean {correlated,constant,lowinfo} [{correlated,constant,lowinfo} ...], -f {correlated,constant,lowinfo} [{correlated,constant,lowinfo} ...]

                        If specified, which feature cleaning methods to use prior to feature selection.
                        Makes use of the featuretools library (featuretools.com). Options are:

                          correlated: remove highly correlated features using featuretools.
                          constant:   remove constant (zero-variance) features. Default.
                          lowinfo:    remove "low information" features via featuretools.

  --drop-nan {all,rows,cols,none}, -d {all,rows,cols,none}

                        How to drop NaN values. Uses Pandas options.

                          none:       Do not remove. Will cause errors for most algorithms. Default.
                          all:        Remove the sample and feature both (row and column) for any NaN.
                          rows:       Drop samples (rows) that contain one or more NaN values.
                          cols:       Drop features (columns) that contain one or more NaN values.

  --n-feat N_FEAT
                        Number of features to select using method specified by --feat-select. NOTE:
                        specifying values greater than e.g. 10-50 with --feat-select=step-up and slower
                        algorithms can easily result in compute times of many hours.

  --htune
                        If provided, use Optuna TPESampler to attempt to optimize classifier performance
                        prior to fitting and evaluating.

  --htune-val {holdout,kfold,k-fold,loocv,mc,none}, -H {holdout,kfold,k-fold,loocv,mc,none}

                        If hyperparamater tuning using `--htune` option, specifies the validation style
                        to use internally for each Optuna trial. Number of trials is specified by
                        `--htune-trials`, so the number of estimator fits interacts with that values
                        and `--htune-val-size`, which has a different meaning depending on validation
                        type chosen. Available options:

                          holdout:    Create a single holdout set and use to validate all Optuna trials.
                                      The float value in (0, 1) specified in `--htune-val-size` sets the
                                      percentage of samples to use for this holdout / test set.

                          kfold:      Use k-fold cross validation to compute performance for each Optuna
                                      trial. The value for `k` is specified by `--htune-val-size`.

                          loocv:      Use Leave-One-Out cross validation. `--htune-val-size` is ignored
                                      in this case.

                          mc:         Use "Monte-Carlo Cross Validation", e.g. generate multiple random
                                      train / test splits. Currently generates 20 splits at 80%/20%
                                      train/test. `--htune-val-size` is ignored in this case.

                          none:       Just fit the full data (e.g. validate on fitting / training data).
                                      Fast but highly biased. `--htune-val-size` is ignored in this case.

  --htune-val-size HTUNE_VAL_SIZE

                        See documentation for `--htune-val` (directly above if using `--help`).

  --htune-trials HTUNE_TRIALS

                        Specifies number of trials in Optuna study, and for each estimator and feature
                        selection method. E.g. fitting two estimators using three feature selection
                        methods with `--htune-trials=100` will results in 2 x 3 x 100 = 600 trials. If
                        also using e.g. the default 3-fold validation for `--htune-val-sizes`, then the
                        total number of estimator fits from tuning will be 600 x 3.

                        NOTE: if you can afford it, it is strongly recommended to set this value to a
                        minimum of 100 (default), or 50 if your budget is constrained. Lower values
                        often will fail to find good fits, given the wide range on hyperparameters
                        needed to make this tool generally useful.

  --test-val {holdout,kfold,k-fold,loocv,mc,none}, -T {holdout,kfold,k-fold,loocv,mc,none}

                        Specify which validation method to use for testing. Same behavour as for
                        `--htune-val` argument (see above).

  --test-val-sizes TEST_VAL_SIZES [TEST_VAL_SIZES ...]

                        Specify sizes of test validation sets. Same behavour as for `--htune-val-sizes`
                        argument (see above), except that multiple sizes can be specified. E.g.

                          python df-analyze.py <omitted> --test-val=kfold --test-val-sizes 5 10 20

                        will efficiently re-use the same trained model and evaluate 5-, 10-, and 20-fold
                        k-fold estimates of performance, and include these all in the final results.

  --outdir OUTDIR
                        Specifies location of all results, as well as cache files for slow computations
                        (e.g.  stepwise feature selection).

  --verbosity VERBOSITY

                        Controls amount of output to stdout and stderr. Options:

                          0:         ERROR: Minimal output and errors only
                          1:         INFO: Logging for each Optuna trial and various interim results.
                          2:         DEBUG: Currently unimplemented.


USAGE EXAMPLE (assumes you have run `poetry shell`):

    python df-analyze.py \
        --df="weather_data.json" \
        --target='temperature' \
        --mode=regress \
        --regressors=svm linear \
        --drop-nan=rows \
        --feat-clean=constant \
        --feat-select=pca pearson \
        --n-feat=5 \
        --htune \
        --test-val=kfold \
        --test-val-size=5 \
        --outdir='./results'
```
# Basic Example

Assumes you have run `poetry shell` or otherwise activated the .venv installed by Poetry.

```sh
python df-analyze.py \
    --df="weather_data.json" \
    --target="temperature" \
    --mode=regress \
    --regressors svm linear \
    --drop-nan=rows \
    --feat-clean=constant \
    --feat-select none pca pearson \
    --n-feat=5 \
    --htune \
    --htune-val=kfold \
    --htune-val-size=3 \
    --test-val=kfold \
    --test-val-size=5 \
    --outdir='./results' \
    --verbosity=1
```