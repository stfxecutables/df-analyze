from src._constants import (
    CLASSIFIERS,
    DEFAULT_OUTDIR,
    REGRESSORS,
)

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

SHEET_HELP_STR = """
The path to the formatted spreadsheet to analyze.

Currently only spreadsheets saved as either `.xlsx` or `.csv` are supported.

If your data is saved as a Pandas `DataFrame`, it must have shape
`(n_samples, n_features)` or `(n_samples, n_features + 1)`. The name of the
column holding the target variable (or feature) can be specified by the
`--target` / `-y` argument, but is "target" by default if such a column name
exists, or the last column if it does not.

If your data is in a NumPy array, the array must have the shape
`(n_samples, n_features + 1)` where the last column is the target for either
classification or prediction.

"""

SEP_HELP_STR = """
Separator used in .csv files. Default ",".
"""

TARGET_HELP_STR = """
The location of the target variable for either regression or classification.

If a string, then `--df` must be a Pandas `DataFrame` and the string passed in
here specifies the name of the column holding the targer variable.

If an integer, and `--df` is a NumPy array only, specifies the column index.

"""

CATEGORICAL_HELP_STR = """
A string or list of strings, e.g.

    --categoricals sex gender ancestry education

that specifies which features will be treated as categorical regardless of the
number of levels or format of the data. If during data cleaning categorical
variables are detected that are NOT specified by the user, a warning will be
raised.
"""

ORDINAL_HELP_STR = """
A string or list of strings, e.g.

    --ordinals star_rating number_of_purchases times_signed_in

that specifies which features will be treated as ordinal regardless of the
number of levels or format of the data. If during data cleaning categorical
variables are detected that are NOT specified by the user, a warning will be
raised. If the values of the specified variables cannot be interpreted as
integers, then df-analyze will exit with an error.
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
Can be a list of elements from: {' '.join(sorted(CLASSIFIERS))}.

"""

REG_HELP_STR = f"""
The list of regressors to use when comparing regression model performance.
Can be a list of elements from: {' '.join(sorted(REGRESSORS))}.

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

  step-up:    Use step-up (forward) feature selection. Costly.

  step-up:    Use step-down (backward) feature selection. Also costly.

NOTE: Feature selection currently uses the full data provided in the `--df`
argument to `df-analyze.py`. Thus, if you take the final reported test results
following feature selection and hyperparamter tuning as truly cross-validated
or heldout test results, you are in fact double-dipping and reporting biased
performance. To properly test the discovered features and optimal estimators,
you should have held-out test data that never gets passed to `df-analyze`.

"""

FEAT_CLEAN_HELP = """
If specified, which feature cleaning methods to use prior to feature selection.
Makes use of the featuretools library (featuretools.com). Options are:

  correlated: remove highly correlated features using featuretools.
  constant:   remove constant (zero-variance) features. Default.
  lowinfo:    remove "low information" features via featuretools.

"""

NAN_HELP = """
How to handle NaN values in non-categorical features. Categorical features
are handled by representing the NaN value as another category level (class),
i.e. one extra one-hot column is created for each categorical feature with a
NaN value.

  drop:      Attempt to remove all non-categorical NaN values. Note this could
             remove all data if a lot of values are missing, which will cause
             errors.
  mean:      Replace all NaN values with the feature mean value.
  median:    Replace all NaN values with the feature median value.
  impute:    Use scikit-learn experimental IterativeImputer to attempt to
             predictively fill NaN values based on other feature values. May
             be computationally demanding on larger datasets.
"""

N_FEAT_HELP = """
Number of features to select using method specified by --feat-select. NOTE:
specifying values greater than e.g. 10-50 with --feat-select=step-up and slower
algorithms can easily result in compute times of many hours.

"""

HTUNE_HELP = """
If provided, use Optuna TPESampler to attempt to optimize classifier performance
prior to fitting and evaluating.

"""

HTUNEVAL_HELP_STR = """
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
              train / test splits, i.e. equivalent to using all the splits
              generated by `sklearn.model_selection.StratifiedShuffleSplit`
              Currently generates 20 splits at 80%%/20%% train/test.
              `--htune-val-size` specifies the size of the test split in this
              case.

  none:       Just fit the full data (e.g. validate on fitting / training data).
              Fast but highly biased. `--htune-val-size` is ignored in this case.

"""

HTUNE_VALSIZE_HELP = """
See documentation for `--htune-val` (directly above if using `--help`). The
meaning of this argument depends on the choice of `--htune-val`:

  holdout:    A float in (0, 1) specifies the proportion of samples from the
              input spreadsheet or table to set aside for evaluating during
              hyperparameter tuning.

              An integer specifies the number of samples to set aside for
              testing.

  kfold:      An integer being one of 3, 5, 10, or 20 specifies the number of
              folds.

  loocv:      Ignored.

  mc:         A float in (0, 1) specifies the proportion of samples from the
              input spreadsheet or table to set aside for evaluating during
              each Monte-Carlo repeat evaluation.

              An integer specifies the number of samples to set aside for
              each repeat.

"""

MC_REPEATS_HELP = """
Ignored unless using Monte-Carlo style cross validation via `--htune-val mc`.
Otherwise, specifies the number of random subsets of proportion
`--htune-val-size` on which to validate the data. Default 10.
"""

HTUNE_TRIALS_HELP = """
Specifies number of trials in Optuna study, and for each estimator and feature
selection method. E.g. fitting two estimators using three feature selection
methods with `--htune-trials=100` will results in 2 x 3 x 100 = 600 trials. If
also using e.g. the default 3-fold validation for `--htune-val-sizes`, then the
total number of estimator fits from tuning will be 600 x 3.

NOTE: if you can afford it, it is strongly recommended to set this value to a
minimum of 100 (default), or 50 if your budget is constrained. Lower values
often will fail to find good fits, given the wide range on hyperparameters
needed to make this tool generally useful.

"""

TEST_VAL_HELP = """
Specify which validation method to use for testing. Same behavour as for
`--htune-val` argument (see above).

"""

TEST_VALSIZES_HELP = """
Specify sizes of test validation sets. Same behavour as for `--htune-val-sizes`
argument (see above), except that multiple sizes can be specified. E.g.

  python df-analyze.py <omitted> --test-val=kfold --test-val-sizes 5 10 20

will efficiently re-use the same trained model and evaluate 5-, 10-, and
20-fold k-fold estimates of performance, and include these all in the final
results.

The meaning of this argument depends on the choice of `--test-val`:

  holdout:    A float in (0, 1) specifies the proportion of samples from the
              input spreadsheet or table to set aside for evaluating during
              hyperparameter tuning.

              An integer specifies the number of samples to set aside for
              testing.

  kfold:      An integer being one of 3, 5, 10, or 20 specifies the number of
              folds.

  loocv:      Ignored.

  mc:         A float in (0, 1) specifies the proportion of samples from the
              input spreadsheet or table to set aside for evaluating during
              each Monte-Carlo repeat evaluation.

              An integer specifies the number of samples to set aside for
              each repeat.
"""

OUTDIR_HELP = f"""
Specifies location of all results, as well as cache files for slow computations
(e.g. stepwise feature selection). If unspecified, will default to
{DEFAULT_OUTDIR}

"""

VERBOSITY_HELP = """
Controls amount of output to stdout and stderr. Options:

  0:         ERROR: Minimal output and errors only
  1:         INFO: Logging for each Optuna trial and various interim results.
  2:         DEBUG: Currently unimplemented.

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

USAGE_STRING = """

    The df-analyze program can be used in one of two modes: CLI mode, and
    spreadsheet mode. In spreadsheet mode, df-analyze options are specified in
    a special format at the top of a spreadsheet or .csv file, and spreadsheet
    columns are given specific names to identify targets, continuous features,
    and categorical features. In spreadsheet mode, only a single argument needs
    to be passed, which is the path to the df-analyze formatted spreadsheet:

        python df-analyze.py --spreadsheet my_formatted_sheet.xlsx


"""
