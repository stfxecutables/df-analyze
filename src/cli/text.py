from pathlib import Path

from src._constants import (
    CLASSIFIERS,
    REGRESSORS,
)
from src.enumerables import ClsScore, DfAnalyzeClassifier, DfAnalyzeRegressor, RegScore

DF_HELP_STR = """
The dataframe to analyze.

Currently only tables saved as either `.xlsx`, `.json` or `.csv`, or NumPy
`ndarray`s saved as "<filename>.npy" are supported, but a file exported by
Pandas `DataFrame.to_*` method is preferred.

If your data is saved as a Pandas `DataFrame`, it must have shape
`(n_samples, n_features)` or `(n_samples, n_features + 1)`. The name of the
column holding the target variable (or feature) can be specified by the
`--target` / `-y` argument, but is "target" by default if such a column name
exists, or the last column if it does not.

"""

SHEET_HELP_STR = """
The path to the formatted spreadsheet to analyze.

Currently only spreadsheets saved as either `.xlsx` or `.csv` are supported.

If your data is saved as a Pandas `DataFrame`, it must have shape
`(n_samples, n_features)` or `(n_samples, n_features + 1)`. The name of the
column holding the target variable (or feature) can be specified by the
`--target` / `-y` argument, but is "target" by default if such a column name
exists, or the last column if it does not.

"""

SEP_HELP_STR = """
Separator used in .csv files. Default ",".
"""

TARGET_HELP_STR = """
The (string) name of the target variable for either regression or
classification.

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

DROP_HELP_STR = """
A string or list of strings, e.g.

    --drops subject_id location_id useless_feature1 useless_feat2

that specifies which features will be removed from the data and not considered
for any inspection, description or univariate analysis, and which will not be
included in any feature selection, model tuning, or final predictive models.

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
Can be a list of elements from: {' '.join(sorted([x.value for x in DfAnalyzeClassifier]))}.

"""

REG_HELP_STR = f"""
The list of regressors to use when comparing regression model performance.
Can be a list of elements from: {' '.join(sorted([x.value for x in DfAnalyzeRegressor]))}.

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

MODEL_SELECT_HELP = """
Methods of model-based feature selection methods to use. Available options are:

  embed:      Select using an embedded method, i.e. a method where the model
              produces values for each feature that can be interpreted as
              feature importances. Which model is used is determined by
              `--embed-model`.

  wrap:       Select using a wrapper method, i.e. a method which uses ("wraps")
              a specific model, and then optimizes the feature set via some
              alternation of model evaluations and feature-space search /
              navigation strategy.

  none:       Do not select features using any model.

"""

WRAP_SELECT_HELP = """
Wrapper-based feature selection method, i.e. method/optimizer to use to
search feature-set space during wrapper-based feature selection. Currently
only (recursive) step-down and step-up methods are supported, but future
versions may support random subset search, LIPO
(http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html,
https://arxiv.org/abs/1703.02628) and evolutionary algorithms such as
particle-swarm optimization and genetic algorithms (since feature selection
is just a black-box optimization problem where the search space is a set of
feature sets).

Model to use in wrapper-based feature selection. Available options are:

  stepup:     Start with the empty feature set, and greedily add the feature
              that most improves prediction of the target variable. Also called
              forward feature selection.

  stepdown:   Start with the full feature set, and remove the feature
              that most improves (or least decreases) prediction of the target
              variable. Also called backward / recursive feature slection or
              elimination.

"""

WRAP_SELECT_MODEL_HELP = """
Model to use during wrapper-based feature selection. Available options are:

  linear:     For classification tasks, logistic regression, and for regression
              tasks, linear regression.

  lgbm:       Use a LightGBM gradient-boosted decision tree model.

"""

EMBED_SELECT_MODEL_HELP = """
Model to use for embedded feature selection. Supported models are:

  linear      Tuned SGDRegressor or SGDClassifier, in both cases with L1
              regularization.

  lgbm:       LightGBM regressor or classifier, depending on task.

"""

SELECT_TUNE_ROUNDS_HELP = """
"""

FEAT_CLEAN_HELP = """
If specified, which feature cleaning methods to use prior to feature selection.
Makes use of the featuretools library (featuretools.com). Options are:

  correlated: remove highly correlated features using featuretools.

  constant:   remove constant (zero-variance) features. Default.

  lowinfo:    remove "low information" features via featuretools.

"""

SELECT_TUNE_ROUNDS_HELP = """
If not the default of zero, the number of tuning rounds to do before evaluating
each feature set during wrapper-based feature selection.

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

N_FEAT_WRAPPER_HELP = """
Number of features to select during wrapper-based feature selection. Note
that specifying values greater than e.g. 10-20 with slower algorithms (e.g.
LightGBM) and for data with a large number of features (e.g. over 50) can
easily result in compute times of many hours.

"""

N_FEAT_NOTE = (
    "Note only two of two of the three options:`--n-filter-total`, "
    "`--n-filter-cont`, and  `--n-filter-cat` may be specified at once, "
    "otherwise the `--n-filter-total` argument will be ignored.\n\n"
)

N_FEAT_TOTAL_FILTER_HELP = f"""
Number or percentage (as a value in [0, 1]) of total features of any kind
(categorical or continuous) to select via filter-based feature selection.
{N_FEAT_NOTE}

"""

N_FEAT_CONT_FILTER_HELP = f"""
Number or percentage (as a value in [0, 1]) of continuous features to select
via filter-based feature selection. {N_FEAT_NOTE}

"""

N_FEAT_CAT_FILTER_HELP = f"""
Number or percentage (as a value in [0, 1]) of categorical features to select
via filter-based feature selection. {N_FEAT_NOTE}

"""


FILTER_METHOD_HELP = (
    "Method(s) to use for filter selection."
    "\n\n"
    "Method 'relief' is the most sophisticated and can detect interactions "
    "among pairs of features without dramatic compute costs (see "
    "https://www.sciencedirect.com/science/article/pii/S1532046418301400 or "
    "https://doi.org/10.1016/j.jbi.2018.07.014 for details and overview). "
    "This is in contrast to the 'assoc' and 'pred' methods (below) which do "
    "not detect any feature interactions. "
    "\n\n"
    "Method 'assoc' is the fastest and is based on a measure of association "
    "between the feature and the target variable, where the measure of "
    "association is appropriate based on the cardinality (e.g. categorical vs. "
    "continuous) of the feature and target. However, because association need "
    "not imply generalizable predictive utility (and because the absence of an "
    "association does not imply an absence of predictive utility), it is "
    "possible that this method selects features that generalize poorly for "
    "prediction tasks. "
    "\n\n"
    "Method 'pred' is based on the k-fold univariate predictive performance of "
    "each feature on the target variable, where the estimator is a lightly "
    "tuned sklearn.linear_model.SGDClassifier or "
    "sklearn.linear_model.SGDRegressor, depending on the task. Computing these "
    "univariate predictive performances is quite expensive, but because of the "
    "internal k-fold validation used, these predictive performance metrics "
    "directly asses the potential predictive utility of each feature.\n\n"
)

ASSOC_SELECT_CONT_CLS_STATS = """
Type of association to use for selecting continuous features when the task or
target is classification / categorical.

"""

ASSOC_SELECT_CAT_CLS_STATS = """
Type of association to use for selecting categorical features when the task or
target is classification / categorical.

"""

ASSOC_SELECT_CONT_REG_STATS = """
Type of association to use for selecting continuous features when the task or
target is regression / continuous.

"""

ASSOC_SELECT_CAT_REG_STATS = """
Type of association to use for selecting categorical features when the task or
target is regression / continuous.

"""

REG_OPTIONS = "".join([f"  {f'{score.value}:': <11}{score.longname()}\n\n" for score in RegScore])
PRED_SELECT_REG_SCORE = f"""
Regression score to use for filter-based selection of features. Options:\n
{REG_OPTIONS}

"""

CLS_OPTIONS = "".join([f"  {f'{score.value}:': <11}{score.longname()}\n\n" for score in ClsScore])
PRED_SELECT_CLS_SCORE = f"""
Classification score to use for filter-based selection of features. Options:\n
{CLS_OPTIONS}

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
Specifies location of all results, as well as cache files for slow
computations (e.g. stepwise feature selection). If unspecified, will attempt
to default to a number of common locations ({Path.home().resolve()}, the
current working directory {Path.cwd().resolve()}, or a temporary directory).

"""

VERBOSITY_HELP = """
Controls amount of output to stdout and stderr. Options:

  0:         ERROR: Minimal output and errors only
  1:         INFO: Logging for each Optuna trial and various interim results.
  2:         DEBUG: Currently unimplemented.

"""

EXPLODE_HELP = """
If this flag is present, silence the warnings about large increases in the
number of features due to one-hot encoding of categoricals.

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
