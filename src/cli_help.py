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

TARGET_HELP_STR = """
The location of the target variable for either regression or classification.

If a string, then `--df` must be a Pandas `DataFrame` and the string passed in
here specifies the name of the column holding the targer variable.

If an integer, and `--df` is a NumPy array only, specifies the column index.

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
Can be a list of elements from {sorted(CLASSIFIERS)}.

"""

REG_HELP_STR = f"""
The list of regressors to use when comparing regression model performance.
Can be a list of elements from {sorted(REGRESSORS)}.

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

  step-up:    Use step-up features selection. Costly.

"""

FEAT_CLEAN_HELP = """
If specified, which feature cleaning methods to use prior to feature selection.
Makes use of the featuretools library (featuretools.com). Options are:

  correlated: remove highly correlated features using featuretools.
  constant:   remove constant (zero-variance) features. Default.
  lowinfo:    remove "low information" features via featuretools.

"""

NAN_HELP = """
How to drop NaN values. Uses Pandas options.

  none:       Do not remove. Will cause errors for most algorithms. Default.
  all:        Remove the sample and feature both (row and column) for any NaN.
  rows:       Drop samples (rows) that contain one or more NaN values.
  cols:       Drop features (columns) that contain one or more NaN values.

"""

N_FEAT_HELP = """
Number of features to select using method specified by --feat-select. NOTE:
specifying values greater than e.g. 10-50 with --feat-select=step-up and slower
algorithms can easily result in compute times of many hours.

"""

HTUNE_HELP = """

"""

HTUNEVAL_HELP_STR = """
If an

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
