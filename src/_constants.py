from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATADIR = ROOT / "data"
TESTDATA = DATADIR / "testing"
TEMPLATES = DATADIR / "templates"
TEST_RESULTS = TESTDATA / "__TEST_RESULTS__"
FULL_RESULTS = TESTDATA / "__FULL_RESULTS__"

# Testing
SIMPLE_XLSX = TEMPLATES / "testing/spreadsheet.xlsx"
SIMPLE_CSV = TEMPLATES / "testing/spreadsheet.csv"
SIMPLE_CSV2 = TEMPLATES / "testing/spreadsheet2.csv"
SIMPLE_ODS = TEMPLATES / "testing/spreadsheet.ods"
COMPLEX_XLSX = TEMPLATES / "testing/spreadsheet_complex.xlsx"
COMPLEX_XLSX2 = TEMPLATES / "testing/spreadsheet_complex2.xlsx"

MUSHROOM_DATA = TESTDATA / "classification/mushrooms/mushrooms.parquet"
MUSHROOM_TYPES = TESTDATA / "classification/mushrooms/types.csv"
ELDER_DATA = TESTDATA / "classification/elder/measurements.csv"
ELDER_TYPES = TESTDATA / "classification/elder/types.csv"


DATAFILE = DATADIR / "MCICFreeSurfer.mat"
DATA_JSON = DATAFILE.parent / "mcic.json"
CLEAN_JSON = DATAFILE.parent / "mcic_clean.json"
UNCORRELATED = DATADIR / "mcic_uncorrelated_cols.json"

CLASSIFIERS = ["rf", "svm", "dtree", "mlp", "bag", "dummy", "lgb"]
REGRESSORS = ["linear", "rf", "svm", "adaboost", "gboost", "mlp", "knn", "lgb"]
DIMENSION_REDUCTION = ["pca", "kpca", "umap"]
WRAPPER_METHODS = ["step-up", "step-down"]
UNIVARIATE_FILTER_METHODS = ["d", "auc", "pearson", "t-test", "u-test", "chi", "info"]
"""
info   - information gain / mutual info
chi    - chi-squared
u-test - Mann-Whitney U

See https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
"""
MULTIVARIATE_FILTER_METHODS = ["fcbf", "mrmr", "relief"]
"""
FCBF - fast correlation-based filter\n
mRMR - minimal-redundancy-maximal-relevance\n
CMIM - conditional mutual information maximization\n
relief - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6299836/
         we likely want MultiSURF (not MultiSURF*), available in
         https://github.com/EpistasisLab/scikit-rebate

See:
https://en.wikipedia.org/wiki/Feature_selection
https://en.wikipedia.org/wiki/Relief_(feature_selection)


https://www.frontiersin.org/articles/10.3389/fbinf.2022.927312/full
Another popular family of filter algorithms is the Relief-based algorithm (RBA)
family (e.g., Relief (Kira and Rendell, 1992), ReliefF (Kononenko, 1994), TURF
(Moore and White, 2007), SURF (Greene et al., 2009), SURF* (Greene et al.,
2010), MultiSURF (Urbanowicz et al., 2018a), MultiSURF* (Granizo-Mackenzie and
Moore, 2013), etc…). Relief does not exhaustively search for feature
interactions. Instead, it scores the importance of a feature according to how
well the feature’s value distinguishes samples that are similar to each other
(e.g., similar genotype) but belong to different classes (e.g., case and
control). Notably, RBAs can detect pair-wise feature interactions, some RBAs
(e.g., ReliefF, MultiSURF) can even detect higher order (>2 way) interactions
(Urbanowicz et al., 2018a).
"""
FEATURE_CLEANINGS = ["correlated", "constant", "lowinfo"]
CLEANINGS_SHORT = ["corr", "const", "info"]
HTUNE_VAL_METHODS = ["holdout", "kfold", "k-fold", "loocv", "mc", "none"]


class __Sentinel:
    pass


SENTINEL = __Sentinel()

VAL_SIZE = 0.20
SEED = 69

DEFAULT_N_STEPWISE_SELECT = 10
"""Number of features to select during stepwise selection"""
MAX_STEPWISE_SELECTION_N_FEATURES = 20
"""Number of features to warn user about feature selection problems"""
MAX_PERF_N_FEATURES = 500
"""Number of features to warn users about general performance problems"""

N_CAT_LEVEL_MIN = 20
"""
Minimum required number of samples for level of a categorical variable to be
considered useful in 5-fold analyses.

Notes
-----

For each level of categorical variable to be predictively useful, there must
be enough samples to be statistically meaningful (or to allow some reasonable
generalization) in each fold used for fitting or analyses. Roughly, this
means each fold needs to see at least 10-20 samples of each level (depending
on how strongly / cleanly the level relates to other features - ultimately
this is just a heuristic). Assuming k-fold is used for validation, then this
means about (1 - 1/k) times 10-20 samples per categorical level would a
reasonable default minimum requirement one might use for culling categorical
levels. Under the typical assumption of k=5, this means we require useful /
reliable categorical levels to have 8-16 samples each.

Inflation is to categorical variables as noise is to continuous ones.
"""

N_TARG_LEVEL_MIN = 30
"""
Minimum required number of samples for level of a categorical target variable
to be considered useful in 5-fold analyses.

Notes
-----
Suppose the smallest target class or level has N samples. With stratified
k-fold, a k-fold training set will have floor((1 - 1/k) * N) samples of that
level. For k in {3, 5}, this is either floor(2N/3) or floor(0.8*N) samples.
Now if we use k-fold again internally (e.g. nested k-fold), with internal
k=m, then there are again floor( (1 - 1/m)(1 - 1/k) * N ) samples of that
level, i.e. for (k, m) in {(3, 3), (3, 5), (5, 5)} we have floor(0.444 * N),
floor(0.5333 * N), or floor(0.64*N) = floor(16N/25) samples in the training
set. In the test set the numbers are just floor(N/9), floor(N/15),
floor(N/25).

So far we use only 5-fold, so to guarantee one sample of each level in an
internal fold, we need floor(N/25) > 1 ==> N > 25.
"""

UNIVARIATE_PRED_MAX_N_SAMPLES = 1500
"""Maximum number of samples to use in univariate predictive analyses"""

# from https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
NAN_STRINGS = [
    "",
    "-1.#IND",
    "-1.#QNAN",
    "-nan",
    "-NaN",
    "#N/A N/A",
    "#N/A",
    "#NA",
    "<NA>",
    "1.#IND",
    "1.#QNAN",
    "n/a",
    "N/A",
    "NA",
    "nan",
    "NaN",
    "Nan",
    "None",
    "null",
    "NULL",
    "-",
    "_",
]

N_EMBED_DEFAULT = 20
N_WRAPPER_DEFAULT = 20
N_FILTER_CONT_DEFAULT = 20
N_FILTER_CAT_DEFAULT = 10
N_FILTER_TOTAL_DEFAULT = 30
P_FILTER_CONT_DEFAULT = 0.10
P_FILTER_CAT_DEFAULT = 0.20
P_FILTER_TOTAL_DEFAULT = 0.25
