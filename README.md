[![DOI](https://zenodo.org/badge/364694785.svg)](https://zenodo.org/badge/latestdoi/364694785)

<!-- omit from toc -->
# Contents

- [What it Does](#what-it-does)
  - [Analysis Pipeline](#analysis-pipeline)
  - [Philosophy](#philosophy)
    - [Recursive / Wrapper Feature Selection and Tuning](#recursive--wrapper-feature-selection-and-tuning)
- [Installation](#installation)
  - [Install Poetry](#install-poetry)
    - [Installing a Compatible Python Version](#installing-a-compatible-python-version)
  - [`pip` Fallback in Case of Poetry Issues](#pip-fallback-in-case-of-poetry-issues)
- [Usage](#usage)
  - [Examples](#examples)
    - [Using Builtin Data](#using-builtin-data)
    - [Using a `df-analyze`-formatted Spreadsheet](#using-a-df-analyze-formatted-spreadsheet)
      - [Overriding Spreadsheet Options](#overriding-spreadsheet-options)
- [Usage on Compute Canada / Digital Research Alliance of Canada / Slurm HPC Clusters](#usage-on-compute-canada--digital-research-alliance-of-canada--slurm-hpc-clusters)
  - [Singularity Container](#singularity-container)
    - [Building](#building)
    - [Running](#running)
  - [Parallelization Options](#parallelization-options)
- [Currently Implemented Program Features and Analyses](#currently-implemented-program-features-and-analyses)
  - [Completed Features](#completed-features)
    - [Single Spreadsheet for Configuration and Data](#single-spreadsheet-for-configuration-and-data)
    - [Automated Data Preproccesing](#automated-data-preproccesing)
    - [Feature Descriptive Statisics](#feature-descriptive-statisics)
    - [Univariate Feature-Target Associations](#univariate-feature-target-associations)
    - [Univariate Prediction Metrics for each Feature-Target Pair](#univariate-prediction-metrics-for-each-feature-target-pair)
  - [In progress / Partially Completed:](#in-progress--partially-completed)
  - [To Do (highest priority first):](#to-do-highest-priority-first)
  - [May Not Implement:](#may-not-implement)

# What it Does

TODO

`df-analyze` automates some common, naive machine-learning approaches

## Analysis Pipeline

TODO

**Data Preparation**

1. Data Loading
   1. Type Conversions
1. Data Cleaning
   1. NaN removal / interpolation
   1. OneHot Encoding
1. Data Preprocessing
   1. Normalization
   1. Outlier removal / clipping

**Feature Selection**

1. Remove junk features
   1. Remove constant features
   1. Remove highly-correlated features
1. Use filter methods
   1. Remove features with minimal univariate relation to target
   1. Keep features with largest filter

**Data Splitting**

1. Split data $X$ into $X_\text{train}$, $X_\text{test}$, with $X_\text{test}$


**Recursive / Wrapper Feature Selection**

1. Step-up selection using $X_\text{train}$

**Hyperparameter Selection**

1. Bayesian (Optuna) with internal 3-fold validation on $X_\text{train}$

**Final Validation**

1. Final k-fold of model tuned and trained on selected features from $X_\text{train}$
1. Final evaluation of trained model on $X_\text{test}$


## Philosophy

1. Data *preparation* is not to be optimized
   - i.e. while one might re-run
1. Proper validation of a feature selection method requires holdout data NOT to
   be used during feature selection, i.e. requires preventing [double
   dipping](https://www.nature.com/articles/nn.2303).

### Recursive / Wrapper Feature Selection and Tuning

- too expensive to do together (e.g. do hyperparameter tuning on each potential
  feature subset)
- in general a highly challenging bilevel optimization problem
- to keep computationally tractable, must choose between:
  1. using a model with "default" hyperparameters for the wrapper selection process
  1. tuning on all features, then using the tuned model for wrapper selection
- neither choice above is likely to be optimal

# Installation

## Install Poetry

This project uses Poetry to manage dependencies, so if you want to reproduce
the results you should [install Poetry](https://python-poetry.org/docs/). Then
you can install dependencies and activate the venv with:

```shell
poetry install  # setup environment
poetry shell  # activate venv
```

The project currently requires python 3.9 or python 3.10 to be installed.

### Installing a Compatible Python Version

If you don't have an updated python version, you might want to look into
[`pyenv`](https://github.com/pyenv/pyenv), which makes it very easy to install
and switch between multiple python versions. For example, if you install
`pyenv`, then you can install a version of python compatible with `df-analyze`
by simply running

```shell
pyenv install 3.10.12
```

Then, running the poetry install commands above *should* automatically find
this Python version.


## `pip` Fallback in Case of Poetry Issues

If there is an issue with the poetry install process, as a fallback, you can
try creating a new virtual environment and installing the versions listed in
the `requirements.txt` file:

```shell
python3 -m venv .venv                      # create a virtual environment
source .venv/bin/activate                  # activate environment
python -m pip install -r requirements.txt  # install requirements
```

Note the above assumes that the `python3` command is a Python 3.9 or 3.10
version, which you could check by running `python3 --version` beforehand.


# Usage

To see how to run `df-analyze` do:

```shell
python df-analyze.py --help
```

which will provide a complete description of all options.


## Examples

Run a classification analysis on the data in `small_classifier_data.json`:

### Using Builtin Data

```bash
python df-analyze.py \
    --df data/small_classifier_data.json \
    --mode classify \
    --classifiers rf svm dtree \
    --feat-select pca pearson none \
    --n-feat 5 \
    --test-val holdout \
    --test-val-sizes 0.3 \
    --outdir ./fast_test_results
```

should work and run quite quickly on the tiny toy dataset included in the repo.
This will produce a lot of terminal output from Optuna, so if you don't want to
see as much add the `--verbosity 0` option to above.

Currently, `classifier` can be one of `svm`, `rf`, `mlp`, `bag` or `dtree`. If
the `--step-up` argument is omitted, instead analyses for other feature
selection methods are used.

### Using a `df-analyze`-formatted Spreadsheet

Run a classification analysis on the data in the file `spreadsheet.xlsx` with
configuration options and columns specifically formatted for `df-analyze`:

```shell
python df-analyze.py --spreadsheet spreadsheet.xlsx
```

TODO: Give much more examples, e.g. an example `.csv` file:

```csv

--categoricals s x0
--target y
--mode classify
--classifiers svm mlp dummy
--nan drop
--feat-select stepup pearson
--n-feat 2
--htune
--htune-trials 50
--outdir ./results


s,x0,x1,x2,x3,y
male,0,0.739547041740053,0.312496254976371,1.12994172215702,0
female,0,0.0944212786495044,0.817089936489298,1.24646946365929,1
unspecified,1,0.323189318693224,0.00806880856795284,0.472934559871207,0
male,2,0.570184677633011,0.289003189610348,1.17633857406493,1

...
```

#### Overriding Spreadsheet Options

When spreadsheet and CLI options conflict, the `df-analyze` will prefer the CLI
args. This allows a base spreadsheet to be setup, and for minor analysis
variants to be performed without requiring copies of the formatted data file.
So for example:

```shell
python df-analyze.py --spreadsheet spreadsheet.xlsx --n-feat 5
python df-analyze.py --spreadsheet spreadsheet.xlsx --n-feat 10
python df-analyze.py --spreadsheet spreadsheet.xlsx --n-feat 20
```

would run three analyses with the options in `spreadsheet.xlsx` (or default
values) but with `--n-feat` set to 5, 10, and 20, respectively, regardless of
what is set for `--n-feat` in `spreadsheet.xlsx`.

# Usage on Compute Canada / Digital Research Alliance of Canada / Slurm HPC Clusters

## Singularity Container

TODO.

### Building

TODO.

### Running

TODO.

## Parallelization Options

TODO.

# Currently Implemented Program Features and Analyses

## Completed Features

### Single Spreadsheet for Configuration and Data

- df-analyze can now be completely configured (including data) in a single
  spreadsheet (.xlsx or .csv)
- e.g. usage: `python df-analyze.py --spreadsheet df-analyze-formatted.xlsx`
- CLI args are simply entered as header lines in the sheet
- user can additionally override spreadsheet args as needed, e.g. `python
  df-analyze.py --spreadsheet df-analyze-formatted.xlsx --target other_feature`
  (command line interface remains functional and completely compatible with
  spreadsheet configuration)

### Automated Data Preproccesing

- **NaN Removal and Handling**
  - samples with NaN target are dropped (see Target Handling below)
  - for continous features, NaNs can be either dropped, mean, median, or
    multiply imputed (default: mean imputation)
  - categorical features encode NaNs as an additional class / level

- **Bad Feature Detection and Removal**
  - features containing unusable datetime data (e.g. timeseries data) are
    automatically detected and removed, with warnings to the user
  - features containing identifiers (e.g. features that are integer or string
    and where each sample has a unique value) are automatically detected and
    removed, with user warnings
  - extremely large categorical features (more categories than abuot 1/5 of
    samples, which pose a problem fork 5-fold splitting) are automatically
    removed with user-warnings
  - "suspicious" integer features (e.g. features with less than 5 unique
    values) are detected and the user is warned to check if categorical or
    ordinal

- **Categorical Feature Handling**
  - user can
    - specify categorical feature names explicitly (preferred)
    - specify a threshold (count) for number of unique values of a feature
      required to count as categorical
  - string features (even if not user-specified) are one-hot encoded
  - NaN values are automatically treated as an additional class level (no
    dropping of NaN samples required)

- **Continuous Feature Handling**
  - continuous features are MinMax normalized to be in [0, 1]
  - this means `df-analyze` is currently **sensitive to extreme values**
  - TODO: make robust (percentile, or even
    [quantile](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer))
    normalization options available, and auto-detect such cases and warn the
    user

- **Target Handling**
  - all samples with NaN targets are dropped (categorical or continuous)
    - rarely makes sense to count correct NaN predictions toward classification
      performance
    - imputing NaNs in a regression target (e.g. mean, median) biases estimates
      of regression performance
  - categorical targets containing a class with 20 or fewer samples in a level
    have the samples corresponding to that level dropped, and the user is
    warned (these cause problems in nested stratified k-fold, and any estimated
    of any metric or performance on such a small class is essentially
    meaningless)
  - continuous or ordinal regression targets are robustly normalized using
    2.5th and 97.5th percentile values
    - with this normalization, 95% of the target values are in [0, 1]
    - thus an MAE of, say, 0.5, means that the error is about half of the
      target (robust) range
    - this also aids in the convergence and fitting of scale-sensitive models
    - this also makes prediction metrics (e.g. MAE) more comparable across
      different targets



### Feature Descriptive Statisics

- **Continuous and ordinal features**:
  - Non-robust:
    - min, mean, max, standard deviation (SD)
  - Robust:
    - 5th and 95th percentiles, median, interquartile range (IQR)
  - Moments/Other:
    - skew, kurtosis, and p-values that skew/kurtosis differ from Gaussian
    - entropy (e.g. differential / continuous entropy)
    - NaN counts and frequency

- **Categorical features**:
  - number of classes / levels
  - min, max, and median of class frequencies
  - heterogeneity (Chi-squared test of equal class sizes) and associated p-value
  - NaN counts and frequency (treated as another class label)

### Univariate Feature-Target Associations

- **Continuous/Ordinal Feature -> Categorical Target**:
  - Statistical: t-test, Mann-Whitney U, Brunner-Munzel W, Pearson r and
    associated p-values
  - Other: Cohen's d, AUROC, mutual information
  - for

- **Continuous/Ordinal Feature -> Continuous Target**:
  - Pearson's and Spearman's r and p-values
  - [F-test](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)
    and p-value
  - mutual information

- **Categorical Feature -> Continuous Target**:
  - [Kruskal-Wallace
    H](https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance)
    and p-value
  - mutual information
  - **NOTE**: There are relatively few measures of association for
    categorical-continuous variable pairs. Kruskal-Wallace H has few
    statistical assumptions, and essentially checks the extent that the medians
    of each level in the categorical variable differ significantly on the
    continuous target, and

- **Categorical Feature -> Categorical Target**:
  - Cramer's V

### Univariate Prediction Metrics for each Feature-Target Pair

- simple predictive models  are hyperparameter tuned (using 5-fold) over a small
  grid for each feature
- a dummy regressor or classifier (e.g. predict target mean, predict largest
  class) is also always fit
- reported metrics are for the best-tuned model mean performance across the 5
  folds:
  - **Continuous/Ordinal Target (e.g. regression)**:
    - Models: DummyRegressor, ElasticNet, Linear regression, SVM with radial basis
    - Metrics: accuracy, AUROC (except for SVM), sensitivity, specificity
  - **Categorical Target (e.g. classification)**:
    - Models: DummyClassifier, Logistic regression, SVM with radial basis
    - Metrics: mean abs. error, mean sq. error, median abs. error, mean abs.
      percentage error, R2, percent variance explained


## In progress / Partially Completed:

- expanded documentation of df-analyze features, configuration, and pipeline
- wrapper feature selection methods
  - will make use of univariate stats and predictions

## To Do (highest priority first):

1. include predictive **confidence measures** (either directly from models that
   output probabilities, or via Platt-scaling) in final fit model stats
1. move / **replicate documentation of df-analyze in a Wiki, README**, or
   other non-code non-CLI source (currently requires user to run `python
   df-analyze.py --help` and produces a very large amount of text)
1. test scikit-rebate
   [MultiSURF](https://epistasislab.github.io/scikit-rebate/using/#multisurf)
   for modern
   **[relief-based](https://en.wikipedia.org/wiki/Relief_(feature_selection))
   filter feature selection**
1. **containerize df-analyze** for reliable behaviour on HPC cluster

## May Not Implement:

- output **matrix of feature-feature associations**
  - a single matrix not useful because categorical / continous feature and
    target pairings mean that such a matrix would be full of different
    association measures
  - would thus require three matrices (cont-cat, cont-cont, cat-cat) to avoid
    above
