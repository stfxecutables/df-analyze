[![DOI](https://zenodo.org/badge/364694785.svg)](https://zenodo.org/badge/latestdoi/364694785)

<!-- omit from toc -->
# Contents

- [Overview](#overview)
- [Installation](#installation)
  - [Install Poetry](#install-poetry)
    - [Installing a Compatible Python Version](#installing-a-compatible-python-version)
  - [`pip` Fallback in Case of Poetry Issues](#pip-fallback-in-case-of-poetry-issues)
- [Usage](#usage)
  - [Quick Start and Examples](#quick-start-and-examples)
    - [Using Builtin Data](#using-builtin-data)
    - [Using a `df-analyze`-formatted Spreadsheet](#using-a-df-analyze-formatted-spreadsheet)
      - [Overriding Spreadsheet Options](#overriding-spreadsheet-options)
  - [Usage on Compute Canada / Digital Research Alliance of Canada / Slurm HPC Clusters](#usage-on-compute-canada--digital-research-alliance-of-canada--slurm-hpc-clusters)
    - [Building the Singularity Container](#building-the-singularity-container)
- [Analysis Pipeline](#analysis-pipeline)
    - [Feature Type and Cardinality Inference](#feature-type-and-cardinality-inference)
    - [Data Preparation](#data-preparation)
    - [Univariate Feature Analyses](#univariate-feature-analyses)
    - [Feature Selection](#feature-selection)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Final Validation](#final-validation)
  - [Philosophy](#philosophy)
    - [Recursive / Wrapper Feature Selection and Tuning](#recursive--wrapper-feature-selection-and-tuning)
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
- [Limitations](#limitations)
  - [Data Types](#data-types)


# Overview

`df-analyze` is a command-line tool for perfoming
[AutoML](https://en.wikipedia.org/w/index.php?title=Automated_machine_learning&oldid=1193286380)
on small to medium-sized tabular datasets. In particular, `df-analyze`
attempts to automate:

- feature type inference
- feature description (e.g. univariate associations and stats)
- data cleaning (e.g. NaN handling and imputation)
- training, validation, and test splitting
- feature selection
- hyperparameter tuning
- model selection and validation

and saves all key tables and outputs from this process.



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

Alternately, [build the Singularity
container](#building-the-singularity-container) and use this for running the
code.


# Usage

For full documentation of `df-analyze` run:

```shell
python df-analyze.py --help
```

which will provide a complete description of all options. Alternately, you
can see what the `--help` option outputs
[here](https://github.com/stfxecutables/df-analyze/blob/develop/docs/arguments.md),
but keep mind the actual outputs of the `--help` command are less likely to
be out of date.


## Quick Start and Examples

Run a classification analysis on the data in `small_classifier_data.json`:

### Using Builtin Data

```bash
python df-analyze.py \
    --df=data/small_classifier_data.json \
    --outdir=./demo_results \
    --mode=classify \
    --classifiers knn lgbm rf lr sgd mlp dummy \
    --embed-select none linear lgbm \
    --feat-select wrap filter embed
```

should work and run quite quickly on the tiny toy dataset included in the repo.
This will produce a lot of terminal output.

### Using a `df-analyze`-formatted Spreadsheet

Run a classification analysis on the data in the file `spreadsheet.xlsx` with
configuration options and columns specifically formatted for `df-analyze`:

```shell
python df-analyze.py --spreadsheet spreadsheet.xlsx
```

Example of an Excel spreadsheet formatted for `df-analyze`:

![](./figures/xlsx1.png)

Another valid Excel spreadsheet:

![](./figures/xlsx2.png)

Example of a `.csv` spreadsheet formatted for `df-analyze`:

```csv
--outdir ./results
--target y
--mode classify
--categoricals s x0
--classifiers knn lgbm dummy
--nan median
--norm robust
--feat-select wrap embed


s,x0,x1,x2,x3,y
male,0,0.739547,0.312496,1.129941,0
female,0,0.094421,0.817089,1.246469,1
unspecified,1,0.323189,0.008068,0.472934,0
male,2,0.570184,0.289003,1.176338,1
...
```

If you have not been introduced to command-line interfaces (CLIs) before,
this convention might seem a bit odd, but `df-analyze` primarily functions as
a CLI program. The logic is that CLI options (e.g. `--mode`) and their
parameters or values (e.g. the `classify` in `--mode classify`) are specified
one-per-line in the file top section / header, with spaces separating
parameters (e.g. the `knn lgbm dummy` parameters passed to the `--classifiers`
option), and with at least one empty line separating these options and
parameters from the actual tabular data.

Thus, the following is an **INVALIDLY FORMATTED** spreadsheet:

```csv
--outdir ./results
--target y
--mode classify
--categoricals s x0
--classifiers knn dummy
--nan median
--norm minmax
--feat-select wrap filter none
s,x0,x1,x2,x3,y
male,0,0.739547,0.312496,1.129941,0
female,0,0.094421,0.817089,1.246469,1
unspecified,1,0.323189,0.008068,0.472934,0
male,2,0.570184,0.289003,1.176338,1
...
```

because no newlines (empty lines) separate the options from the data.



#### Overriding Spreadsheet Options

When spreadsheet and CLI options conflict, the `df-analyze` will prefer the CLI
args. This allows a base spreadsheet to be setup, and for minor analysis
variants to be performed without requiring copies of the formatted data file.
So for example:

```shell
python df-analyze.py --spreadsheet sheet.xlsx --outdir ./results --nan mean
python df-analyze.py --spreadsheet sheet.xlsx --outdir ./results --nan median
python df-analyze.py --spreadsheet sheet.xlsx --outdir ./results --nan impute
```

would run three analyses with the options in `spreadsheet.xlsx` (or default
values) but with the handing of NaN value differing for each run, regardless
of what is set for `--n-feat` in `spreadsheet.xlsx`. Note that the same
output directory can be specified each time, as `df-analyze` will ensure that
all results are saved to a separate subfolder (with a unique hash reflecting
the unique combinations of options passed to `df-analyze`). This ensures data
should be overwritten only if the exact same arguments are passed twice (e.g.
perhaps if manually cleaning your data and re-running).

## Usage on Compute Canada / Digital Research Alliance of Canada / Slurm HPC Clusters

If the singularity container `df_analyze.sif` is available in the project
root, then it can be used to run arbitrary python scripts with the [helper
script](https://github.com/stfxecutables/df-analyze/blob/master/run_python_with_home.sh)
inlcluded in the repo. E.g.

```bash
cd df-analyze
./run_python_with_home.sh test/test_main.py
```

### Building the Singularity Container

This should be built on a cluster that enables the `--fakeroot` option or on a
Linux machine where you have `sudo` privileges, and the same architecture as
the cluster (likely, x86_64).

```bash
cd df-analyze/containers
./build_container_cc.sh
```

# Analysis Pipeline

The overall data preparation and analysis process comprises six steps (some
optional):

1. [Feature Type and Cardinalty Inference](#feature-type-and-cardinality-inference)
1. [Data Preparation and Preprocessing](#data-preparation)
1. [Univariate Feature Analyses](#univariate-feature-analyses)
1. [Feature Selection (optional)](#feature-selection)
1. [Hyperparameter tuning](#hyperparameter-tuning)
1. [Final validation and analyses](#final-validation)

### Feature Type and Cardinality Inference

Features are checked, in order of priority, for useless feature or features
that cannot be used by `df-anaylze`:

1. Constancy (all values identical or identical except NaNs)
2. Sequential datetime data
3. Identifiers (all values unique and not continuous / floats)

Then, features are identified as either (1) binary, or (2), one of:

1. Ordinal
2. Continuous
3. Categorical

based on a number of heuristics relating to the unique values and counts of
these values, and the string representations of the features.

### Data Preparation


1. Data Loading
   1. Type Conversions
   1. NaN unification (detecting less common NaN representations)
1. Data Cleaning
   1. Remove samples with NaN in target variable
   1. Remove junk features (constant, timeseries, identifiers)
   1. NaNs: remove or add indicators and interpolate
   1. Categorical deflation (replace undersampled classes / levels with NaN)
1. Feature Encoding
   1. Binary categorical encoding
      1. represented as single [0, 1] feature if no NaNs
      1. single NaN indicator feature added if feature is binary plus NaNs
   1. One-hot encoding of categoricals (NaN = one additional class / level)
   1. Ordinals treated as continuous
   1. Robust normalization of continuous features
2. Target Encoding
   1. Categorical targets are deflated and label encoded to values in $[0, n]$
   2. Continuous targets are robustly min-max normalized (to middle 95% of values)


### Univariate Feature Analyses

1. Univariate associations
2. Univariate predictive performances
   1. classification task / categorical target
      - tune
        [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
        (equivalent to linear SVM and/or Logistic Regression)
      - report 5-fold mean accuracy, AUROC, sensitivity and specificity
   2. regression task / continuous target
      - tune
        [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor)
        (equivalent to regularized linear regression)
      - report 5-fold mean MAE, MSqE, $R^2$, percent explained variance, and
        median absolute error

### Feature Selection

1. Remove junk features
   1. Remove highly-correlated features
1. Use filter methods
   1. Remove features with minimal univariate relation to target
   1. Keep features with largest filter metrics
1. Split data $X$ into $X_\text{train}$, $X_\text{test}$, with $X_\text{test}$
1. Step-up selection using $X_\text{train}$
1. Bayesian (Optuna) with internal 3-fold validation on $X_\text{train}$

### Hyperparameter Tuning

### Final Validation

1. Final k-fold of model tuned and trained on selected features from $X_\text{train}$
1. Final evaluation of trained model on $X_\text{test}$


## Philosophy

1. Data *preparation* is not to be optimized / tuned
   - i.e. while one might re-run analyses with different normalization options,
     this is not expected to have a major impact on results, and so normalization
     options are not included in final comparison tables
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

For this reason we prefer filter-based feature selection [methods]()


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
  - extremely large categorical features (more categories than about 1/5 of
    samples, which pose a problem for 5-fold splitting) are automatically
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

- simple linear predictive models (sklearn [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html), [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor))  are hyperparameter tuned (using 5-fold) over a small
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

# Limitations

- single target
- column names with spaces?

## Data Types

`df-anaylze` currently cannot handle:

- time-series or sequence data where the task is *forecasting*,
  - i.e. where the target variable is either a categorical or continuous
    variable that represents some subsequent or future state of a sequence of
    samples in the training data
- unencoded text data / natural language processing (NLP) tasks
  - e.g. data where a sample feature is a collection of words, like a
    sentence (as this counts as sequence data)
  - data where each sample of each feature is a single word, with implied
    ordering among features (e.g. the data is sentences, and feature1 is
    word1, feature2 is word2, and so on, with longer sentences getting a
    special padding token, e.g. short sentences become "[PAD]", "[PAD]", ...,
    "[PAD]", and long sentences are truncated)
- unsupervised tasks (e.g. clustering, representation learning, dimension
  reduction)