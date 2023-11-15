[![DOI](https://zenodo.org/badge/364694785.svg)](https://zenodo.org/badge/latestdoi/364694785)

<!-- omit from toc -->
# Contents

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