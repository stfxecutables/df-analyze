[![DOI](https://zenodo.org/badge/364694785.svg)](https://zenodo.org/badge/latestdoi/364694785)

This project uses Poetry to manage dependencies, so if you want to reproduce
the results you should [install Poetry](https://python-poetry.org/docs/). Then
you can install dependencies and activate the venv with:

```shell
poetry install  # setup environment
poetry shell  # activate venv
```

The project currently requires python 3.9 or python 3.10 to be installed.

## Dealing with Poetry Failures

### `pip` Fallback

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

# Running the Script

To run the script do:

```shell
python df-analyze.py --help
```

For example,

```shell
python df-analyze.py \
    --df=data/small_classifier_data.json \
    --mode=classify \
    --classifiers rf svm dtree \
    --feat-select pca pearson none \
    --n-feat 5 \
    --test-val holdout \
    --test-val-sizes 0.3 \
    --outdir=./fast_test_results
```

should work and run quite quickly on the tiny toy dataset included in the repo.
This will produce a lot of terminal output from Optuna, so if you don't want to
see as much add the `--verbosity 0` option to above.

Currently, `classifier` can be one of `svm`, `rf`, `mlp`, `bag` or `dtree`. If
the `--step-up` argument is omitted, instead analyses for other feature
selection methods are used.