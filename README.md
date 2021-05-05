# Setup

This project uses Poetry to manage dependencies, so if you want to reproduce the results you should
[install Poetry](https://python-poetry.org/docs/). Then you can install dependencies and activate
the venv with:

```shell
poetry install  # setup environment
poetry shell  # activate venv
```

# Running the Script

To run the script do:

```shell
python run.py --classifier=[classifier] [--step-up]
```

Currently, `classifier` can be one of `svm`, `rf`, `mlp`, `bag` or `dtree`. If the `--step-up`
argument is omitted, instead analyses for other feature selection methods are used.