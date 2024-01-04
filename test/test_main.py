from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import sys
from argparse import ArgumentParser, Namespace
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from src._types import Estimator, FeatureSelection
from src.analysis.analyses import full_estimator_analysis
from src.analysis.univariate.associate import target_associations
from src.analysis.univariate.predict.predict import univariate_predictions
from src.cli.cli import ProgramOptions, get_options
from src.loading import load_spreadsheet
from src.preprocessing.inspection.inspection import inspect_data
from src.preprocessing.prepare import prepare_data
from src.saving import FileType, ProgramDirs, try_save
from src.testing.datasets import TestDataset, all_ds, fast_ds, med_ds, slow_ds
from src.utils import Debug


def do_main(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    options = ProgramOptions.random(ds)
    options.to_json()

    is_cls = options.is_classification
    prog_dirs = options.program_dirs
    target = options.target
    categoricals = options.categoricals
    ordinals = options.ordinals

    df = options.load_df()

    inspection = inspect_data(df, target, categoricals, ordinals, _warn=True)
    prog_dirs.save_inspect_reports(inspection)
    prog_dirs.save_inspect_tables(inspection)

    prepared = prepare_data(df, target, inspection, is_cls)
    prog_dirs.save_prepared_raw(prepared)
    prog_dirs.save_prep_report(prepared.to_markdown())

    associations = target_associations(prepared)
    prog_dirs.save_univariate_assocs(associations)
    prog_dirs.save_assoc_report(associations.to_markdown())

    predictions = univariate_predictions(prepared, is_cls)
    prog_dirs.save_univariate_preds(predictions)
    prog_dirs.save_pred_report(predictions.to_markdown())

    # select features via filter methods first
    # NOTE: No report to save for below, since it is obvious based on the
    # outputs, just output a simple .csv file selected.csv with the selected
    # feature names
    filtered = filter_select_features(associations, predictions, options)
    prog_dirs.save_filter_selected_features(filtered)

    # TODO: have function below dispatch to embedded or wrapper method(s)
    # depending on user feature selection arguments.
    # TODO: make embedded and wrapper selection mutually exclusive. Only two
    # phases of feature selection: filter selection, and model-based
    # selection, where model-based selection means either embedded or wrapper
    # (stepup, stepdown) methods.
    selected = model_select_features(
        prepared, filtered, options.feat_select, options.classifiers, options.regressors
    )
    prog_dirs.save_model_selected_features(selected)

    results = evaluate_models(options, prepared, selected)
    prog_dirs.save_final_reports(results)
    prog_dirs.save_final_tables(results)
    prog_dirs.save_final_plots(results)

    """
    if options.embedded_select:
        selected

    """


@fast_ds
def test_main_fast(dataset: tuple[str, TestDataset]) -> None:
    do_main(dataset)
