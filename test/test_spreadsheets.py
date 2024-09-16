from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import sys
from pathlib import Path

import pandas as pd
from cli_test_helpers import ArgvContext
from pandas import DataFrame

from df_analyze._constants import TEMPLATES
from df_analyze.analysis.univariate.associate import target_associations
from df_analyze.analysis.univariate.predict.predict import univariate_predictions
from df_analyze.cli.cli import ProgramOptions, get_options
from df_analyze.enumerables import (
    DfAnalyzeClassifier,
    DfAnalyzeRegressor,
    EmbedSelectionModel,
    FeatureSelection,
    WrapperSelection,
    WrapperSelectionModel,
)
from df_analyze.hypertune import evaluate_tuned
from df_analyze.nonsense import silence_spam
from df_analyze.preprocessing.cleaning import sanitize_names
from df_analyze.preprocessing.inspection.inspection import inspect_data
from df_analyze.preprocessing.prepare import prepare_data
from df_analyze.selection.filter import filter_select_features
from df_analyze.selection.models import model_select_features
from df_analyze.testing.datasets import (
    FAST_INSPECTION,
    FASTEST,
    MEDIUM_INSPECTION,
    SLOW_INSPECTION,
    TestDataset,
    fast_ds,
    turbo_ds,
)

DATA = TEMPLATES / "binary_classification.csv"
DATALINES = DATA.read_text()
ARGS_MIN = (TEMPLATES / "minimum_args.txt").read_text()
ARGS_FULL = (TEMPLATES / "spread_full_args.txt").read_text()

SPREAD_MIN = TEMPLATES / "student/minimal_sheet.csv"
SPREAD_MIN.write_text(f"{ARGS_MIN}\n\n{DATALINES}")

SPREAD_FULL = TEMPLATES / "student/full_sheet.csv"
# SPREAD_FULL.write_text(f"{ARGS_FULL}\n\n{DATALINES}")


def do_main(minimal: bool) -> None:
    sheet = SPREAD_MIN if minimal else SPREAD_FULL
    with ArgvContext("df-analyze.py", "--spreadsheet", f"{sheet}"):
        options = get_options()

    options.to_json()

    is_cls = options.is_classification
    prog_dirs = options.program_dirs
    target = options.target
    categoricals = options.categoricals
    ordinals = options.ordinals
    drops = options.drops

    df = options.load_df()
    df, renames = sanitize_names(df, target)
    prog_dirs.save_renames(renames)

    categoricals = renames.rename_columns(categoricals)
    ordinals = renames.rename_columns(ordinals)
    drops = renames.rename_columns(drops)

    df, inspection = inspect_data(df, target, categoricals, ordinals, drops, _warn=True)
    prog_dirs.save_inspect_reports(inspection)
    prog_dirs.save_inspect_tables(inspection)

    prepared = prepare_data(df, target, inspection, is_cls)
    prog_dirs.save_prepared_raw(prepared)
    prog_dirs.save_prep_report(prepared.to_markdown())
    prep_train, prep_test = prepared.split()

    associations = target_associations(prep_train)
    prog_dirs.save_univariate_assocs(associations)
    prog_dirs.save_assoc_report(associations.to_markdown())

    predictions = univariate_predictions(prep_train, is_cls)
    prog_dirs.save_univariate_preds(predictions)
    prog_dirs.save_pred_report(predictions.to_markdown())

    # select features via filter methods first
    assoc_filtered, pred_filtered = filter_select_features(
        prep_train, associations, predictions, options
    )
    prog_dirs.save_filter_report(assoc_filtered)
    prog_dirs.save_filter_report(pred_filtered)

    # TODO: make embedded and wrapper selection mutually exclusive. Only two
    # phases of feature selection: filter selection, and model-based
    # selection, wher model-based selection means either embedded or wrapper
    # (stepup, stepdown) methods.
    selected = model_select_features(prep_train, options)
    # selected = ModelSelected.random(ds)
    prog_dirs.save_model_selection_reports(selected)
    prog_dirs.save_model_selection_data(selected)

    silence_spam()
    eval_results = evaluate_tuned(
        prepared=prepared,
        prep_train=prep_train,
        prep_test=prep_test,
        assoc_filtered=assoc_filtered,
        pred_filtered=pred_filtered,
        model_selected=selected,
        options=options,
    )
    print(eval_results.to_markdown())
    prog_dirs.save_eval_report(eval_results)
    prog_dirs.save_eval_tables(eval_results)
    prog_dirs.save_eval_data(eval_results)


def test_spreadsheet_minimal() -> None:
    do_main(minimal=True)


def test_spreadsheet_full() -> None:
    do_main(minimal=False)


if __name__ == "__main__":
    # each about 5 minutes on M2 Macbook Air with 8 cores, using all
    # do_main(minimal=True)
    do_main(minimal=False)
