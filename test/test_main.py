from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path


from src.analysis.univariate.associate import target_associations
from src.analysis.univariate.predict.predict import univariate_predictions
from src.cli.cli import ProgramOptions
from src.preprocessing.inspection.inspection import inspect_data
from src.preprocessing.prepare import prepare_data
from src.selection.models import model_select_features
from src.testing.datasets import TestDataset, fast_ds


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
    df_train, df_test = train_test_split(df)

    inspection = inspect_data(df, target, categoricals, ordinals, _warn=True)
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
    selected = model_select_features(prep_train, filtered, options)
    prog_dirs.save_model_selection_reports(selected)

    tuned = tune_models(prep_train, selected, options)
    prog_dirs.save_tuned(tuned)

    results = evaluate_models(prep_test, selected, tuned, options)
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
