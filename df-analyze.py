from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
SRC = Path(__file__).resolve().parent / "src"  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
sys.path.append(str(SRC))  # isort: skip
# fmt: on


import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import List, TypeVar, Union
from warnings import warn

from pandas import DataFrame

from src.df_analyze.analysis.univariate.associate import target_associations
from src.df_analyze.analysis.univariate.predict.predict import univariate_predictions
from src.df_analyze.cli.cli import ProgramOptions, get_options
from src.df_analyze.hypertune import evaluate_tuned
from src.df_analyze.nonsense import silence_spam
from src.df_analyze.preprocessing.cleaning import sanitize_names
from src.df_analyze.preprocessing.inspection.inspection import inspect_data
from src.df_analyze.preprocessing.prepare import prepare_data
from src.df_analyze.selection.filter import filter_select_features
from src.df_analyze.selection.models import model_select_features

RESULTS_DIR = Path(__file__).parent / "results"

T = TypeVar("T")


def listify(item: Union[T, list[T], tuple[T, ...]]) -> List[T]:
    if isinstance(item, list):
        return item
    if isinstance(item, tuple):
        return [i for i in item]
    return [item]


def sort_df(df: DataFrame) -> DataFrame:
    """Auto-detect if classification or regression based on columns and sort"""
    cols = [c.lower() for c in df.columns]
    is_regression = False
    sort_col = None
    for col in cols:
        if ("mae" in col) and ("sd" not in col):
            is_regression = True
            sort_col = col
            break
    if sort_col is None:
        return df
    ascending = is_regression
    return df.sort_values(by=sort_col, ascending=ascending)


def print_sorted(df: DataFrame) -> None:
    """Auto-detect if classification or regression based on columns"""
    cols = [c.lower() for c in df.columns]
    is_regression = None
    sort_col = None
    for col in cols:
        if "mae" in col:
            is_regression = True
            break
        if "acc" in col:
            is_regression = False
            break
    if is_regression is None:
        print(df.to_markdown(tablefmt="simple", floatfmt="0.3f"))
        return
    sort_col = "mae" if is_regression else "acc"
    ascending = is_regression
    table = df.sort_values(by=sort_col, ascending=ascending).to_markdown(
        tablefmt="simple", floatfmt="0.3f"
    )
    print(table)


def log_options(options: ProgramOptions) -> None:
    opts = deepcopy(options.__dict__)

    print("Will run analyses with options:")
    pprint(opts, indent=2, depth=2, compact=False)


def main() -> None:
    # TODO:
    # get arguments
    # run analyses based on args
    options = get_options()
    # if options.verbosity.value > 0:
    #     log_options(options)

    print(options.drops)
    # sys.exit(0)
    is_cls = options.is_classification
    prog_dirs = options.program_dirs
    target = options.target
    categoricals = options.categoricals
    ordinals = options.ordinals
    drops = options.drops
    # joblib_cache = options.program_dirs.joblib_cache
    # if joblib_cache is not None:
    #     memory = Memory(location=joblib_cache)

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
    prog_dirs.save_eval_report(eval_results)
    prog_dirs.save_eval_tables(eval_results)
    prog_dirs.save_eval_data(eval_results)
    try:
        print(eval_results.to_markdown())
    except ValueError as e:
        warn(
            f"Got error when attempting to print final report:\n{e}\n"
            f"Details:\n{traceback.format_exc()}"
        )


if __name__ == "__main__":
    main()
