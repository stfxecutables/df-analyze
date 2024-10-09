import traceback
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from df_analyze.analysis.univariate.associate import target_associations
from df_analyze.analysis.univariate.predict.predict import univariate_predictions
from df_analyze.cli.cli import get_options
from df_analyze.hypertune import evaluate_tuned
from df_analyze.nonsense import silence_spam
from df_analyze.preprocessing.cleaning import sanitize_names
from df_analyze.preprocessing.inspection.inspection import inspect_data
from df_analyze.preprocessing.prepare import prepare_data
from df_analyze.selection.filter import filter_select_features
from df_analyze.selection.models import model_select_features

RESULTS_DIR = Path(__file__).parent / "results"

ARGS = """python df-analyze.py --df heart-c.parquet --target target --mode classify --classifiers knn lgbm rf lr sgd mlp svm dummy --regressors knn lgbm rf elastic sgd mlp svm dummy --feat-select filter embed wrap --embed-select lgbm linear --wrapper-select step-up --wrapper-model linear --norm robust --nan drop median --filter-method assoc pred --filter-assoc-cont-classify mut_info --filter-assoc-cat-classify mut_info --filter-assoc-cont-regress mut_info --filter-assoc-cat-regress mut_info --filter-pred-regress mae --filter-pred-classify acc --htune-trials 50 --htune-cls-metric acc --htune-reg-metric mae --test-val-size 0.4 --outdir ./hardcoded_results"""


def main() -> None:
    # TODO:
    # get arguments
    # run analyses based on args
    options = get_options(ARGS)

    is_cls = options.is_classification
    prog_dirs = options.program_dirs
    target = options.target
    categoricals = options.categoricals
    ordinals = options.ordinals
    drops = options.drops
    grouper = options.grouper
    # joblib_cache = options.program_dirs.joblib_cache
    # if joblib_cache is not None:
    #     memory = Memory(location=joblib_cache)
    X = np.random.standard_normal([1000, 100])
    y = np.random.randint(0, 2, [1000])

    # important columns have str names when not reading from disk
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.concat(
        [DataFrame(data=X, columns=cols), Series(name="target", data=y)], axis=1
    )

    # df = options.load_df()
    df, renames = sanitize_names(df, target)
    prog_dirs.save_renames(renames)

    categoricals = renames.rename_columns(categoricals)
    ordinals = renames.rename_columns(ordinals)
    drops = renames.rename_columns(drops)

    df, inspection = inspect_data(
        df=df,
        target=target,
        grouper=grouper,
        categoricals=categoricals,
        ordinals=ordinals,
        drops=drops,
        _warn=True,
    )
    prog_dirs.save_inspect_reports(inspection)
    prog_dirs.save_inspect_tables(inspection)

    prepared = prepare_data(
        df=df,
        target=target,
        grouper=grouper,
        results=inspection,
        is_classification=is_cls,
    )
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
