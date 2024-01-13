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

from src.analysis.univariate.associate import target_associations
from src.analysis.univariate.predict.predict import univariate_predictions
from src.cli.cli import ProgramOptions
from src.enumerables import DfAnalyzeClassifier, DfAnalyzeRegressor
from src.hypertune import evaluate_tuned
from src.nonsense import silence_spam
from src.preprocessing.inspection.inspection import inspect_data
from src.preprocessing.prepare import prepare_data
from src.selection.filter import filter_select_features
from src.selection.models import ModelSelected
from src.testing.datasets import FAST_INSPECTION, TestDataset, fast_ds

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def do_main(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname == "dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc":
        return  # target undersampled levels
    options = ProgramOptions.random(ds)
    if ds.is_classification:
        # options.classifiers = tuple([x for x in DfAnalyzeClassifier])
        options.classifiers = (
            DfAnalyzeClassifier.Dummy,
            DfAnalyzeClassifier.KNN,
            DfAnalyzeClassifier.SGD,
            # DfAnalyzeClassifier.LR,
            # DfAnalyzeClassifier.MLP,
            DfAnalyzeClassifier.LGBM,
            # DfAnalyzeClassifier.SVM,
        )
        # options.classifiers = (DfAnalyzeClassifier.MLP,)
    else:
        options.regressors = tuple([x for x in DfAnalyzeRegressor])
        options.regressors = (
            DfAnalyzeRegressor.Dummy,
            DfAnalyzeRegressor.KNN,
            DfAnalyzeRegressor.SGD,
            DfAnalyzeRegressor.ElasticNet,
            # DfAnalyzeRegressor.MLP,
            DfAnalyzeRegressor.LGBM,
            # DfAnalyzeRegressor.SVM,
        )
        # options.regressors = (DfAnalyzeRegressor.MLP,)
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
    selected = model_select_features(prep_train, filtered, options)
    selected = ModelSelected.random(ds)
    prog_dirs.save_model_selection_reports(selected)

    silence_spam()
    eval_results = evaluate_tuned(
        prepared=prepared,
        prep_train=prep_train,
        prep_test=prep_test,
        assoc_filtered=assoc_filtered,
        pred_filtered=pred_filtered,
        model_selected=model_selected,
        options=options,
    )
    print(eval_results.to_markdown())
    prog_dirs.save_eval_report(eval_results)

    # prog_dirs.save_final_reports(results)
    # prog_dirs.save_final_tables(results)
    # prog_dirs.save_final_plots(results)

    """
    if options.embedded_select:
        selected

    """


@fast_ds
def test_main_fast(dataset: tuple[str, TestDataset]) -> None:
    do_main(dataset)


if __name__ == "__main__":
    for dsname, ds in FAST_INSPECTION:
        if dsname in [
            "abalone",
            "analcatdata_marketing",
            "ada_prior",
            "BNG(lowbwt)",
            "cmc",
        ]:
            continue
        if not ds.is_classification:
            continue
        do_main((dsname, ds))
