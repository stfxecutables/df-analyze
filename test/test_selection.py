from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from time import perf_counter
from typing import Any, Callable

from _pytest.capture import CaptureFixture

from src._constants import ROOT
from src.analysis.univariate.associate import CatAssociation, ContAssociation
from src.cli.cli import ProgramOptions
from src.enumerables import (
    ClsScore,
    EmbedSelectionModel,
    RegScore,
    WrapperSelection,
    WrapperSelectionModel,
)
from src.selection.embedded import embed_select_features
from src.selection.filter import (
    FilterSelected,
    filter_by_univariate_associations,
    filter_by_univariate_predictions,
)
from src.selection.stepwise import stepwise_select
from src.testing.datasets import FAST_INSPECTION, TestDataset, fast_ds, med_ds, slow_ds

DATA = ROOT / "data/banking/bank.json"


def do_association_select(dataset: tuple[str, TestDataset]) -> FilterSelected:
    dsname, ds = dataset
    assocs = ds.associations(load_cached=True)
    prepared = ds.prepared(load_cached=True)
    for _ in range(25):
        options = ProgramOptions.random(ds)
        filtered = filter_by_univariate_associations(
            prepared=prepared,
            associations=assocs,
            cont_metric=ContAssociation.random(),
            cat_metric=CatAssociation.random(),
            n_cont=options.n_filter_cont,
            n_cat=options.n_filter_cat,
            n_total=options.n_feat_filter,
        )
        print(filtered.cont_scores)
        print(filtered.cat_scores)
        assert len(filtered.selected) > 0
    return filtered  # type: ignore


def do_predict_select(dataset: tuple[str, TestDataset]) -> FilterSelected:
    dsname, ds = dataset
    predictions = ds.predictions(load_cached=True)
    prepared = ds.prepared(load_cached=True)
    for _ in range(25):
        options = ProgramOptions.random(ds)
        filtered = filter_by_univariate_predictions(
            prepared=prepared,
            predictions=predictions,
            cont_metric=RegScore.random(),
            cat_metric=ClsScore.random(),
            n_cont=options.n_filter_cont,
            n_cat=options.n_filter_cat,
        )
        print(filtered.cont_scores)
        print(filtered.cat_scores)
        assert len(filtered.selected) > 0
    return filtered  # pyright: ignore


def do_embed_select(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    prepared = ds.prepared(load_cached=True)
    prep_train = prepared.representative_subsample()[0]
    options = ProgramOptions.random(ds)
    # options.embed_select = EmbedSelectionModel.Linear
    # selected = embed_select_features(prep_train=prep_train, filtered=None, options=options)
    # if selected is None or (selected.selected is None):
    #     raise ValueError("Impossible!")
    # assert len(selected.selected) > 0

    options.embed_select = EmbedSelectionModel.LGBM
    selected = embed_select_features(prep_train=prep_train, filtered=None, options=options)
    if selected is None or (selected.selected is None):
        raise ValueError("Impossible!")
    assert len(selected.selected) > 0


def do_forward_select(dataset: tuple[str, TestDataset], linear: bool) -> None:
    dsname, ds = dataset
    prepared = ds.prepared(load_cached=True)
    prep_train = prepared.representative_subsample()[0]
    options = ProgramOptions.random(ds)
    options.wrapper_model = WrapperSelectionModel.Linear if linear else WrapperSelectionModel.LGBM
    options.wrapper_select = WrapperSelection.StepUp
    options.n_feat_wrapper = 10
    if prepared.X.shape[1] <= 10:
        return
    results = stepwise_select(prep_train=prep_train, options=options)
    if results is None:
        raise ValueError("Impossible")
    selected, scores = results
    if len(selected) != options.n_feat_wrapper:
        raise ValueError("Did not select correct number of features")
    for fname, score in scores.items():
        print(f"{fname:>30}  {round(score, 3)}")


def do_linear_forward_select(dataset: tuple[str, TestDataset]) -> None:
    do_forward_select(dataset, linear=True)


def do_lgbm_forward_select(dataset: tuple[str, TestDataset]) -> None:
    do_forward_select(dataset, linear=False)


def do_backward_select(dataset: tuple[str, TestDataset], linear: bool) -> None:
    dsname, ds = dataset
    prepared = ds.prepared(load_cached=True)
    prep_train = prepared.representative_subsample()[0]
    options = ProgramOptions.random(ds)
    options.wrapper_model = WrapperSelectionModel.Linear if linear else WrapperSelectionModel.LGBM
    options.wrapper_select = WrapperSelection.StepDown
    options.n_feat_wrapper = prepared.X.shape[1] - 10
    if options.n_feat_wrapper <= 0:
        return
    results = stepwise_select(prep_train=prep_train, options=options)
    if results is None:
        raise ValueError("Impossible")
    selected, scores = results
    if len(selected) != options.n_feat_wrapper:
        raise ValueError("Did not select correct number of features")
    for fname, score in scores.items():
        print(f"{fname:>30}  {round(score, 3)}")


def do_linear_backward_select(dataset: tuple[str, TestDataset]) -> None:
    do_backward_select(dataset, linear=True)


def do_lgbm_backward_select(dataset: tuple[str, TestDataset]) -> None:
    do_backward_select(dataset, linear=False)


def do_logged(
    f: Callable[[tuple[str, TestDataset]], Any],
    file: Path,
    dataset: tuple[str, TestDataset],
) -> Any:
    start = perf_counter()
    f(dataset)
    elapsed = perf_counter() - start
    unit = "s"
    if elapsed > 60:
        elapsed /= 60
        unit = "min"
    elapsed = round(elapsed, 1)

    dsname, ds = dataset
    prep = ds.prepared(load_cached=True)
    with open(file, "a") as handle:
        handle.write(f"{dsname:>30}  {str(prep.X.shape):>15}  {elapsed} ({unit})\n")
        handle.flush()


@fast_ds
def test_associate_select_fast(dataset: tuple[str, TestDataset]) -> None:
    dsname = dataset[0]
    if dsname in ["credit-approval_reproduced"]:
        return
    do_association_select(dataset)


@med_ds
def test_associate_select_med(dataset: tuple[str, TestDataset]) -> None:
    do_association_select(dataset)


@slow_ds
def test_associate_select_slow(dataset: tuple[str, TestDataset]) -> None:
    do_association_select(dataset)


@fast_ds
def test_predict_select_fast(dataset: tuple[str, TestDataset]) -> None:
    dsname = dataset[0]
    if dsname in ["credit-approval_reproduced"]:
        return
    do_predict_select(dataset)


@med_ds
def test_predict_select_med(dataset: tuple[str, TestDataset]) -> None:
    do_predict_select(dataset)


@slow_ds
def test_predict_select_slow(dataset: tuple[str, TestDataset]) -> None:
    do_predict_select(dataset)


@fast_ds
def test_embed_select_fast(dataset: tuple[str, TestDataset], capsys: CaptureFixture) -> None:
    file = ROOT / "lgbm_embed_fast_runtimes.txt"
    with capsys.disabled():
        do_logged(do_embed_select, file, dataset)


@med_ds
def test_embed_select_med(dataset: tuple[str, TestDataset], capsys: CaptureFixture) -> None:
    file = ROOT / "lgbm_embed_med_runtimes.txt"
    with capsys.disabled():
        do_logged(do_embed_select, file, dataset)


@fast_ds
def test_linear_forward_select_fast(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    file = ROOT / "linear_forward_select_runtimes.txt"
    with capsys.disabled():
        do_logged(do_linear_forward_select, file, dataset)


@fast_ds
def test_lgbm_forward_select_fast(dataset: tuple[str, TestDataset], capsys: CaptureFixture) -> None:
    file = ROOT / "lgbm_forward_select_runtimes.txt"
    with capsys.disabled():
        do_logged(do_lgbm_forward_select, file, dataset)


@fast_ds
def test_linear_backward_select_fast(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    file = ROOT / "linear_backward_select_runtimes.txt"
    with capsys.disabled():
        do_logged(do_linear_backward_select, file, dataset)


@fast_ds
def test_lgbm_backward_select_fast(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    file = ROOT / "lgbm_backward_select_runtimes.txt"
    with capsys.disabled():
        do_logged(do_lgbm_backward_select, file, dataset)


if __name__ == "__main__":
    for dsname, ds in FAST_INSPECTION:
        if dsname != "abalone":
            continue
        print(dsname)
        # do_forward_select((dsname, ds))
        do_backward_select((dsname, ds))
