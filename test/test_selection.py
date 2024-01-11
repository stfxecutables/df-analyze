from time import perf_counter

from _pytest.capture import CaptureFixture

from src._constants import ROOT
from src.analysis.univariate.associate import CatAssociation, ContAssociation
from src.cli.cli import ProgramOptions
from src.enumerables import ClsScore, EmbedSelectionModel, RegScore
from src.selection.embedded import embed_select_features
from src.selection.filter import (
    FilterSelected,
    filter_by_univariate_associations,
    filter_by_univariate_predictions,
)
from src.testing.datasets import TestDataset, fast_ds, med_ds, slow_ds

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
    with capsys.disabled():
        start = perf_counter()
        do_embed_select(dataset)
        elapsed = perf_counter() - start
        unit = "s"
        if elapsed > 60:
            elapsed /= 60
            unit = "min"

        dsname, ds = dataset
        prep = ds.prepared(load_cached=True)
        with open(ROOT / "lgbm_embed_runtimes.txt", "a") as handle:
            handle.write(f"{dsname:>30}  {str(prep.X.shape):>10}  {elapsed} ({unit})\n")
            handle.flush()


@med_ds
def test_embed_select_med(dataset: tuple[str, TestDataset], capsys: CaptureFixture) -> None:
    # with capsys.disabled():
    do_embed_select(dataset)
