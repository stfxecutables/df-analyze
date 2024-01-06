from math import ceil

import numpy as np
from _pytest.capture import CaptureFixture

from src._constants import ROOT
from src.cli.cli import ProgramOptions, get_options
from src.enumerables import DfAnayzeEmbedSelector
from src.feature_selection import select_features
from src.selection.embedded import embed_select_features
from src.selection.filter import filter_by_univariate_associations, filter_by_univariate_predictions
from src.testing.datasets import TestDataset, fast_ds, med_ds, slow_ds

DATA = ROOT / "data/banking/bank.json"


def do_association_select(dataset: tuple[str, TestDataset]) -> list[str]:
    dsname, ds = dataset
    prepared = ds.prepared(load_cached=True)
    assocs = ds.associations(load_cached=True)
    results = filter_by_univariate_associations(
        prepared=prepared,
        associations=assocs,
    )
    return results


def do_predict_select(dataset: tuple[str, TestDataset]) -> list[str]:
    dsname, ds = dataset
    prepared = ds.prepared(load_cached=True)
    predictions = ds.predictions(load_cached=True)
    results = filter_by_univariate_predictions(
        prepared=prepared,
        predictions=predictions,
    )
    return results


def do_embed_select(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    prepared = ds.prepared(load_cached=True)
    prep_train = prepared.representative_subsample()[0]
    options = ProgramOptions.random(ds)
    options.embed_select = DfAnayzeEmbedSelector.Linear
    selected = embed_select_features(prep_train=prep_train, filtered=None, options=options)
    assert len(selected.selected) > 0

    options.embed_select = DfAnayzeEmbedSelector.LGBM
    selected = embed_select_features(prep_train=prep_train, filtered=None, options=options)
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
    # with capsys.disabled():
    do_embed_select(dataset)


class TestBasicSelections:
    def test_multiple_args(self, capsys: CaptureFixture) -> None:
        selects = [
            "pca",
            "pca pca",
            "pca pca pca",
        ]
        base = f"--df {DATA} --target y --drop-nan none "
        args = "--n-feat {n} --feat-select {select}"
        for select in selects:
            n = np.random.randint(2, 10)
            options = get_options(base + args.format(n=n, select=select))
            df = select_features(options.selection_options, "pca", classifier=None)
            assert df.shape[1] == n + 1, "Incorrect number of features"

    def test_pca(self, capsys: CaptureFixture) -> None:
        base = f"--df {DATA} --target y --drop-nan none "
        args = "--n-feat {n} --feat-select {select}"
        for n in range(2, 5):
            options = get_options(base + args.format(n=n, select="pca"))
            df = select_features(options.selection_options, "pca", classifier=None)
            assert df.shape[1] == n + 1, "Incorrect number of features"

    def test_kpca(self, capsys: CaptureFixture) -> None:
        base = f"--df {DATA} --target y --drop-nan none "
        args = "--n-feat {n} --feat-select {select}"
        for n in range(2, 5):
            options = get_options(base + args.format(n=n, select="kpca"))
            df = select_features(options.selection_options, "kpca", classifier=None)
            assert df.shape[1] == n + 1, "Incorrect number of features"

    def test_d(self, capsys: CaptureFixture) -> None:
        base = f"--df {DATA} --target y --drop-nan none "
        args = "--n-feat {n} --feat-select {select}"
        for n in range(2, 5):
            options = get_options(base + args.format(n=n, select="d"))
            df = select_features(options.selection_options, "d", classifier=None)
            assert df.shape[1] == n + 1, "Incorrect number of features"

    def test_pearson(self, capsys: CaptureFixture) -> None:
        base = f"--df {DATA} --target y --drop-nan none "
        args = "--n-feat {n} --feat-select {select}"
        for n in range(2, 5):
            options = get_options(base + args.format(n=n, select="pearson"))
            df = select_features(options.selection_options, "pearson", classifier=None)
            assert df.shape[1] == n + 1, "Incorrect number of features"

    def test_auc(self, capsys: CaptureFixture) -> None:
        base = f"--df {DATA} --target y --drop-nan none "
        args = "--n-feat {n} --feat-select {select}"
        for n in range(2, 5):
            options = get_options(base + args.format(n=n, select="auc"))
            df = select_features(options.selection_options, "auc", classifier=None)
            assert df.shape[1] == n + 1, "Incorrect number of features"


def test_stepup(capsys: CaptureFixture) -> None:
    N = 3
    args = f"--df {DATA} --target y --drop-nan none --n-feat {N} --feat-select step-up --classifiers svm"
    options = get_options(args)
    with capsys.disabled():
        df = select_features(options.selection_options, "step-up", classifier="svm")
        assert df.shape[1] == N + 1, "Incorrect number of features"
