import numpy as np
from _pytest.capture import CaptureFixture

from src._constants import ROOT
from src.cleaning import get_clean_data
from src.feature_selection import select_features
from src.options import get_options

DATA = ROOT / "data/banking/bank.json"


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
