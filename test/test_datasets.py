from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import numpy as np
import pytest

from src.testing.datasets import TEST_DATASETS, TestDataset


@pytest.mark.parametrize("dataset", [*TEST_DATASETS.items()], ids=lambda pair: str(pair[0]))
def test_loading(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    assert df.shape[0] > 0
    assert df.shape[1] > 0


@pytest.mark.parametrize("dataset", [*TEST_DATASETS.items()], ids=lambda pair: str(pair[0]))
def test_categoricals(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    assert isinstance(ds.categoricals, list)
    assert all(isinstance(c, str) for c in ds.categoricals)


@pytest.mark.parametrize("dataset", [*TEST_DATASETS.items()], ids=lambda pair: str(pair[0]))
def test_splitting(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()
    if ds.is_classification:
        assert num_classes == len(np.unique(np.concatenate([y_tr, y_test])))
    assert np.isnan(np.ravel(X_tr)).sum() == 0
    assert np.isnan(np.ravel(X_test)).sum() == 0


if __name__ == "__main__":
    pytest.main()
