from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import numpy as np
import pytest

from src.testing.datasets import TEST_DATASETS


def test_loading() -> None:
    for name, ds in TEST_DATASETS.items():
        df = ds.load()
        assert df.shape[0] > 0
        assert df.shape[1] > 0


def test_categoricals() -> None:
    for name, ds in TEST_DATASETS.items():
        assert isinstance(ds.categoricals, list)
        assert all(isinstance(c, str) for c in ds.categoricals)


def test_splitting() -> None:
    for name, ds in TEST_DATASETS.items():
        X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()
        if ds.is_classification:
            assert num_classes == len(np.unique(np.concatenate([y_tr, y_test])))
        assert np.isnan(np.ravel(X_tr)).sum() == 0
        assert np.isnan(np.ravel(X_test)).sum() == 0


if __name__ == "__main__":
    pytest.main()
