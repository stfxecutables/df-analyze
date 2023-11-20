from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import pytest

from test.datasets import TEST_DATASETS


def test_loading() -> None:
    for name, ds in TEST_DATASETS.items():
        df = ds.load()
        assert df.shape[0] > 0
        assert df.shape[1] > 0


def test_categoricals() -> None:
    for name, ds in TEST_DATASETS.items():
        assert isinstance(ds.categoricals, list)
        assert all(isinstance(c, str) for c in ds.categoricals)


if __name__ == "__main__":
    pytest.main()
