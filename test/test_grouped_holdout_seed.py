from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import numpy as np
import pandas as pd

from df_analyze.preprocessing.prepare import PreparedData


def _prepared(n: int = 200) -> PreparedData:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.integers(0, 2, size=n), name="target")
    groups = pd.Series(np.repeat(np.arange(n // 2), 2), name="g")  # pairs of rows per group
    return PreparedData(X=X, y=y, groups=groups, is_classification=True,
                        X_cont=X, X_cat=pd.DataFrame(index=X.index))


def _test_ids(prep: PreparedData, seed: int) -> frozenset:
    _, test = prep.split(train_size=0.75, seed=seed)
    return frozenset(test.groups.tolist())


def test_grouped_holdout_depends_on_seed() -> None:
    prep = _prepared()
    splits = {_test_ids(prep, seed) for seed in range(1, 6)}
    assert len(splits) > 1  # previously every seed gave the identical test set


def test_grouped_holdout_is_reproducible_and_keeps_groups_together() -> None:
    prep = _prepared()
    assert _test_ids(prep, 3) == _test_ids(prep, 3)
    train, test = prep.split(train_size=0.75, seed=3)
    assert not set(train.groups.tolist()) & set(test.groups.tolist())
