from __future__ import annotations

import pandas as pd
import pytest

from df_analyze.hypertune import _subset_splits


@pytest.mark.fast
def test_subset_splits_regex_fallback_selects_matching_columns() -> None:
    train = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
            "other": [3.0, 4.0],
        },
        index=["feature", "other"],
    )
    test = pd.DataFrame(
        {
            "feature": [5.0, 6.0],
            "other": [7.0, 8.0],
        },
        index=["feature", "other"],
    )

    X_train, X_test = _subset_splits(
        train,
        test,
        selection="embed_lgbm",
        cols=["feature["],
    )

    assert X_train.columns.to_list() == ["feature"]
    assert X_test.columns.to_list() == ["feature"]
    assert X_train.index.to_list() == ["feature", "other"]
    assert X_test.index.to_list() == ["feature", "other"]
    assert X_train["feature"].to_list() == [1.0, 2.0]
    assert X_test["feature"].to_list() == [5.0, 6.0]
