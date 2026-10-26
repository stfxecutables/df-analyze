from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from df_analyze.hypertune import _get_cols, _get_splits

# Column names as df-analyze prepares them: NaN indicators "<name>_NAN" and one-hot
# levels "<name>_<level>". The urine dipstick levels contain regex metacharacters.
COLUMNS = [
    "Age", "Age_NAN", "Mother_Age", "car1", "car10",
    "Ketones_in_Urine_+", "Ketones_in_Urine_++", "Ketones_in_Urine_+++", "Ketones_in_Urine_nan",
    "Blood_group_A(+)", "Blood_group_O(-)", "Stool_diarrhea", "Stool_normal", "Sex_male",
]


def _prep(n: int) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    return SimpleNamespace(X=pd.DataFrame(rng.random((n, len(COLUMNS))), columns=COLUMNS))


def _select(cols: list[str], selection: str = "embed") -> list[str]:
    train, test = _prep(8), _prep(4)
    X_train, X_test = _get_splits(train, test, selection, _get_cols(selection, cols))
    assert X_train.columns.tolist() == X_test.columns.tolist()
    assert len(X_train) == 8 and len(X_test) == 4  # rows are never dropped
    return X_train.columns.tolist()


def test_regex_metacharacters_in_levels_do_not_crash() -> None:
    # Previously: re.error "multiple repeat", then KeyError from the fallback.
    assert _select(["Ketones_in_Urine"]) == [
        "Ketones_in_Urine_+", "Ketones_in_Urine_++", "Ketones_in_Urine_+++", "Ketones_in_Urine_nan",
    ]
    assert _select(["Ketones_in_Urine_++"]) == ["Ketones_in_Urine_++"]
    assert _select(["Blood_group_A(+)"]) == ["Blood_group_A(+)"]


def test_feature_keeps_its_indicator_but_not_unrelated_columns() -> None:
    assert _select(["Age"]) == ["Age", "Age_NAN"]  # not "Mother_Age"
    assert _select(["car1"]) == ["car1"]  # not "car10"


def test_categorical_name_keeps_all_levels() -> None:
    assert _select(["Stool"], selection="pred") == ["Stool_diarrhea", "Stool_normal"]


def test_assoc_level_specific_names_are_stripped() -> None:
    # assoc selection reports names like "<feature>__<target>.<k>"
    assert _select(["Age__y.0", "Age__y.1", "Sex"], selection="assoc") == ["Age", "Age_NAN", "Sex_male"]


@pytest.mark.parametrize("selection", ["none"])
def test_no_selection_returns_everything(selection: str) -> None:
    assert _select(["Age"], selection=selection) == COLUMNS
