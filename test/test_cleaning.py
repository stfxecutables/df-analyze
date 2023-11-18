import pytest
from _pytest.capture import CaptureFixture

from src._constants import ELDER_DATA, MUSHROOM_DATA, ROOT
from src.cli.cli import get_options
from src.preprocessing.cleaning import (
    detect_timestamps,
    encode_categoricals,
    get_clean_data,
    handle_nans,
    load_as_df,
    normalize,
)


class TestNanHandling:
    def test_drop(self, capsys: CaptureFixture) -> None:
        options = get_options(f"--df {DATA} --target y --nan drop")
        df = load_as_df(DATA, spreadsheet=False)
        clean = handle_nans(df, target=options.target, nans=options.nan_handling)
        assert clean.isna().sum().sum() == 0


def test_timestamp_detection() -> None:
    df = load_as_df(MUSHROOM_DATA, spreadsheet=False)
    detect_timestamps(df, "target")

    df = load_as_df(ELDER_DATA, spreadsheet=False)
    with pytest.raises(ValueError):
        detect_timestamps(df, "temperature")
