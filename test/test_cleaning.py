from _pytest.capture import CaptureFixture

from src._constants import ROOT
from src.cli.cli import get_options
from src.preprocessing.cleaning import (
    encode_categoricals,
    get_clean_data,
    handle_nans,
    load_as_df,
    normalize,
)

DATA = ROOT / "data/banking/bank.json"


class TestNanHandling:
    def test_drop(self, capsys: CaptureFixture) -> None:
        options = get_options(f"--df {DATA} --target y --nan drop")
        df = load_as_df(DATA, spreadsheet=False)
        clean = handle_nans(df, target=options.target, nans=options.nan_handling)
        assert clean.isna().sum().sum() == 0

    def test_drop_all(self, capsys: CaptureFixture) -> None:
        options = get_options(f"--df {DATA} --target y --drop-nan all")
        df = get_clean_data(options.cleaning_options)
        assert not df.isnull().any().any()

    def test_drop_rows(self, capsys: CaptureFixture) -> None:
        options = get_options(f"--df {DATA} --target y --drop-nan rows")
        df = get_clean_data(options.cleaning_options)
        assert not df.isnull().any().any()

    def test_drop_cols(self, capsys: CaptureFixture) -> None:
        options = get_options(f"--df {DATA} --target y --drop-nan cols")
        df = get_clean_data(options.cleaning_options)
        assert not df.isnull().any().any()
