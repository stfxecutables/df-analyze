from _pytest.capture import CaptureFixture

from src._constants import ROOT
from src.cleaning import get_clean_data
from src.options import get_options

DATA = ROOT / "data/banking/bank.json"


class TestNanCleaning:
    def test_drop_none(self, capsys: CaptureFixture) -> None:
        options = get_options(f"--df {DATA} --target y --drop-nan none")
        get_clean_data(options.cleaning_options)

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
