from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from cli_test_helpers import ArgvContext
from testing.datasets import TEST_DATASETS

from src.cli.cli import get_options

PATH = list(TEST_DATASETS.values())[0].datapath


def test_classifiers() -> None:
    opts = get_options(f"--df {PATH} --categoricals one two three")
    assert opts.categoricals == ["one", "two", "three"]


def test_quoted_classifiers() -> None:
    # NOTE: can also just confirm manually that this does work to allow
    # column names with spaces in them.
    with ArgvContext(
        "df-analyze.py",
        "--df",
        f"{PATH}",
        "--categoricals",
        "a one",
        "a two",
        "--verbosity",
        "0",
    ):
        opts = get_options()
    assert opts.categoricals == ["a one", "a two"]


if __name__ == "__main__":
    opts = get_options()
    print(opts.categoricals)
