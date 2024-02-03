from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from enum import EnumMeta
from io import StringIO
from tempfile import TemporaryDirectory
from typing import Tuple, Type, Union

import pytest
from cli_test_helpers import ArgvContext

from src.analysis.univariate.associate import (
    CatClsStats,
    CatRegStats,
    ContClsStats,
    ContRegStats,
)
from src.cli.cli import ProgramOptions, Verbosity, get_options, random_cli_args
from src.enumerables import (
    ClsScore,
    DfAnalyzeClassifier,
    DfAnalyzeRegressor,
    EmbedSelectionModel,
    FeatureSelection,
    NanHandling,
    Normalization,
    RegScore,
    WrapperSelection,
    WrapperSelectionModel,
)
from src.loading import load_spreadsheet
from src.preprocessing.cleaning import sanitize_names
from src.preprocessing.inspection.inspection import inspect_data
from src.testing.datasets import ALL_DATASETS, TEST_DATASETS, TestDataset, all_ds, fast_ds


def do_random_spreadsheet(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    with TemporaryDirectory() as tempdir:
        for _ in range(10):
            args, datafile = random_cli_args(ds, Path(tempdir), spreadsheet=True)
            buf = StringIO()
            ds.load().to_csv(buf, index=False)
            table = buf.getvalue()
            header = "\n".join(args) + "\n\n"
            content = header + table
            datafile.write_text(content)
            options = get_options(" ".join(args))
            target = options.target
            categoricals = options.categoricals
            ordinals = options.ordinals

            df = options.load_df()
            df, renames = sanitize_names(df, target)
            inspect_data(df, target, categoricals, ordinals, _warn=True)


@fast_ds
def test_rand_spreadsheet(dataset: tuple[str, TestDataset]) -> None:
    do_random_spreadsheet(dataset)
