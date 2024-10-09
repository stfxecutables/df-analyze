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

from df_analyze.analysis.univariate.associate import (
    CatClsStats,
    CatRegStats,
    ContClsStats,
    ContRegStats,
)
from df_analyze.cli.cli import ProgramOptions, Verbosity, get_options, random_cli_args
from df_analyze.enumerables import (
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
from df_analyze.loading import load_spreadsheet
from df_analyze.preprocessing.cleaning import sanitize_names
from df_analyze.preprocessing.inspection.inspection import inspect_data
from df_analyze.testing.datasets import (
    ALL_DATASETS,
    TEST_DATASETS,
    TestDataset,
    all_ds,
    fast_ds,
)


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
            drops = options.drops

            df = options.load_df()
            df, renames = sanitize_names(df, target)
            categoricals = renames.rename_columns(categoricals)
            ordinals = renames.rename_columns(ordinals)
            drops = renames.rename_columns(drops)

            inspect_data(
                df=df,
                target=target,
                grouper=None,
                categoricals=categoricals,
                ordinals=ordinals,
                _warn=True,
            )


@fast_ds
def test_rand_spreadsheet(dataset: tuple[str, TestDataset]) -> None:
    do_random_spreadsheet(dataset)
