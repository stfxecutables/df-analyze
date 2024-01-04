from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import sys
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from dataclasses import dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from src.cli.cli import ProgramOptions
from src.preprocessing.inspection.inspection import InspectionResults
from src.saving import ProgramDirs, get_hash
from src.testing.datasets import TestDataset, all_ds, fast_ds, med_ds, slow_ds


@all_ds
def test_random_options(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    for _ in range(100):
        ProgramOptions.random(ds)


@all_ds
def test_hashing(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    for _ in range(100):
        opts1 = ProgramOptions.random(ds)
        opts2 = ProgramOptions.random(ds)
        opts11 = deepcopy(opts1)
        opts22 = deepcopy(opts2)
        hsh1 = opts1.hash()
        hsh2 = opts2.hash()
        hsh11 = opts11.hash()
        hsh22 = opts22.hash()
        assert hsh1 != hsh2
        assert hsh1 == hsh11
        assert hsh2 == hsh22


def are_equal(obj1: Any, obj2: Any) -> bool:
    if is_dataclass(obj1) and is_dataclass(obj2):
        if obj1.__dict__.keys() != obj2.__dict__.keys():
            return False

        for key in obj1.__dict__.keys():
            attr1 = obj1.__dict__[key]
            attr2 = obj2.__dict__[key]
            eq = are_equal(attr1, attr2)
            if not eq:
                return False
        return True
    else:
        return obj1 == obj2


@all_ds
def test_json(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    tempdir = TemporaryDirectory()
    try:
        for _ in range(10):
            opts = ProgramOptions.random(ds)
            if opts.outdir is not None:
                rmtree(opts.outdir)
            root = Path(tempdir.name)
            opts.program_dirs = ProgramDirs.new(root=root)
            opts.to_json()
            assert root.exists()
            assert opts.program_dirs.options is not None
            assert opts.program_dirs.options.exists()
            opts2 = ProgramOptions.from_json(opts.program_dirs.root)
            for attr in opts.__dict__:
                attr1 = getattr(opts, attr)
                attr2 = getattr(opts2, attr)
                if not are_equal(attr1, attr2):
                    raise ValueError(
                        f"Saved attribute {attr}: {attr2} does not equal initial "
                        f"value of: {attr1}"
                    )
    except Exception as e:
        raise e
    finally:
        tempdir.cleanup()


@all_ds
def test_inspection_json(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    tempdir = TemporaryDirectory()
    inspections = ds.inspect(load_cached=True, force=False)
    outdir = Path(tempdir.name)
    outfile = outdir / "inspections.json"
    inspections.to_json(outfile)
    loaded = InspectionResults.from_json(outfile)
    assert isinstance(loaded, InspectionResults)
    for attr in inspections.__dict__:
        attr1 = getattr(inspections, attr)
        attr2 = getattr(loaded, attr)
        if not are_equal(attr1, attr2):
            raise ValueError(
                f"Saved attribute {attr}: {attr2} does not equal initial value of: {attr1}"
            )
