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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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
from src.saving import get_hash
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
