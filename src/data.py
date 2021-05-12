import os

from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series

from src._constants import FEATURE_CLEANINGS, CLEANINGS_SHORT
from src.options import ProgramOptions


class Paths:
    """Holds paths for various computationally intensive intermediates. """

    def __init__(self, options: ProgramOptions) -> None:
        self.raw: Path
        self.cleaned: Path
        self.selected: Path
        self.results: Path
        self.outdir: Path
        self.cachedir: Path

        self.raw = options.datapath
        self.outdir = options.outdir
        self.cachedir = self.outdir / "cached"
        self.cleaned = self.cleaned_path(options)


    def cleaned_path(self, options: ProgramOptions) -> Path:
        mapping = dict(zip(FEATURE_CLEANINGS, CLEANINGS_SHORT))
        opts = tuple(sorted(map(lambda o: mapping[o], options.feat_clean)))
        hexlabel = md5(str(opts).encode()).hexdigest()
        return self.cachedir / f"{hexlabel}.json"


class DataResource:
    """Container to hold data and data-related paths """

    def __init__(self, options: ProgramOptions) -> None:
        self.options: ProgramOptions = options
        self.paths = Paths(options)

        self.id: str = datapath.stem

        self.raw: Optional[DataFrame] = None
        self.is_raw_loaded: bool = False

        self.all_files = [self.cleanpath, self.feat_cleaned, self.feat_selected]

    @property
    def df(self) -> DataFrame:
        if self.is_raw_loaded:
            return self.raw
        else:
            return self.load_raw_df()
        pass

    def load_raw_df(self) -> DataFrame:
        FILETYPES = [".json", ".csv", ".npy"]
        path = self.datapath
        if path.suffix not in FILETYPES:
            raise ValueError("Invalid data file. Currently must be one of ")
        if path.suffix == ".json":
            df = pd.read_json(str(path))
        elif path.suffix == ".csv":
            df = pd.read_csv(str(path))
        elif path.suffix == ".npy":
            arr: ndarray = np.load(str(path), allow_pickle=False)
            if arr.ndim != 2:
                raise RuntimeError(
                    f"Invalid NumPy data in {path}. NumPy array must be two-dimensional."
                )
            cols = [f"c{i}" for i in range(arr.shape[1])]
            df = DataFrame(data=arr, columns=cols)
        else:
            raise RuntimeError("Unreachable!")
        return df

