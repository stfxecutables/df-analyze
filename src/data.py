import os

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series

class DataResource:
    """Container to hold data and data-related paths"""
    def __init__(self, datapath: Path, outdir: Optional[Path] = None) -> None:
        datapath = datapath.resolve()
        outdir = outdir.resolve()
        if not datapath.exists():
            raise FileNotFoundError(f"The specified file {datapath} does not exist.")
        if not datapath.is_file():
            raise FileNotFoundError(f"The object at {datapath} is not a file.")
        if outdir.exists():
            if not outdir.is_dir():
                raise FileExistsError(f"The specified output directory {outdir} already exists and is not a directory.")
        else:
            os.makedirs(outdir, exist_ok=True)

        self.datapath: Path = datapath
        self.id: str = datapath.stem
        self.outdir: Path = outdir
        self.cleanpath: Path = self.outdir / f"{self.id}_clean.json"


        self.all_files = [
            self.cleanpath,
            self.
        ]
