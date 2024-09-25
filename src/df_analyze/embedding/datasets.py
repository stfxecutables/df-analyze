import os
import sys
import traceback
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import (
    Optional,
    Union,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas import DataFrame, Series
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer
from torch import Tensor
from tqdm import tqdm


class VisionDataset:
    def __init__(
        self,
        name: Optional[str],
        path: Path,
    ) -> None:
        self.name = name
        self.path = path
        self._df = None

    def __str__(self) -> str:
        cls = f"{self.__class__.__name__}"
        return f"{cls}('{self.name}')"

    __repr__ = __str__

    @property
    def X(self) -> Series:
        df = self.load_raw()
        return df["image"]

    @property
    def y(self) -> Series:
        df = self.load_raw()
        return df["target"]

    def validate_datapath(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(
                f"No data found at {path} (realpath: {os.path.realpath(path)})"
            )
        if path.suffix != ".parquet":
            raise ValueError(
                "Data must be in .parquet format. Please refer to df-analyze documentation"
            )
        return path

    def load_raw(self) -> DataFrame:
        # https://github.com/python-pillow/Pillow/issues/4987#issuecomment-710994934
        #
        # "
        # To protect against potential DOS attacks caused by “decompression
        # bombs” (i.e. malicious files which decompress into a huge amount of
        # data and are designed to crash or cause disruption by using up a lot of
        # memory), Pillow will issue a DecompressionBombWarning if the number of
        # pixels in an image is over a certain limit, PIL.Image.MAX_IMAGE_PIXELS.
        #
        # This threshold can be changed by setting PIL.Image.MAX_IMAGE_PIXELS. It
        # can be disabled by setting Image.MAX_IMAGE_PIXELS = None.
        #
        # If desired, the warning can be turned into an error with
        # warnings.simplefilter('error', Image.DecompressionBombWarning) or
        # suppressed entirely with warnings.simplefilter('ignore',
        # Image.DecompressionBombWarning). See also the logging documentation to
        # have warnings output to the logging facility instead of stderr.
        #
        # If the number of pixels is greater than twice
        # PIL.Image.MAX_IMAGE_PIXELS, then a DecompressionBombError will be
        # raised instead. So:
        #
        #   from PIL import Image
        #   Image.MAX_IMAGE_PIXELS = None   # disables the warning
        #   Image.open(...)   # whatever operation you now run should work
        # "

        Image.MAX_IMAGE_PIXELS = None  # disables the warning

        if self._df is not None:
            return self._df
        pq = self.root / "all.parquet"
        # PyArrow engine needed to properly decode bytes column
        df = pd.read_parquet(pq, engine="pyarrow")
        im = df["image"].apply(lambda b: Image.open(BytesIO(b)))
        if self.name == "Handwritten-Mathematical-Expression-Convert-LaTeX":
            # TODO: these images have only two dimensions since BW, so expand
            ...
        self._df = pd.concat([df["label"], im], axis=1) if "label" in df.columns else im
        return self._df


class NLPDataset:
    def __init__(
        self,
        name: str,
        root: Path,
        datafiles: dict[str, Union[Path, list[str], None]],
        is_cls: bool,
    ) -> None:
        self.name = name
        self.root = root
        self.is_cls = self.is_classification = is_cls
        self.datafiles: dict[str, Union[Path, list[str], None]] = deepcopy(datafiles)
        self.datafiles.pop("root")
        self.labels = self.datafiles.pop("labels", None)
        for subset, path in self.datafiles.items():  # make paths rel to root
            if path is not None and isinstance(path, Path):
                self.datafiles[subset] = self.root / path
        targets = self.datafiles.pop("targetcols")
        assert isinstance(targets, list)
        if len(targets) < 1:
            raise ValueError(f"Must specify targets for data: {name} at {root}")
        self.targets: list[str] = targets
        self.textcol = cast(str, self.datafiles.pop("textcol"))
        self.dropcols = cast(list[str], self.datafiles.pop("dropcols"))
        self.namecols = cast(list[str], self.datafiles.pop("labelnamecols"))
        self._df = None

    @property
    def X(self) -> Series:
        df = self.load_raw()
        return df[self.textcol]

    @property
    def y(self) -> Series:
        df = self.load_raw()
        return df[self.targets[0]]

    def load_raw(self, ignore_decompression_warning: bool = True) -> DataFrame:
        # https://github.com/python-pillow/Pillow/issues/4987#issuecomment-710994934
        #
        # "
        # To protect against potential DOS attacks caused by “decompression
        # bombs” (i.e. malicious files which decompress into a huge amount of
        # data and are designed to crash or cause disruption by using up a lot of
        # memory), Pillow will issue a DecompressionBombWarning if the number of
        # pixels in an image is over a certain limit, PIL.Image.MAX_IMAGE_PIXELS.
        #
        # This threshold can be changed by setting PIL.Image.MAX_IMAGE_PIXELS. It
        # can be disabled by setting Image.MAX_IMAGE_PIXELS = None.
        #
        # If desired, the warning can be turned into an error with
        # warnings.simplefilter('error', Image.DecompressionBombWarning) or
        # suppressed entirely with warnings.simplefilter('ignore',
        # Image.DecompressionBombWarning). See also the logging documentation to
        # have warnings output to the logging facility instead of stderr.
        #
        # If the number of pixels is greater than twice
        # PIL.Image.MAX_IMAGE_PIXELS, then a DecompressionBombError will be
        # raised instead. So:
        #
        #   from PIL import Image
        #   Image.MAX_IMAGE_PIXELS = None   # disables the warning
        #   Image.open(...)   # whatever operation you now run should work
        # "
        if ignore_decompression_warning:
            Image.MAX_IMAGE_PIXELS = None  # disables the warning

        if self._df is not None:
            return self._df

        if self.datafiles["all"] is not None:  # just load this file instead
            df = _load_datafile(self.datafiles["all"])
            if df is None:
                raise ValueError("Impossible!")
            self._df = df
            return df

        # If a simple "all" file is not provided, vertically concat all others
        datafiles = deepcopy(self.datafiles)
        datafiles.pop("all")
        dfs = [_load_datafile(file) for file in datafiles.values()]
        # TODO: check for duplicates
        df = pd.concat(dfs, axis=0, ignore_index=True)
        self._df = df
        return df

    def load(self) -> DataFrame:
        raw = self.load_raw()
        if len(self.targets) == 1:
            targetcol = self.targets[0]
        else:  # for now, just use first target
            targetcol = self.targets[0]

        textcol = self.textcol
        return raw.loc[:, [textcol, targetcol]].copy()
