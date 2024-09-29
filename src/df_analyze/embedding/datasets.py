import os
import sys
import traceback
from abc import abstractmethod, abstractproperty
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

from df_analyze.embedding.cli import EmbeddingModality, EmbeddingOptions


class EmbeddingDataset:
    def __init__(
        self,
        datapath: Path,
        name: Optional[str],
    ) -> None:
        self.datapath = self.validate_datapath(datapath)
        self.name = name or self.datapath.name
        self._df = None

    def __str__(self) -> str:
        cls = f"{self.__class__.__name__}"
        return f"{cls}('{self.name}' @ {self.datapath})"

    __repr__ = __str__

    def validate_datapath(self, path: Path) -> Path:
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
        try:
            return pd.read_parquet(self.datapath)
        except Exception as e:
            raise RuntimeError(
                f"Could not load .parquet data at {self.datapath}. Details above."
            ) from e

    @abstractproperty
    def X(self) -> Series: ...

    @abstractproperty
    def y(self) -> Series: ...

    @abstractmethod
    def load(self) -> DataFrame:
        if self._df is not None:
            return self._df
        df = self.load_raw()
        self.validate_cols(df)
        df = self.validate_data(df)
        self._df = df
        return self._df

    @abstractmethod
    def validate_cols(self, df: DataFrame) -> DataFrame:
        """Check that columns are correctly named and with the correct type"""
        df = df.reset_index(drop=True)  # just in case
        VALID_COLS = [
            sorted(["image", "label"]),
            sorted(["image", "target"]),
            sorted(["text", "label"]),
            sorted(["text", "target"]),
        ]

        cols = sorted(df.columns.tolist())
        if cols not in VALID_COLS:
            raise ValueError(
                "Malformed data. Data must have only two columns, i.e. be one of:"
                f"{VALID_COLS}. Got: {cols}"
            )
        if "image" in cols:
            if not df["image"].apply(lambda x: isinstance(x, bytes)).all():
                raise TypeError(
                    "Found column 'image' in data, but not all rows are `bytes` type"
                )
        if "text" in cols:
            if not df["image"].apply(lambda x: isinstance(x, str)).all():
                raise TypeError(
                    "Found column 'text' in data, but not all rows are `str` type"
                )
        if "label" in cols:
            try:
                labels = df["label"].astype(np.int64, errors="raise")
            except Exception as e:
                raise TypeError(
                    "Found column 'label' in data, but encountered error (above) "
                    "when attempting to coerce to np.int64."
                ) from e
            if not (labels == df["label"]).all():
                raise TypeError(
                    "Found column 'label' in data, but labels change values after "
                    "coercion to np.int64. This likely means labels are stored in "
                    "a floating point or other format. Convert them to an integer format "
                    "to resolve this error."
                )
        if "target" in cols:
            try:
                target = df["target"].astype(float, errors="raise")
            except Exception as e:
                raise TypeError(
                    "Found column 'target' in data, but encountered error (above) "
                    "when attempting to coerce to float."
                ) from e
            if not (target == df["target"]).all():
                raise TypeError(
                    "Found column 'target' in data, but targets change values after "
                    "coercion to float. This likely means you have either NaN values "
                    "or other inappropriate data types (e.g. object, string) in your "
                    "target column. Convert these values to floating point or remove "
                    "them in order to resolve this error."
                )

    @abstractmethod
    def validate_data(self, df: DataFrame) -> DataFrame:
        """
        To be performed AFTER .load(), this checks some basic sanity stuff
        like whether all images have a reasonable shape, such as image size and
        channels, or text length, and also number of samples.
        """


class VisionDataset(EmbeddingDataset):
    def __init__(
        self,
        datapath: Path,
        name: Optional[str],
    ) -> None:
        super().__init__(datapath=datapath, name=name)

    @property
    def X(self) -> Series:
        df = self.load_raw()
        return df["image"]

    @property
    def y(self) -> Series:
        df = self.load_raw()
        return df["target"]

    def validate_data(self, df: DataFrame) -> DataFrame:
        pass

    def load(self) -> DataFrame:
        if self._df is not None:
            return self._df

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

        # PyArrow engine needed to properly decode bytes column
        df = pd.read_parquet(self.datapath, engine="pyarrow")
        im = df["image"].apply(lambda b: Image.open(BytesIO(b)))

        self._df = pd.concat([df["label"], im], axis=1) if "label" in df.columns else im
        return self._df


class NLPDataset(EmbeddingDataset):
    def __init__(
        self,
        datapath: Path,
        name: Optional[str],
        is_cls: bool,
    ) -> None:
        super().__init__(datapath=datapath, name=name)

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


def dataset_from_opts(opts: EmbeddingOptions) -> Union[VisionDataset, NLPDataset]:
    if opts.modality is EmbeddingModality.NLP:
        return NLPDataset(name=opts.name, datapath=opts.datapath)
