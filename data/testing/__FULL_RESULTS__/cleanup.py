import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from shutil import rmtree
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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm
from typing_extensions import Literal

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    feats = sorted(ROOT.rglob("features"))
    hashed = [feat.parent for feat in feats if feat.name == "features"]
    hashed = sorted(set(hashed))
    print(hashed)
    removes = []
    for dir in hashed:
        opts = list(dir.rglob("options.json"))
        if len(opts) > 0:
            print("Keep: ", dir)
        else:
            removes.append(dir)

    for remove in tqdm(removes, desc="deleting"):
        rmtree(remove)
