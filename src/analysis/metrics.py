import os
import sys
from argparse import ArgumentParser, Namespace
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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.stats.contingency import association, crosstab
from sklearn.metrics import cohen_kappa_score as _kappa
from sklearn.metrics import confusion_matrix as confusion
from typing_extensions import Literal


def _cramer_v(y1: ndarray, y2: ndarray) -> float:
    if len(np.unique(y1)) == 1 or len(np.unique(y2)) == 1:
        # can't correlate constants...
        return float("nan")
    ct = crosstab(y1, y2)[1]
    return float(association(observed=ct, correction=False))
