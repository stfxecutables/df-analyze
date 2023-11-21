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
from sklearn.metrics import roc_auc_score
from typing_extensions import Literal


def _cramer_v(y1: ndarray, y2: ndarray) -> float:
    if len(np.unique(y1)) == 1 or len(np.unique(y2)) == 1:
        # can't correlate constants...
        return float("nan")
    ct = crosstab(y1, y2)[1]
    return float(association(observed=ct, correction=False))


def cohens_d(g0: Union[ndarray, Series], g1: Union[ndarray, Series]) -> float:
    m0 = np.nanmean(g0)
    m1 = np.nanmean(g1)
    s0 = np.nanstd(g0, ddof=1)
    s1 = np.nanstd(g1, ddof=1)
    n0 = len(g0) - 1
    n1 = len(g1) - 1
    s_pool = np.sqrt((n0 * s0**2 + n1 * s1**2) / (n0 + n1))
    if s_pool == 0:
        return np.nan
    return (m1 - m0) / s_pool


def auroc(x: Union[ndarray, Series], y_bin: Union[ndarray, Series]) -> float:
    return roc_auc_score(y_bin, x, multi_class="raise")
