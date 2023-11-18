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
from typing_extensions import Literal


def feature_descriptions(df: DataFrame) -> DataFrame:
    """
    univariate descriptive stats for each feature (robust and non-robust measures of scale and
    location—e.g. mean, median, SD, IQR, and some higher moments—e.g. skew,
    kurtosis), entropy

    For each non-categorical feature:
        - min, mean, max, sd
        - p05, median, p95, IQR
        - skew, kurtosis
        - differential (continuous) entropy (scipy.stats.differential_entropy)
    """
    raise NotImplementedError()


def feature_target_stats(df: DataFrame) -> DataFrame:
    """
    For each non-categorical feature:

        Binary classification:
            - t-test
            - Mann-Whitney U
            - Cohen's d
            - AUROC
            - Mutual Info (sklearn.feature_selection.mutual_info_classif)
            - largest class proportion

        Multiclass classification (for each target class):
            - as above

        Multiclass classification (means over each target class above):
            - as above

        Regression:
            - Pearson correlation (scipy.stats.pearsonr)
            - Spearman correlation (scipy.stats.spearmanr)
            - F-statistic (sklearn.feature_selection.f_regression)
            - Mutual Info (sklearn.feature_selection.mutual_info_regression)

        Multiclass classification

    """
    raise NotImplementedError()
