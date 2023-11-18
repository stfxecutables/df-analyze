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

    For each categorical feature:
        - n_cats / n_levels (number of categories)
        - min_freq, max_freq, rng_freq (i.e. min/max/range class proportions)
        - homogeneity / class balance (scipy.stats.chisquare)

    For each ordinal feature:
        - min, mean, max, sd
        - p05, median, p95, IQR
        - skew, kurtosis
        - differential (continuous) entropy (scipy.stats.differential_entropy)

    """
    raise NotImplementedError()


def feature_target_stats(df: DataFrame) -> DataFrame:
    """
    For each non-categorical (including ordinal) feature:

        Binary classificataion target:
            - t-test
            - Mann-Whitney U
            - Cohen's d
            - AUROC
            - Mutual Info (sklearn.feature_selection.mutual_info_classif)
            - largest class proportion

        Multiclass classification target (for each target class / level):
            - as above

        Multiclass classification target (means over each target class above):
            - as above

        Regression target:
            - Pearson correlation (scipy.stats.pearsonr)
            - Spearman correlation (scipy.stats.spearmanr)
            - F-statistic (sklearn.feature_selection.f_regression)
            - Mutual Info (sklearn.feature_selection.mutual_info_regression)

    For each categorical feature:

        Binary classificataion target:
            - Cramer's V
            - Mutual Info (sklearn.feature_selection.mutual_info_classif)

        Multiclass classification target (for each target class / level):
            - as above

        Multiclass classification target (means over each target class above):
            - means of above

        Regression target:
            - Mutual Info (sklearn.feature_selection.mutual_info_regression)
            - Kruskal-Wallace H? (scipy.stats.kruskal) (or ANOVA)
            - mean AUROC of each level? (No...)
            - max AUROC of each level? (Yes?)

        Kruskal-Wallace H basically looks at the distribution of continuous
        values for each level of the categorical, and compares if the medians
        are different. Thus, a low H value implies each level sort of looks
        the same on the continuous target, and implies there is not much
        predictive value of the categorical variable, whereas a high H value
        implies the opposite.

            - Pearson correlation (scipy.stats.pearsonr)
            - Spearman correlation (scipy.stats.spearmanr)
            - F-statistic (sklearn.feature_selection.f_regression)
            - Mutual Info (sklearn.feature_selection.mutual_info_regression)

    """
    raise NotImplementedError()
