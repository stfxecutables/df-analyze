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
from preprocessing.inspection.inspection import InspectionResults
from typing_extensions import Literal


def inspection_summary(results: InspectionResults) -> None:
    """Generate a short overview of the automated inspection"""
    ...


def inspection_report(results: InspectionResults) -> None:
    """Generate a complete report of the automated inspection"""
    ...
