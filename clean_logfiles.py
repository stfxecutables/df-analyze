from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import re
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


def load_log_text() -> tuple[str, Path]:
    parser = ArgumentParser()
    parser.add_argument("logfile", type=Path)
    args = parser.parse_args()
    logfile = Path(args.logfile)
    outfile = logfile.parent / f"{logfile.stem}.clean{logfile.suffix}"
    text = logfile.read_text()
    return text, outfile


def clean_text(text: str) -> str:
    regs = [
        r"\|[ ▏▎▍▌▋▊▉█▏████████████████]+\|",
        "Best trial",
        "is deprecated",
        "FutureWarning",
        r"warnings.warn\(",
        "__UNSORTED",  # hack for loading bug
    ]

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        matches = [re.search(reg, line) is not None for reg in regs]
        if any(matches):
            continue
        cleaned.append(line)
    cleaned = [line for line in cleaned if line != ""]
    clean = "\n".join(cleaned)
    return clean


if __name__ == "__main__":
    text, outfile = load_log_text()
    clean = clean_text(text)
    outfile.write_text(clean)
    print(clean)
