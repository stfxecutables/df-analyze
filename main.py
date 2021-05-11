import os
from argparse import ArgumentParser
from pathlib import Path
from time import ctime
from typing import Any, Dict, List
from pprint import pprint

import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.options import get_cli_args


if __name__ == "__main__":
    args = get_cli_args()
    pprint(args.__dict__, indent=2, width=80)
