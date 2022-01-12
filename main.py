from pathlib import Path
from pprint import pformat, pprint

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.io.arff import loadarff

from src.cli import get_options

if __name__ == "__main__":
    options = get_options()
    print(options)
