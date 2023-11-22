from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import pandas as pd
from pandas import DataFrame

from src._constants import TESTDATA

CLASSIFICATIONS = TESTDATA / "classification"
REGRESSIONS = TESTDATA / "regression"
ALL = sorted(list(CLASSIFICATIONS.glob("*")) + list(REGRESSIONS.glob("*")))


class TestDataset:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.is_classification: bool = root.parent.name == "classification"
        self.datapath = root / f"{root.name}.parquet"
        self.types = root / "types.csv"
        df = pd.read_csv(self.types)
        df = df.loc[df["type"] == "categorical"]
        df = df.loc[df["feature_name"] != "target"]
        self.categoricals = df["feature_name"].to_list()

    def load(self) -> DataFrame:
        return pd.read_parquet(self.datapath)


TEST_DATASETS: dict[str, TestDataset] = {p.name: TestDataset(p) for p in ALL}
