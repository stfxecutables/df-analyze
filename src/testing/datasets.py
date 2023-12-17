from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import pickle
from typing import Literal, Optional, cast
from warnings import catch_warnings, filterwarnings

import jsonpickle
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split as tt_split
from sklearn.preprocessing import KBinsDiscretizer

from src._constants import TESTDATA
from src.analysis.univariate.associate import target_associations
from src.enumerables import NanHandling
from src.preprocessing.cleaning import (
    clean_regression_target,
    drop_unusable,
    encode_categoricals,
    encode_target,
    handle_continuous_nans,
    normalize,
)
from src.preprocessing.inspection.inspection import InspectionResults, inspect_data
from src.preprocessing.prepare import PreparedData, prepare_data

CLASSIFICATIONS = TESTDATA / "classification"
REGRESSIONS = TESTDATA / "regression"
ALL = sorted(list(CLASSIFICATIONS.glob("*")) + list(REGRESSIONS.glob("*")))

TEST_CACHE = TESTDATA / "__TEST_CACHE__"
TEST_CACHE.mkdir(exist_ok=True, parents=True)


class TestDataset:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.is_classification: bool = root.parent.name == "classification"
        self.datapath = root / f"{root.name}.parquet"
        self.dsname = root.name
        self.types = root / "types.csv"
        df = pd.read_csv(self.types)
        dfc = df.loc[df["type"] != "categorical"]
        dfc = dfc.loc[dfc["feature_name"] != "target"]
        df = df.loc[df["type"] == "categorical"]
        df = df.loc[df["feature_name"] != "target"]
        self.categoricals = df["feature_name"].to_list()
        self.continuous = dfc["feature_name"].to_list()
        self.is_multiclass = False

        df = self.load()
        if self.is_classification:
            num_classes = len(np.unique(df["target"].astype(str)))
            self.is_multiclass = num_classes > 2
        self.shape = df.shape

        self.inspect_cachefile = TEST_CACHE / f"{self.dsname}_inspect.json"
        self.prep_cachefile = TEST_CACHE / f"{self.dsname}_prepare.pickle"

        self.assoc_cachedir = TEST_CACHE / f"{self.dsname}_associate"
        self.preds_cachedir = TEST_CACHE / f"{self.dsname}_preds"
        if not self.assoc_cachedir.exists():
            self.assoc_cachedir.mkdir(exist_ok=True, parents=True)
        if not self.preds_cachedir.exists():
            self.preds_cachedir.mkdir(exist_ok=True, parents=True)

        self.assoc_cachefiles = {
            "cont": self.assoc_cachedir / "cont.parquet",
            "cat": self.assoc_cachedir / "cat.parquet",
        }
        self.preds_cachefiles = {
            "cont": self.preds_cachedir / "cont.parquet",
            "cat": self.preds_cachedir / "cat.parquet",
        }

    def inspect(self, load_cached: bool = True, overwrite_cache: bool = False) -> InspectionResults:
        if load_cached and overwrite_cache:
            raise ValueError("Cannot both use and over-write cache, this is pointless.")

        if load_cached and self.inspect_cachefile.exists():
            results = jsonpickle.decode(self.inspect_cachefile.read_text())
            return cast(InspectionResults, results)

        df = self.load()
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            results = inspect_data(df, "target", self.categoricals, [], _warn=False)
        if overwrite_cache:
            enc = str(jsonpickle.encode(results))
            self.inspect_cachefile.write_text(enc)
            return results

        if not self.inspect_cachefile.exists():
            enc = str(jsonpickle.encode(results))
            self.inspect_cachefile.write_text(enc)
        return results

    def prepared(self, load_cached: bool = True, overwrite_cache: bool = False) -> PreparedData:
        if load_cached and self.prep_cachefile.exists():
            # results = jsonpickle.decode(self.prep_cachefile.read_text())
            with open(self.prep_cachefile, "rb") as handle:
                results = pickle.load(handle)
            return cast(PreparedData, results)

        df = self.load()
        inspect = self.inspect(load_cached=True)
        prep = prepare_data(df, "target", inspect, self.is_classification)

        if overwrite_cache:
            # enc = str(jsonpickle.encode(prep))
            # self.prep_cachefile.write_text(enc)
            with open(self.prep_cachefile, "wb") as handle:
                pickle.dump(prep, handle)
            return prep

        if not self.prep_cachefile.exists():
            # enc = str(jsonpickle.encode(prep))
            # self.prep_cachefile.write_text(enc)
            with open(self.prep_cachefile, "wb") as handle:
                pickle.dump(prep, handle)
        return prep

    def associations(
        self, load_cached: bool = True, overwrite_cache: bool = False
    ) -> tuple[Optional[DataFrame], Optional[DataFrame]]:
        if load_cached and all(file.exists() for file in self.assoc_cachefiles.values()):
            conts = pd.read_parquet(self.assoc_cachefiles["cont"])
            cats = pd.read_parquet(self.assoc_cachefiles["cat"])
            if conts.empty:
                conts = None
            if cats.empty:
                cats = None
            return conts, cats

        prep = self.prepared(load_cached=True)
        conts, cats = target_associations(prep)
        if conts is None:
            conts = DataFrame()
        if cats is None:
            cats = DataFrame()

        if overwrite_cache:
            conts.to_parquet(self.assoc_cachefiles["cont"])
            cats.to_parquet(self.assoc_cachefiles["cat"])
            if conts.empty:
                conts = None
            if cats.empty:
                cats = None
            return conts, cats

        if not all(file.exists() for file in self.assoc_cachefiles.values()):
            conts.to_parquet(self.assoc_cachefiles["cont"])
            cats.to_parquet(self.assoc_cachefiles["cat"])

        if conts.empty:
            conts = None
        if cats.empty:
            cats = None
        return conts, cats

    def predictions(
        self, load_cached: bool = True, overwrite_cache: bool = False
    ) -> tuple[Optional[DataFrame], Optional[DataFrame]]:
        if load_cached and all(file.exists() for file in self.preds_cachefiles.values()):
            conts = pd.read_parquet(self.preds_cachefiles["cont"])
            cats = pd.read_parquet(self.preds_cachefiles["cat"])
            if conts.empty:
                conts = None
            if cats.empty:
                cats = None
            return conts, cats

        prep = self.prepared(load_cached=True)
        conts, cats = target_associations(prep)
        if conts is None:
            conts = DataFrame()
        if cats is None:
            cats = DataFrame()

        if overwrite_cache:
            conts.to_parquet(self.assoc_cachefiles["cont"])
            cats.to_parquet(self.assoc_cachefiles["cat"])
            if conts.empty:
                conts = None
            if cats.empty:
                cats = None
            return conts, cats

        if not all(file.exists() for file in self.assoc_cachefiles.values()):
            conts.to_parquet(self.assoc_cachefiles["cont"])
            cats.to_parquet(self.assoc_cachefiles["cat"])

        if conts.empty:
            conts = None
        if cats.empty:
            cats = None
        return conts, cats

    def load(self) -> DataFrame:
        return pd.read_parquet(self.datapath)

    def train_test_split(
        self, test_size: float = 0.2
    ) -> tuple[DataFrame, DataFrame, Series, Series, int]:
        df = self.load()
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            results = self.inspect(load_cached=True)
            df = drop_unusable(df, results)
            df, X_cont, nan_ind = handle_continuous_nans(
                df, target="target", results=results, nans=NanHandling.Median
            )
            df = encode_categoricals(
                df,
                target="target",
                results=results,
            )[0]
            df = normalize(df, "target")
            df = df.copy(deep=True)

            X = df.drop(columns="target")
            y = df["target"]

            if self.is_classification:
                X, y = encode_target(X, y)
            else:
                X, y = clean_regression_target(X, y)

            strat = y
            if not self.is_classification:
                yy = y.to_numpy().reshape(-1, 1)
                strat = KBinsDiscretizer(n_bins=3, encode="ordinal").fit_transform(yy)
                strat = strat.ravel()
        X_tr, X_test, y_tr, y_test = tt_split(X, y, test_size=test_size, stratify=strat)
        num_classes = len(np.unique(y)) if self.is_classification else 1
        return X_tr, X_test, y_tr, y_test, num_classes

    __test__ = False  # https://stackoverflow.com/a/59888230


def fake_data(
    mode: Literal["classify", "regress"], noise: float = 1.0
) -> tuple[DataFrame, DataFrame, Series, Series]:
    N = 100
    C = 20

    X_cont_tr = np.random.standard_normal([N, C])
    X_cont_test = np.random.standard_normal([N, C])

    cat_sizes = np.random.randint(2, 20, C)
    cats_tr = [np.random.randint(0, c) for c in cat_sizes]
    cats_test = [np.random.randint(0, c) for c in cat_sizes]

    X_cat_tr = np.empty([N, C])
    for i, cat in enumerate(cats_tr):
        X_cat_tr[:, i] = cat

    X_cat_test = np.empty([N, C])
    for i, cat in enumerate(cats_test):
        X_cat_test[:, i] = cat

    df_cat_tr = pd.get_dummies(DataFrame(X_cat_tr))
    df_cat_test = pd.get_dummies(DataFrame(X_cat_test))

    df_cont_tr = DataFrame(X_cont_tr)
    df_cont_test = DataFrame(X_cont_test)

    df_tr = pd.concat([df_cont_tr, df_cat_tr], axis=1)
    df_test = pd.concat([df_cont_test, df_cat_test], axis=1)

    cols = [f"f{i}" for i in range(df_tr.shape[1])]
    df_tr.columns = cols
    df_test.columns = cols

    weights = np.random.uniform(0, 1, 2 * C)
    y_tr = np.dot(df_tr.values, weights) + np.random.normal(0, noise, N)
    y_test = np.dot(df_test.values, weights) + np.random.normal(0, noise, N)

    if mode == "classify":
        encoder = KBinsDiscretizer(n_bins=2, encode="ordinal")
        encoder.fit(y_tr.reshape(-1, 1))
        y_tr = encoder.transform(y_tr.reshape(-1, 1))
        y_test = encoder.transform(y_test.reshape(-1, 1))

    target_tr = Series(np.asarray(y_tr).ravel(), name="target")
    target_test = Series(np.asarray(y_test).ravel(), name="target")

    return df_tr, df_test, target_tr, target_test


__UNSORTED: list[tuple[str, TestDataset]] = [(p.name, TestDataset(p)) for p in ALL]

TEST_DATASETS: dict[str, TestDataset] = dict(sorted(__UNSORTED, key=lambda p: p[1].load().shape[0]))

INSPECTION_TIMES = {
    "KDD98": 68.49440933300002,
    "KDDCup09_appetency": 29.841349791,
    "KDDCup09_churn": 24.500925083,
    "Traffic_violations": 7.761734875000002,
    "okcupid-stem": 7.070780333000002,
    "kick": 5.860810333000003,
    "nomao": 5.264413917000006,
    "adult": 4.739781624999999,
    "news_popularity": 3.348081999999998,
    "jungle_chess_2pcs_endgame_complete": 3.2254894579999984,
    "OnlineNewsPopularity": 3.165630499999999,
    "mushrooms": 3.1523351669999897,
    "Mercedes_Benz_Greener_Manufacturing": 2.320836,
    "adult-census": 2.0808727919999974,
    "kdd_internet_usage": 1.856913500000001,
    "SpeedDating": 1.7916436250000025,
    "internet_usage": 1.7481538749999999,
    "fps_benchmark": 1.5828882909999997,
    "bank-marketing": 1.5273100420000034,
    "cholesterol": 1.4553109579999997,
    "ames_housing": 1.2510204170000012,
    "ipums_la_97-small": 1.0387160410000007,
    "ipums_la_99-small": 0.9914642499999999,
    "ipums_la_98-small": 0.9687000410000017,
    "Insurance": 0.9027092499999974,
    "colleges": 0.8229736249999995,
    "cleveland": 0.7004754589999997,
    "jasmine": 0.6840260829999991,
    "ozone_level": 0.6536169579999989,
    "house_prices_nominal": 0.6510138750000003,
    "BNG(lowbwt)": 0.46148729200000105,
    "health_insurance": 0.4552565830000006,
    "dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc": 0.3985209169999999,
    "jungle_chess_2pcs_endgame_rat_elephant": 0.35591020899999926,
    "jungle_chess_2pcs_endgame_rat_lion": 0.35248429099999967,
    "jungle_chess_2pcs_endgame_rat_panther": 0.3501694999999998,
    "hypothyroid": 0.3442913749999992,
    "Midwest_Survey_nominal": 0.2911310829999998,
    "KDD": 0.28601433400000076,
    "telco-customer-churn": 0.28339487499999905,
    "Midwest_Survey": 0.27862320800000084,
    "ada_prior": 0.2538542500000016,
    "Midwest_survey2": 0.24950150000000093,
    "shrutime": 0.22340208299999986,
    "arrhythmia": 0.22146737500000047,
    "Kaggle_bike_sharing_demand_challange": 0.21438554099999863,
    "soybean": 0.16840979200000028,
    "student_dropout": 0.15945937499999907,
    "cylinder-bands": 0.11440770899999997,
    "credit_approval": 0.09032287500000002,
    "vote": 0.09019129200000009,
    "dresses-sales": 0.08566837500000002,
    "credit-approval_reproduced": 0.07974441699999968,
    "colic": 0.07908112500000009,
    "primary-tumor": 0.07623075000000057,
    "analcatdata_marketing": 0.07507525000000026,
    "abalone": 0.07501758300000105,
    "wine_quality": 0.07421587500000015,
    "student_performance_por": 0.07353629099999992,
    "pbcseq": 0.06485062500000005,
    "water-treatment": 0.059749041999999974,
    "heart-c": 0.058186541000000425,
    "solar_flare": 0.05271508300000072,
    "dermatology": 0.0504779580000001,
    "cps_85_wages": 0.04649529100000027,
    "analcatdata_reviewer": 0.045098624999999615,
    "elder": 0.0446324580000006,
    "cmc": 0.041142250000000935,
    "pbc": 0.04081912499999962,
    "forest_fires": 0.03245541599999946,
}

FAST_INSPECTION: list[tuple[str, TestDataset]] = []
MEDIUM_INSPECTION: list[tuple[str, TestDataset]] = []
SLOW_INSPECTION: list[tuple[str, TestDataset]] = []
for dsname, ds in TEST_DATASETS.items():
    if dsname in INSPECTION_TIMES:
        runtime = INSPECTION_TIMES[dsname]
        if runtime < 1.0:
            FAST_INSPECTION.append((dsname, ds))
        elif runtime < 5.0:
            MEDIUM_INSPECTION.append((dsname, ds))
        else:
            SLOW_INSPECTION.append((dsname, ds))

FAST_INSPECTION = sorted(FAST_INSPECTION, key=lambda d: str(d[0]).lower())
MEDIUM_INSPECTION = sorted(MEDIUM_INSPECTION, key=lambda d: str(d[0]).lower())
SLOW_INSPECTION = sorted(SLOW_INSPECTION, key=lambda d: str(d[0]).lower())


# https://stackoverflow.com/a/5409569
def composed(*decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


all_ds = pytest.mark.parametrize(
    "dataset",
    [*TEST_DATASETS.items()],
    ids=lambda pair: str(pair[0]),
)
fast_ds = composed(
    pytest.mark.parametrize(
        "dataset",
        FAST_INSPECTION,
        ids=lambda pair: str(pair[0]),
    ),
    pytest.mark.fast,
)
med_ds = composed(
    pytest.mark.parametrize(
        "dataset",
        MEDIUM_INSPECTION,
        ids=lambda pair: str(pair[0]),
    ),
    pytest.mark.med,
)
slow_ds = composed(
    pytest.mark.parametrize(
        "dataset",
        SLOW_INSPECTION,
        ids=lambda pair: str(pair[0]),
    ),
    pytest.mark.slow,
)
