from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Optional

import numpy as np
import pytest
from tqdm.contrib.concurrent import process_map

from src.testing.datasets import TEST_DATASETS, TestDataset


@pytest.mark.parametrize("dataset", [*TEST_DATASETS.items()], ids=lambda pair: str(pair[0]))
def test_loading(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    assert df.shape[0] > 0
    assert df.shape[1] > 0


@pytest.mark.parametrize("dataset", [*TEST_DATASETS.items()], ids=lambda pair: str(pair[0]))
def test_categoricals(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    assert isinstance(ds.categoricals, list)
    assert all(isinstance(c, str) for c in ds.categoricals)


@pytest.mark.parametrize("dataset", [*TEST_DATASETS.items()], ids=lambda pair: str(pair[0]))
def test_splitting(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()
    if ds.is_classification:
        assert num_classes == len(np.unique(np.concatenate([y_tr, y_test])))
    assert np.isnan(np.ravel(X_tr)).sum() == 0
    assert np.isnan(np.ravel(X_test)).sum() == 0


def split(dataset: tuple[str, TestDataset]) -> dict[str, Optional[float]]:
    dsname, ds = dataset
    known = {
        "vote": None,
        "soybean": None,
        "hypothyroid": None,
        "ozone_level": None,
        "primary-tumor": None,
        "KDDCup09_churn": None,
        "internet_usage": None,
        "Midwest_Survey": None,
        "ipums_la_97-small": None,
        "Traffic_violations": None,
        "KDDCup09_appetency": None,
        "kdd_internet_usage": None,
        "analcatdata_reviewer": None,
        "analcatdata_marketing": None,
        "Midwest_Survey_nominal": None,
        "Mercedes_Benz_Greener_Manufacturing": None,
        "abalone": 0.5551178455352783,
        "ada_prior": 0.39563798904418945,
        "adult-census": 2.371124029159546,
        "adult": 38.323091983795166,
        "ames_housing": 1.0358541011810303,
        "arrhythmia": 1.300044059753418,
        "bank-marketing": 2.3752236366271973,
        "BNG(lowbwt)": 1.0574250221252441,
        "cholesterol": 0.2857170104980469,
        "cleveland": 0.2697298526763916,
        "cmc": 0.19095087051391602,
        "colic": 0.20183682441711426,
        "colleges": 49.636914014816284,
        "community_crime": 4.07422399520874,
        "cps_85_wages": 0.09392690658569336,
        "credit_approval": 1.1579420566558838,
        "credit-approval_reproduced": 0.10203003883361816,
        "cylinder-bands": 0.5937507152557373,
        "dermatology": 0.34897589683532715,
        "dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc": 0.803919792175293,
        "dresses-sales": 0.21149682998657227,
        "elder": 0.6571128368377686,
        "forest_fires": 0.5370151996612549,
        "fps_benchmark": None,
        "health_insurance": 0.6708548069000244,
        "heart-c": 0.23261594772338867,
        "house_prices_nominal": 1.0235998630523682,
        "Insurance": 0.9916539192199707,
        "ipums_la_98-small": 1.5606281757354736,
        "ipums_la_99-small": 1.974257230758667,
        "jasmine": 1.4095349311828613,
        "jungle_chess_2pcs_endgame_complete": 4.627909183502197,
        "jungle_chess_2pcs_endgame_rat_elephant": 0.6946661472320557,
        "jungle_chess_2pcs_endgame_rat_lion": 0.7239737510681152,
        "jungle_chess_2pcs_endgame_rat_panther": 0.6999568939208984,
        "Kaggle_bike_sharing_demand_challange": 0.4061439037322998,
        "KDD": 0.6705400943756104,
        "kick": 109.22614789009094,
        "mushrooms": 58.034183979034424,
        "news_popularity": 4.29059100151062,
        "nomao": 6.066659212112427,
        "okcupid-stem": 79.76939296722412,
        "OnlineNewsPopularity": 4.644309043884277,
        "pbc": 0.20116686820983887,
        "pbcseq": 0.9756228923797607,
        "shrutime": 0.49880099296569824,
        "solar_flare": 0.26229095458984375,
        "SpeedDating": 3.5273396968841553,
        "student_dropout": 2.3282999992370605,
        "student_performance_por": 0.14645624160766602,
        "telco-customer-churn": 0.45568203926086426,
        "water-treatment": 0.9395740032196045,
        "wine_quality": 0.20794892311096191,
    }
    if dsname in known:
        return {dsname: known[dsname]}

    print(f"Trying to process {dsname}")
    start = time.time()
    try:
        sink = StringIO()
        with redirect_stdout(sys.stderr):
            with redirect_stderr(sink):
                X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()
        duration = time.time() - start
    except Exception:
        duration = None
    print({dsname: duration})
    return {dsname: duration}


if __name__ == "__main__":
    # pytest.main()
    # results = process_map(split, [*TEST_DATASETS.items()], disable=True)
    for dsname, ds in TEST_DATASETS.items():
        # KDD98 has some huge categorical problems ()
        if dsname in ["KDD98"]:
            continue

        if dsname != "KDDCup09_appetency":
            continue
        print(dsname)
        warnings.simplefilter("error")
        ds.train_test_split()

    """
    Problems:



    """
