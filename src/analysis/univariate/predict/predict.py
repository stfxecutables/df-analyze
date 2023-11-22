from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, LogisticRegressionCV
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV

"""Dict of classifiers and hparam search grid"""
REG_MODELS = {
    ElasticNetCV: {
        "l1_ratio": [[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]],
    },
    LinearRegression: {},
    SVR: {
        # see https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0088-0#Sec6
        # for a possible tuning range on C, gamma
        "kernel": ["rbf"],
        "gamma": []
    }
    LGBMRegressor: ...,
    DummyRegressor: ...,
}

CLS_MODELS = {
    LogisticRegressionCV: ...,
    SVC: ...,
    LGBMClassifier: ...,
    DummyClassifier: ...,
}
