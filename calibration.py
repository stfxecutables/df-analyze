import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
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
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from sklearn.svm import SVC
from typing_extensions import Literal

from src.models.base import DfAnalyzeModel
from src.models.knn import KNNClassifier
from src.models.linear import LRClassifier
from src.models.svm import SVMClassifier
from src.testing.datasets import TEST_DATASETS


def calibration_plot(tuned: DfAnalyzeModel, X: DataFrame, y_true: Series, num_classes: int) -> None:
    probs = tuned.predict_proba(X)
    bins = 10

    if num_classes > 2:
        fig, axes = plt.subplots(nrows=num_classes)
        for i, ax in enumerate(axes):
            idx = y_true == i
            yt = idx
            yp = probs[:, i]
            loss = brier_score_loss(y_true=yt, y_prob=yp)
            CalibrationDisplay.from_predictions(y_true=yt, y_prob=yp, ax=ax, n_bins=bins)
            ax.set_title(f"Class {i}: Brier={loss:0.5f} (lower=better)")
    else:
        fig, ax = plt.subplots()
        CalibrationDisplay.from_predictions(y_true=y_true, y_prob=probs[:, 1], ax=ax, n_bins=bins)
        loss = brier_score_loss(y_true=y_true, y_prob=probs[:, 1])
        ax.set_title(f"Brier loss = {loss:0.5f} (lower=better)")

    fig.set_size_inches(w=10, h=10)
    fig.tight_layout()
    plt.show()


def main() -> None:
    dsname, ds = [*TEST_DATASETS.items()][5]
    # dsname, ds = [*TEST_DATASETS.items()][2]
    X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()

    C = 1e3
    # for C in [1e3, 1e4, 1e5]:
    #     svc = SVC(kernel="rbf", C=C, probability=False)
    #     svc.fit(X_tr, y_tr)
    #     score = svc.score(X_test, y_test)
    #     print(f"C={C:1.1e} Acc: {score:0.3f}")
    # sys.exit()
    svc = SVC(kernel="rbf", C=C, probability=False)
    svp = SVC(kernel="rbf", C=C, probability=True)

    start = time.time()
    svc.fit(X_tr, y_tr)
    duration = time.time() - start
    score = svc.score(X_test, y_test)
    preds = svc.predict(X_test)
    print(f"SVC Acc: {score:0.3f},  time: {duration:0.4f}")

    start = time.time()
    svp.fit(X_tr, y_tr)
    duration_p = time.time() - start
    score_p = svp.score(X_test, y_test)
    print(f"SVP Acc: {score_p:0.3f}, time: {duration_p:0.4f}")

    preds_p = svp.predict(X_test)
    probs = svp.predict_proba(X_test)
    print(f"Preds the same: {np.all(preds == preds_p)}")

    bins = 10
    if num_classes > 2:
        fig, axes = plt.subplots(nrows=num_classes)
        for i, ax in enumerate(axes):
            idx = y_test == i
            yt = idx
            yp = probs[:, i]
            loss = brier_score_loss(y_true=yt, y_prob=yp)
            CalibrationDisplay.from_predictions(y_true=yt, y_prob=yp, ax=ax, n_bins=bins)
            ax.set_title(f"Class {i}: Brier={loss:0.5f} (lower=better)")
    else:
        fig, ax = plt.subplots()
        CalibrationDisplay.from_predictions(y_true=y_test, y_prob=probs[:, 1], ax=ax, n_bins=bins)
        loss = brier_score_loss(y_true=y_test, y_prob=probs[:, 1])
        ax.set_title(f"Brier loss = {loss:0.5f} (lower=better)")
    fig.set_size_inches(w=10, h=10)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    dsname, ds = [*TEST_DATASETS.items()][5]
    # dsname, ds = [*TEST_DATASETS.items()][2]

    X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()
    model = SVMClassifier()
    # model = KNNClassifier()
    model.htune_optuna(X_tr, y_tr, n_trials=48, n_jobs=-1, verbosity=1)
    calibration_plot(model, X_test, y_true=y_test, num_classes=num_classes)
    ...
