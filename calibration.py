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

    # C = 1e3
    """
    C=1e0, gamma=1e-3, time:  2.3873, Acc: 0.676
    C=1e0, gamma=1e-2, time:  1.9640, Acc: 0.697
    C=1e0, gamma=1e-1, time:  1.9769, Acc: 0.751
    C=1e1, gamma=1e-4, time:  2.3879, Acc: 0.676
    C=1e1, gamma=1e-3, time:  1.9439, Acc: 0.702
    C=1e1, gamma=1e-2, time:  1.6249, Acc: 0.771
    C=1e1, gamma=1e-1, time:  2.3689, Acc: 0.746
    C=1e2, gamma=1e-5, time:  2.3852, Acc: 0.676
    C=1e2, gamma=1e-4, time:  1.9410, Acc: 0.702
    C=1e2, gamma=1e-3, time:  1.5933, Acc: 0.775
    C=1e2, gamma=1e-2, time:  1.7900, Acc: 0.759
    C=1e2, gamma=1e-1, time:  2.3846, Acc: 0.721
    C=1e3, gamma=1e-6, time:  2.3799, Acc: 0.675
    C=1e3, gamma=1e-5, time:  1.9431, Acc: 0.702
    C=1e3, gamma=1e-4, time:  1.5920, Acc: 0.773
    C=1e3, gamma=1e-3, time:  1.7631, Acc: 0.775
    C=1e3, gamma=1e-2, time:  2.6966, Acc: 0.734
    C=1e3, gamma=1e-1, time:  2.3175, Acc: 0.720
    C=1e4, gamma=1e-7, time:  2.3752, Acc: 0.675
    C=1e4, gamma=1e-6, time:  1.9323, Acc: 0.702
    C=1e4, gamma=1e-5, time:  1.5921, Acc: 0.773
    C=1e4, gamma=1e-4, time:  1.7763, Acc: 0.767
    C=1e4, gamma=1e-3, time:  3.6912, Acc: 0.765
    C=1e4, gamma=1e-2, time:  3.0300, Acc: 0.690
    C=1e4, gamma=1e-1, time:  2.3028, Acc: 0.719
    C=1e5, gamma=1e-7, time:  1.9390, Acc: 0.702
    C=1e5, gamma=1e-6, time:  1.5879, Acc: 0.772
    C=1e5, gamma=1e-5, time:  1.7750, Acc: 0.766
    C=1e5, gamma=1e-4, time:  3.7669, Acc: 0.767
    C=1e5, gamma=1e-3, time: 11.4467, Acc: 0.733
    """
    for C in [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]:
        for gamma in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]:
            svc = SVC(kernel="rbf", C=C, gamma=gamma, probability=False)
            start = time.time()
            svc.fit(X_tr, y_tr)
            duration = time.time() - start
            score = svc.score(X_test, y_test)
            print(f"C={C:1.1e}, gamma={gamma:1.1e}, time: {duration:0.4f}, Acc: {score:0.3f}")
    sys.exit()

    # C=1e3, gamma=1e-3, time:  1.7631, Acc: 0.775
    svc = SVC(kernel="rbf", C=1e-3, gamma=1e-3, probability=False)
    svp = SVC(kernel="rbf", C=1e-3, gamma=1e-3, probability=True)

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
    main()
    sys.exit()
    dsname, ds = [*TEST_DATASETS.items()][5]
    # dsname, ds = [*TEST_DATASETS.items()][2]

    X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()
    model = SVMClassifier()
    # model = KNNClassifier()
    model.htune_optuna(X_tr, y_tr, n_trials=48, n_jobs=-1, verbosity=1)
    calibration_plot(model, X_test, y_true=y_test, num_classes=num_classes)
    ...
