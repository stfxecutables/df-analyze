from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import numpy as np
import torch
from pytest import CaptureFixture
from sklearn.model_selection import train_test_split

from src.models.mlp import MLPEstimator
from src.testing.datasets import TEST_DATASETS, fake_data

MAX_N = 1000
MAX_EPOCHS = 10


def test_mlp_classifier(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        for dsname, ds in TEST_DATASETS.items():
            if not ds.is_classification:
                continue
            X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()
            if len(X_tr) > MAX_N:
                X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=MAX_N, stratify=y_tr)

            try:
                model = MLPEstimator(
                    num_classes=num_classes, model_args=dict(verbose=1, max_epochs=MAX_EPOCHS)
                )
                model.fit(X_train=X_tr, y_train=y_tr)
                train_loss = model.model.history[:, "train_loss"]
                if np.any(np.isnan(train_loss)):
                    raise ValueError("NaNs in loss")

                preds = model.predict(X_test)
                if preds.dtype != np.int64:
                    raise TypeError("`.predict()` is not returning int64")
                if len(preds.shape) != 1:
                    raise ValueError("`.predict()` is not flat.")

                probs = model.predict_proba(X_test)
                if probs.dtype not in [np.float64, np.float32]:
                    raise TypeError("`.predict_proba()` is not returning float")
                exp = (X_test.shape[0], num_classes)
                if probs.shape != exp:
                    raise ValueError(
                        "`.predict_proba()` is not returning correct shape. Expected: "
                        f"{exp} but got: {probs.shape}"
                    )
                totals = np.sum(probs, axis=1)
                ones = np.ones_like(totals)
                try:
                    np.testing.assert_almost_equal(totals, ones)
                except AssertionError:
                    raise ValueError(
                        "`.predict_proba()` is not returning valid probabilities. "
                        f"Got: {probs} and axis=1 totals {totals}"
                    )

            except Exception as e:
                raise RuntimeError(f"Got error for dataset {dsname}:") from e


def test_mlp_regressor(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        for dsname, ds in TEST_DATASETS.items():
            if ds.is_classification:
                continue
            X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()
            if len(X_tr) > MAX_N:
                X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=MAX_N)

            try:
                model = MLPEstimator(
                    num_classes=num_classes, model_args=dict(verbose=1, max_epochs=MAX_EPOCHS)
                )
                model.fit(X_train=X_tr, y_train=y_tr)
                train_loss = model.model.history[:, "train_loss"]
                if np.any(np.isnan(train_loss)):
                    raise ValueError("NaNs in loss")

                preds = model.predict(X_test)
                if preds.dtype not in [np.float32, np.float64]:
                    raise TypeError("`.predict()` is not returning float")
                if len(preds.shape) != 1:
                    raise ValueError("`.predict()` is not flat.")

            except Exception as e:
                raise RuntimeError(f"Got error for dataset {dsname}:") from e


def test_mlp_classifier_tune(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        for dsname, ds in TEST_DATASETS.items():
            if not ds.is_classification:
                continue
            X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()
            if len(X_tr) > MAX_N:
                X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=MAX_N, stratify=y_tr)

            try:
                model = MLPEstimator(
                    num_classes=num_classes, model_args=dict(verbose=1, max_epochs=MAX_EPOCHS)
                )
                model.htune_optuna(X_train=X_tr, y_train=y_tr, n_trials=10, n_jobs=1, verbosity=1)
                score = model.htune_eval(X_train=X_tr, y_train=y_tr, X_test=X_test, y_test=y_test)
                print(f"Tuned score: {score}")
            except Exception as e:
                raise RuntimeError(f"Got error tuning model on {dsname}") from e


def test_mlp_regressor_tune(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        for dsname, ds in TEST_DATASETS.items():
            if ds.is_classification:
                continue
            X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()
            if len(X_tr) > MAX_N:
                X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=MAX_N)

            try:
                model = MLPEstimator(
                    num_classes=num_classes, model_args=dict(verbose=1, max_epochs=MAX_EPOCHS)
                )
                model.htune_optuna(X_train=X_tr, y_train=y_tr, n_trials=10, n_jobs=1, verbosity=1)
                score = model.htune_eval(X_train=X_tr, y_train=y_tr, X_test=X_test, y_test=y_test)
                print(f"Tuned score: {score}")
            except Exception as e:
                raise RuntimeError(f"Got error tuning model on {dsname}") from e


def test_mlp_tune_parallel(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        for dsname, ds in TEST_DATASETS.items():
            X_tr, X_test, y_tr, y_test, num_classes = ds.train_test_split()
            if len(X_tr) > MAX_N:
                strat = y_tr if ds.is_classification else None
                X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=MAX_N, stratify=strat)

            try:
                model = MLPEstimator(
                    num_classes=num_classes, model_args=dict(verbose=1, max_epochs=MAX_EPOCHS)
                )
                model.htune_optuna(X_train=X_tr, y_train=y_tr, n_trials=10, n_jobs=-1, verbosity=1)
                score = model.htune_eval(X_train=X_tr, y_train=y_tr, X_test=X_test, y_test=y_test)
                print(f"Tuned score: {score}")
            except Exception as e:
                raise RuntimeError(f"Got error tuning model on {dsname}") from e
