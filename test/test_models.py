from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import logging
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from typing import Literal, Union

import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from pandas import DataFrame, Series
from pytest import CaptureFixture
from sklearn.preprocessing import KBinsDiscretizer

from df_analyze.device import device_info_from_resolved
from df_analyze.enumerables import ClassifierScorer, RegressorScorer
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.models.dummy import DummyClassifier, DummyRegressor
from df_analyze.models.gandalf import LOGS, GandalfEstimator
from df_analyze.models.knn import KNNClassifier, KNNRegressor
from df_analyze.models.lgbm import (
    LightGBMClassifier,
    LightGBMRegressor,
    LightGBMRFClassifier,
    LightGBMRFRegressor,
)
from df_analyze.models.linear import (
    ElasticNetRegressor,
    LRClassifier,
    SGDClassifier,
    SGDRegressor,
)
from df_analyze.models.svm import SVMClassifier, SVMRegressor

C = 10


def fake_data(
    mode: Literal["classify", "regress"], noise: float = 1.0
) -> tuple[DataFrame, DataFrame, Series, Series]:
    N = 100
    X_cont_tr = np.random.standard_normal([N, C])
    X_cont_test = np.random.standard_normal([N, C])

    cat_sizes = np.random.randint(2, 5, C)
    cats_tr = [np.random.randint(0, c, N) for c in cat_sizes]
    cats_test = [np.random.randint(0, c, N) for c in cat_sizes]

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
        encoder.fit(np.concatenate([y_tr.reshape(-1, 1), y_test.reshape(-1, 1)]))
        y_tr = encoder.transform(y_tr.reshape(-1, 1))
        y_test = encoder.transform(y_test.reshape(-1, 1))

    target_tr = Series(np.asarray(y_tr).ravel(), name="target")
    target_test = Series(np.asarray(y_test).ravel(), name="target")

    return df_tr, df_test, target_tr, target_test


def check_basics(model: DfAnalyzeModel, mode: Literal["classify", "regress"]) -> None:
    X_tr, X_test, y_tr, y_test = fake_data(mode)
    try:
        model.fit(X_train=X_tr, y_train=y_tr)
        model.predict(X_test)
    except ValueError as e:  # handle braindead Optuna race condition
        print(e)
        print(e.args)
        if "No trials are completed yet" in str(e):
            pass
        elif "No trials are completed yet" in " ".join(e.args):
            pass
        else:
            traceback.print_exc()
            raise e


def check_optuna_tune_metric(
    model: DfAnalyzeModel,
    mode: Literal["classify", "regress"],
    metric: Union[ClassifierScorer, RegressorScorer],
) -> tuple[float, ndarray | Series | None]:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None  # noqa: F841
    X_tr, X_test, y_tr, y_test = fake_data(mode)
    const_target = (len(y_tr.unique()) == 1) or (len(y_test.unique()) == 1)
    while const_target:
        X_tr, X_test, y_tr, y_test = fake_data(mode)
        const_target = (len(y_tr.unique()) == 1) or (len(y_test.unique()) == 1)

    # metric = ClassifierScorer.default() if is_cls else RegressorScorer.default()
    model = deepcopy(model)
    try:
        study = model.htune_optuna(
            X_train=X_tr,
            y_train=y_tr,
            g_train=None,
            metric=metric,  # type: ignore
            # shitty Optuna implementation seems to dispatch as many jobs as cores,
            # even if you specify less trials, and but then also have some kind of
            # improper process.join() or other race condition so that it doesn't
            # properly wait for things to finish, resulting in an error like below:
            #
            # def get_best_trial(self, study_id: int) -> FrozenTrial:
            #     with self._lock:
            #         self._check_study_id(study_id)
            #
            #         best_trial_id = self._studies[study_id].best_trial_id
            #
            #         if best_trial_id is None:
            # >               raise ValueError("No trials are completed yet.")
            # E               ValueError: No trials are completed yet.
            #
            # These erors also seem to be uncatchable (raised by some sub-process)
            # so the only way to ignore the issue is to use 1 job for testing, but
            # then of course we aren't testing the paralellism, which is kinf of
            # the whole point. Not sure why Optuna sucks so hard.
            #
            # n_trials=40 if ON_CLUSTER else 8,
            n_trials=4,
            n_jobs=1,
        )

        if not hasattr(study, "best_params"):
            raise RuntimeError("No trials ever ran for some reason.")
        overrides = study.best_params
        print(overrides)
    except ValueError as e:  # handle braindead Optuna race condition
        msg = (
            f"Got error for metric: {metric.name}. Targets:\n"
            f"y_tr unique values: {np.unique(y_tr, return_counts=True)}\n"
            f"{y_tr}\n"
            f"y_test unique values: {np.unique(y_test, return_counts=True)}\n"
            f"{y_test}\n"
        )
        print(e)
        if "No trials are completed yet" in str(e):
            raise ValueError(
                f"No trials completed by Optuna for some reason. Info:\n{msg}"
            )
        raise ValueError(msg) from e
    # print(f"Best params: {overrides}")
    model.refit_tuned(X_tr, y_tr, tuned_args=overrides)
    preds = model.tuned_predict(X_test)
    try:
        score = metric.tuning_score(y_true=y_test, y_pred=preds)
    except Exception:
        raise ValueError(
            "Could not get score on final generated test data. Maybe all same class?"
        )
    return score, preds


def check_optuna_tune(
    model: DfAnalyzeModel,
    mode: Literal["classify", "regress"],
) -> None:
    is_cls = model.is_classifier
    metrics = ClassifierScorer if is_cls else RegressorScorer
    # metric = ClassifierScorer.default() if is_cls else RegressorScorer.default()
    for metric in metrics:
        print(
            metric.name,
            check_optuna_tune_metric(model=model, mode=mode, metric=metric)[0],
        )


@pytest.mark.fast
class TestDummy:
    def test_dummy_cls(self) -> None:
        model = DummyClassifier()
        check_basics(model, "classify")

    def test_dummy_reg(self) -> None:
        model = DummyRegressor()
        check_basics(model, "regress")

    def test_dummy_cls_tune(self, capsys: CaptureFixture) -> None:
        logging.captureWarnings(capture=True)
        model = DummyClassifier()
        check_optuna_tune(model, "classify")

    def test_dummy_reg_tune(self, capsys: CaptureFixture) -> None:
        model = DummyRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestLinear:
    def test_lin_cls(self) -> None:
        model = LRClassifier()
        check_basics(model, "classify")

    def test_lin_reg(self) -> None:
        model = ElasticNetRegressor()
        check_basics(model, "regress")

    def test_lin_cls_tune(self, capsys: CaptureFixture) -> None:
        logging.captureWarnings(capture=True)
        logger = logging.getLogger("py.warnings")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())
        try:
            model = LRClassifier()
            check_optuna_tune(model, "classify")
        except Exception as e:
            raise e
        finally:
            logging.captureWarnings(capture=False)

    def test_lin_reg_tune(self, capsys: CaptureFixture) -> None:
        model = ElasticNetRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestSGDLinear:
    def test_sgd_cls(self) -> None:
        model = SGDClassifier()
        check_basics(model, "classify")

    def test_sgd_reg(self) -> None:
        model = SGDRegressor()
        check_basics(model, "regress")

    def test_sgd_cls_tune(self, capsys: CaptureFixture) -> None:
        logging.captureWarnings(capture=True)
        logger = logging.getLogger("py.warnings")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())
        try:
            model = SGDClassifier()
            check_optuna_tune(model, "classify")
        except Exception as e:
            raise e
        finally:
            logging.captureWarnings(capture=False)

    def test_sgd_reg_tune(self, capsys: CaptureFixture) -> None:
        model = SGDRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestKNN:
    def test_knn_cls(self) -> None:
        model = KNNClassifier()
        check_basics(model, "classify")

    def test_knn_reg(self) -> None:
        model = KNNRegressor()
        check_basics(model, "regress")

    def test_knn_cls_tune(self, capsys: CaptureFixture) -> None:
        model = KNNClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_knn_reg_tune(self, capsys: CaptureFixture) -> None:
        model = KNNRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestSVM:
    def test_svm_cls(self) -> None:
        model = SVMClassifier()
        check_basics(model, "classify")

    def test_svm_reg(self) -> None:
        model = SVMRegressor()
        check_basics(model, "regress")

    def test_svm_cls_tune(self, capsys: CaptureFixture) -> None:
        model = SVMClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_svm_reg_tune(self, capsys: CaptureFixture) -> None:
        model = SVMRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestLightGBM:
    def test_lgbm_cls(self) -> None:
        model = LightGBMClassifier()
        check_basics(model, "classify")

    def test_lgbm_reg(self) -> None:
        model = LightGBMRegressor()
        check_basics(model, "regress")

    def test_lgbm_rf_cls(self) -> None:
        model = LightGBMRFClassifier()
        check_basics(model, "classify")

    def test_lgbm_rf_reg(self) -> None:
        model = LightGBMRFRegressor()
        check_basics(model, "regress")

    def test_lgbm_cls_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_lgbm_reg_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")

    def test_lgbm_rf_cls_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMRFClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_lgbm_rf_reg_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMRFRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestGandalf:
    def test_cuda_loader_defaults_keep_training_batching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import df_analyze.models.gandalf as gandalf_module

        monkeypatch.delenv(gandalf_module.GANDALF_LOADER_WORKERS_ENV, raising=False)
        monkeypatch.delenv(gandalf_module.GANDALF_PIN_MEMORY_ENV, raising=False)

        model = GandalfEstimator(num_classes=C)
        model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

        loader = model._pred_loader(DataFrame(np.zeros((1024, 4))), g=None)
        assert loader.batch_size == gandalf_module.BATCH
        assert model._loader_kwargs() == {
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": True,
        }

    def test_small_train_split_keeps_a_training_batch(self) -> None:
        model = GandalfEstimator(num_classes=2)
        X = DataFrame(np.zeros((20, 4)))
        y = Series([0, 1] * 10)

        train_loader, val_loader = model._train_val_loaders(X, y)

        assert len(train_loader) > 0
        assert train_loader.batch_size <= len(train_loader.dataset)  # type: ignore[arg-type]
        assert val_loader.batch_size <= len(val_loader.dataset)  # type: ignore[arg-type]

    def test_pred_loader_rejects_empty_data(self) -> None:
        model = GandalfEstimator(num_classes=2)

        with pytest.raises(ValueError, match="empty dataset"):
            model._pred_loader(DataFrame(np.empty((0, 4))), g=None)

    def test_classification_tuning_scores_probabilities(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        class FakeModel:
            def __init__(self, **kwargs: object) -> None:
                return None

        class FakeTrainer:
            log_dir = None

            def fit(self, **kwargs: object) -> None:
                return None

        class ValDataset:
            y = torch.tensor([0, 1])

            def __len__(self) -> int:
                return 2

            def __getitem__(self, idx: int) -> torch.Tensor:
                return torch.tensor([float(idx)])

        class Metric:
            value = "score"

            def get_scores(
                self, y_true: Series, y_pred: Series, y_prob: np.ndarray
            ) -> dict[str, float]:
                assert np.allclose(y_prob.sum(axis=1), 1.0)
                assert list(y_pred) == [0, 1]
                return {"score": 0.75}

            def higher_is_better(self) -> bool:
                return True

        train = DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)
        val = DataLoader(ValDataset(), batch_size=1)
        model = GandalfEstimator(num_classes=2)
        model.model_cls = FakeModel  # type: ignore[assignment]
        monkeypatch.setattr(model, "optuna_args", lambda trial: {})
        monkeypatch.setattr(model, "_train_val_loaders", lambda **kwargs: (train, val))
        monkeypatch.setattr(model, "_get_trainer", lambda **kwargs: FakeTrainer())
        monkeypatch.setattr(model, "_load_best_checkpoint", lambda trainer, model: None)
        monkeypatch.setattr(
            model,
            "_predict_logits",
            lambda model, loader: torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
        )

        objective = model.optuna_objective(
            DataFrame(np.zeros((2, 1))),
            Series([0, 1]),
            None,
            Metric(),  # type: ignore[arg-type]
        )

        assert objective(object()) == 0.75  # type: ignore[arg-type]

    def test_cuda_virtual_batch_choices_match_cpu_defaults(self) -> None:
        class TrialStub:
            def __init__(self) -> None:
                self.choices: dict[str, list[int]] = {}

            def suggest_float(self, name: str, low: float, high: float, **kwargs) -> float:
                return low

            def suggest_categorical(self, name: str, choices: list) -> object:
                self.choices[name] = list(choices)
                return choices[0]

        trial = TrialStub()
        model = GandalfEstimator(num_classes=C)
        model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

        model.optuna_args(trial)  # type: ignore[arg-type]

        assert trial.choices["virtual_batch"] == [8, 16, 32, 64]

    def test_cuda_loader_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import df_analyze.models.gandalf as gandalf_module

        monkeypatch.setenv(gandalf_module.GANDALF_LOADER_WORKERS_ENV, "2")
        monkeypatch.setenv(gandalf_module.GANDALF_PIN_MEMORY_ENV, "1")

        model = GandalfEstimator(num_classes=C)
        model.set_device_info(device_info_from_resolved("auto", "cuda", torch_cuda_available=True, cuda_driver_available=True))

        assert model._loader_kwargs() == {
            "num_workers": 2,
            "persistent_workers": True,
            "pin_memory": True,
        }

    def test_checkpoint_uses_private_dir_and_metric_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import df_analyze.models.gandalf as gandalf_module
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        checkpoint_kwargs: list[dict] = []
        trainer_kwargs: list[dict] = []

        class FakeCheckpoint:
            def __init__(self, **kwargs: object) -> None:
                checkpoint_kwargs.append(dict(kwargs))

        class FakeEarlyStopping:
            def __init__(self, **kwargs: object) -> None:
                pass

        class FakeTrainer:
            def __init__(self, **kwargs: object) -> None:
                trainer_kwargs.append(dict(kwargs))

        monkeypatch.setattr(gandalf_module, "LOGS", tmp_path / "gandalf_logs")
        monkeypatch.setattr(gandalf_module, "ModelCheckpoint", FakeCheckpoint)
        monkeypatch.setattr(gandalf_module, "LightningEarlyStopping", FakeEarlyStopping)
        monkeypatch.setattr(gandalf_module, "Trainer", FakeTrainer)

        data = TensorDataset(torch.arange(4.0).reshape(4, 1), torch.arange(4.0))
        train = DataLoader(data, batch_size=2)
        val = DataLoader(data, batch_size=2)

        GandalfEstimator(num_classes=C)._get_trainer(train=train, val=val)
        GandalfEstimator(num_classes=1)._get_trainer(train=train, val=val)

        cls_checkpoint, reg_checkpoint = checkpoint_kwargs
        assert cls_checkpoint["monitor"] == "val/metric"
        assert cls_checkpoint["mode"] == "max"
        assert cls_checkpoint["save_top_k"] == 1
        assert cls_checkpoint["save_weights_only"] is True
        assert cls_checkpoint["dirpath"].parent == (
            tmp_path / "gandalf_logs" / "checkpoints"
        )
        assert reg_checkpoint["mode"] == "min"
        assert len(trainer_kwargs) == 2
        assert trainer_kwargs[0]["num_sanity_val_steps"] == 0

    def test_gandalf_cls(self, capsys: CaptureFixture) -> None:
        try:
            model = GandalfEstimator(num_classes=C)
            with capsys.disabled():
                check_basics(model, "classify")
        except Exception as e:
            raise e
        finally:
            rmtree(LOGS, ignore_errors=True)

    def test_gandalf_reg(self, capsys: CaptureFixture) -> None:
        try:
            model = GandalfEstimator(num_classes=1)
            with capsys.disabled():
                check_basics(model, "regress")
        except Exception as e:
            raise e
        finally:
            rmtree(LOGS, ignore_errors=True)

    def test_gandalf_cls_tune(self, capsys: CaptureFixture) -> None:
        try:
            model = GandalfEstimator(num_classes=C)
            with capsys.disabled():
                check_optuna_tune(model, "classify")
        except Exception as e:
            raise e
        finally:
            rmtree(LOGS, ignore_errors=True)

    def test_gandalf_reg_tune(self, capsys: CaptureFixture) -> None:
        try:
            model = GandalfEstimator(num_classes=1)
            with capsys.disabled():
                check_optuna_tune(model, "regress")
        except Exception as e:
            raise e
        finally:
            rmtree(LOGS, ignore_errors=True)
