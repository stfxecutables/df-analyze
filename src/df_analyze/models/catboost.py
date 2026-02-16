# ref: CatBoost paper (ordered boosting, categorical processing): https://arxiv.org/abs/1706.09516

from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import Any, Callable, Mapping, Optional, Type, Union

import numpy as np
import optuna
from optuna import Study, Trial
from pandas import DataFrame, Series

from df_analyze._constants import SEED
from df_analyze.enumerables import Scorer
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.splitting import OmniKFold

try:
    from catboost import CatBoostClassifier as CBCatBoostClassifier
    from catboost import CatBoostRegressor as CBCatBoostRegressor
except ImportError as exc:
    CBCatBoostClassifier = None
    CBCatBoostRegressor = None
    _CATBOOST_IMPORT_ERROR = exc
else:
    _CATBOOST_IMPORT_ERROR = None


class CatBoostEstimator(DfAnalyzeModel):
    shortname = "catboost"
    longname = "CatBoost Estimator"
    timeout_s = 30 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.needs_calibration = False
        self.model_cls: Type[Any] = type(None)
        self.target_cols: list[str] = []
        self.default_args = dict(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=SEED,
            allow_writing_files=False,
            verbose=0,
        )
        self.grid = None

    def _assert_available(self) -> None:
        if _CATBOOST_IMPORT_ERROR is not None:
            raise ImportError(
                "CatBoost is not installed. Install it with `pip install catboost`."
            ) from _CATBOOST_IMPORT_ERROR

    def _set_target_cols(self, y: Union[Series, DataFrame]) -> None:
        if isinstance(y, DataFrame):
            self.target_cols = [str(col) for col in y.columns]
            return
        name = y.name if y.name is not None else "target"
        self.target_cols = [str(name)]

    def _loss_for_target(self, ycol: Series) -> str:
        return "RMSE"

    def _loss_args_for_target(self, ycol: Series) -> dict[str, Any]:
        return {}

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            iterations=trial.suggest_int("iterations", 100, 400, step=50),
        )

    def _fit_single_target(
        self,
        X: DataFrame,
        y: Series,
        kwargs: Mapping[str, Any],
    ) -> Any:
        model_args = {**dict(kwargs), **self._loss_args_for_target(y)}
        model_cls, clean_args = self.model_cls_args(model_args)
        model = model_cls(**clean_args)
        model.fit(X, y)
        return model

    def _fit_target_models(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if isinstance(y, DataFrame):
            return {
                str(col): self._fit_single_target(X, y[col], kwargs)
                for col in self.target_cols
            }
        return self._fit_single_target(X, y, kwargs)

    def fit(self, X_train: DataFrame, y_train: Union[Series, DataFrame]) -> None:
        self._assert_available()
        self._set_target_cols(y_train)
        kwargs = {**self.fixed_args, **self.default_args, **self.model_args}
        self.model = self._fit_target_models(X=X_train, y=y_train, kwargs=kwargs)

    def refit_tuned(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> None:
        self._assert_available()
        self._set_target_cols(y)
        tuned_args = tuned_args or {}
        kwargs = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **tuned_args,
        }
        self.tuned_model = self._fit_target_models(X=X, y=y, kwargs=kwargs)
        self.tuned_args = dict(tuned_args)
        self.is_refit = True

    def _predict_values(self, model: Any, X: DataFrame) -> np.ndarray:
        return np.asarray(model.predict(X)).reshape(-1)

    def _predict_from_model(
        self, model_obj: Union[Any, dict[str, Any]], X: DataFrame
    ) -> Union[Series, DataFrame]:
        if isinstance(model_obj, dict):
            out = {target: self._predict_values(model_obj[target], X) for target in self.target_cols}
            return DataFrame(out, index=X.index)
        preds = self._predict_values(model_obj, X)
        name = self.target_cols[0] if len(self.target_cols) > 0 else None
        return Series(preds, name=name, index=X.index)

    def predict(self, X: DataFrame) -> Union[Series, DataFrame]:
        if self.model is None:
            raise RuntimeError("Need to call `model.fit()` before calling `.predict()`")
        return self._predict_from_model(self.model, X)

    def tuned_predict(self, X: DataFrame) -> Union[Series, DataFrame]:
        if self.tuned_model is None:
            raise RuntimeError(
                "Need to call `model.tune()` before calling `.tuned_predict()`"
            )
        return self._predict_from_model(self.tuned_model, X)

    def optuna_objective(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series],
        metric: Scorer,
        n_folds: int = 5,
    ) -> Callable[[Trial], float]:
        self._assert_available()
        y_df = y_train.to_frame() if isinstance(y_train, Series) else y_train
        y_split = self._split_target_for_cv(y_df)

        kf = OmniKFold(
            n_splits=n_folds,
            is_classification=self.is_classifier,
            grouped=g_train is not None,
            labels=None,
            warn_on_fallback=False,
            df_analyze_phase="Tuning internal splits",
        )
        splits, _ = kf.split(X_train=X_train, y_train=y_split, g_train=g_train)
        target_cols = [str(col) for col in y_df.columns]

        def objective(trial: Trial) -> float:
            opt_args = self.optuna_args(trial)
            full_args = {
                **self.fixed_args,
                **self.default_args,
                **self.model_args,
                **opt_args,
            }
            scores = []
            for step, (idx_train, idx_test) in enumerate(splits):
                X_tr = X_train.iloc[idx_train]
                X_te = X_train.iloc[idx_test]
                y_tr = y_df.iloc[idx_train]
                y_te = y_df.iloc[idx_test]

                if y_tr.shape[1] == 1:
                    y_tr_col = y_tr.iloc[:, 0]
                    y_te_col = y_te.iloc[:, 0]
                    model = self._fit_single_target(X_tr, y_tr_col, full_args)
                    preds = np.asarray(model.predict(X_te)).reshape(-1)
                    score = float(metric.tuning_score(y_te_col.to_numpy(), preds))
                else:
                    target_scores = []
                    for col in target_cols:
                        model = self._fit_single_target(X_tr, y_tr[col], full_args)
                        preds = np.asarray(model.predict(X_te)).reshape(-1)
                        target_scores.append(
                            float(metric.tuning_score(y_te[col].to_numpy(), preds))
                        )
                    score = float(np.nanmean(target_scores))

                scores.append(score)
                trial.report(float(np.mean(scores)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        return objective

    def htune_optuna(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series],
        metric: Scorer,
        n_trials: int = 100,
        n_jobs: int = -1,
        verbosity: int = optuna.logging.ERROR,
    ) -> Study:
        self._assert_available()
        return super().htune_optuna(
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            metric=metric,
            n_trials=n_trials,
            n_jobs=n_jobs,
            verbosity=verbosity,
        )


class CatBoostClassifier(CatBoostEstimator):
    shortname = "catboost"
    longname = "CatBoost Classifier"

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = CBCatBoostClassifier  # type: ignore 

    def _loss_for_target(self, ycol: Series) -> str:
        n_classes = int(ycol.nunique(dropna=False))
        return "Logloss" if n_classes <= 2 else "MultiClass"

    def predict_proba(
        self, X: DataFrame
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if self.tuned_model is None:
            raise RuntimeError(
                "Need to call `model.tune()` before calling `.predict_proba()`"
            )
        if isinstance(self.tuned_model, dict):
            return {
                target: np.asarray(self.tuned_model[target].predict_proba(X))
                for target in self.target_cols
            }
        return np.asarray(self.tuned_model.predict_proba(X))


class CatBoostRegressor(CatBoostEstimator):
    shortname = "catboost"
    longname = "CatBoost Regressor"

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = CBCatBoostRegressor  # type: ignore
