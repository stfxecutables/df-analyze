from __future__ import annotations

import json
import os
import re
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from pprint import pprint
from random import choice, randint
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)
from warnings import warn

import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
)

from df_analyze._constants import SEED, VAL_SIZE
from df_analyze._types import Classifier, CVMethod, EstimationMode, Estimator
from df_analyze.cli.cli import ProgramOptions
from df_analyze.device import cleanup_accelerator_memory, is_fatal_accelerator_error
from df_analyze.legacy.src.objectives import (
    bagging_classifier_objective,
    dtree_classifier_objective,
    mlp_classifier_objective,
    rf_classifier_objective,
    svm_classifier_objective,
)
from df_analyze.models.dummy import DummyClassifier, DummyRegressor
from df_analyze.testing.datasets import TestDataset

if TYPE_CHECKING:
    from df_analyze.models.base import DfAnalyzeModel
import jsonpickle

from df_analyze.enumerables import ClassifierScorer, FeatureSelection, RegressorScorer
from df_analyze.preprocessing.prepare import PreparedData
from df_analyze.saving import add_fold_idx
from df_analyze.scoring import (
    sensitivity,
    specificity,
)
from df_analyze.selection.embedded import EmbedSelectionModel
from df_analyze.selection.filter import FilterSelected
from df_analyze.selection.models import ModelSelected

Splits = Iterable[Tuple[ndarray, ndarray]]

LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_LAYER_SIZES = [4, 8, 16, 32]
N_SPLITS = 5
TABPFN_UNKNOWN_CATEGORY = "__df_analyze_unknown__"


def _model_shortname(model_or_cls: Any) -> str:
    name = getattr(model_or_cls, "shortname", None)
    if name is None:
        name = getattr(model_or_cls, "__name__", "")
    return str(name).lower()


def _is_tabpfn_weights_error(error: BaseException) -> bool:
    return error.__class__.__name__ == "TabPFNLocalWeightsError"


def _backend_looks_accelerated(backend: Any) -> bool:
    text = str(backend).lower()
    return any(marker in text for marker in ("cuda", "gpu", "cuml", "mps"))


def _actual_backend_device(backend: Any, accelerated: bool) -> str:
    if not accelerated:
        return "cpu"

    text = str(backend).lower()
    if "mps" in text:
        return "mps"
    if any(marker in text for marker in ("cuda", "gpu", "cuml")):
        return "cuda"
    return "accelerated"


def _device_result_fields(
    model: DfAnalyzeModel,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    info = model.device_info
    resolved = info.resolved
    backend = model.backend_name(params)
    accelerated = bool(model.uses_accelerated_backend(params))
    # The backend string is the authoritative model implementation label; the
    # device field is a compact grouping inferred from that backend.
    return {
        "device": _actual_backend_device(backend, accelerated),
        "requested_device": info.requested,
        "resolved_device": resolved,
        "torch_device": info.torch_device,
        "accelerated": accelerated,
        "backend": backend,
    }


@dataclass
class HtuneResult:
    selection: Literal["none", "assoc", "pred", "embed", "wrap"]
    selected_cols: list[str]
    embed_select_model: Optional[EmbedSelectionModel]
    model_cls: Type[DfAnalyzeModel]
    model: DfAnalyzeModel
    params: dict[str, Any]  # optuna best_params
    metric: Union[ClassifierScorer, RegressorScorer]
    score: float
    preds_test: Union[Series, ndarray]
    preds_train: Union[Series, ndarray]
    probs_test: Optional[ndarray]
    probs_train: Optional[ndarray]
    device: str = "cpu"
    requested_device: str = "cpu"
    resolved_device: str = "cpu"
    torch_device: str = "cpu"
    accelerated: bool = False
    backend: str = "sklearn-cpu"
    model_version: str = ""
    tune_seconds: float = 0.0
    eval_seconds: float = 0.0
    total_seconds: float = 0.0

    @no_type_check  # Pyright mucks up the conditionals here...
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HtuneResult):
            return False

        def array_equal(left: Any, right: Any) -> bool:
            return bool(
                np.array_equal(
                    np.asarray(left).round(8),
                    np.asarray(right).round(8),
                )
            )

        if (self.probs_test is None) and (other.probs_test is None):
            probs_test_equal = True
        elif self.probs_test is not None and other.probs_test is not None:
            probs_test_equal = array_equal(self.probs_test, other.probs_test)
        else:
            probs_test_equal = False

        if (self.probs_train is None) and (other.probs_train is None):
            probs_train_equal = True
        elif self.probs_train is not None and other.probs_train is not None:
            probs_train_equal = array_equal(self.probs_train, other.probs_train)
        else:
            probs_train_equal = False

        probs_equal = probs_test_equal and probs_train_equal

        ret = (
            self.selection == other.selection
            and sorted(self.selected_cols) == sorted(other.selected_cols)
            and self.model_cls == other.model_cls
            and self.embed_select_model == other.embed_select_model
            and
            # self.model == other.model and  # NOTE: do not check since weights
            self.params == other.params
            and round(self.score, 8) == round(other.score, 8)
            and self.metric == other.metric
            and self.model_version == other.model_version
            and array_equal(self.preds_test, other.preds_test)
            and array_equal(self.preds_train, other.preds_train)
            and probs_equal
        )
        return bool(ret)

    def to_row(self) -> DataFrame:
        selector = self.embed_select_model
        embed = "none" if selector is None else selector.value
        return DataFrame(
            {
                "selection": self.selection,
                "embed_selector": embed,
                "model": self.model.shortname,
                "n_selected_features": len(self.selected_cols),
                "params": str(jsonpickle.encode(self.params)),
                "metric": self.metric.value,
                "score": self.score,
                "device": self.device,
                "requested_device": self.requested_device,
                "resolved_device": self.resolved_device,
                "torch_device": self.torch_device,
                "accelerated": self.accelerated,
                "backend": self.backend,
                "model_version": self.model_version,
                "tune_seconds": self.tune_seconds,
                "eval_seconds": self.eval_seconds,
                "total_seconds": self.total_seconds,
            },
            index=[0],
        )

    def to_preds_json(self) -> str:
        if self.probs_test is None:
            probs_test = None
            probs_dtype = None
        else:
            probs_test = self.probs_test.tolist()
            probs_dtype = str(self.probs_test.dtype)

        if self.probs_train is None:
            probs_train = None
        else:
            probs_train = self.probs_train.tolist()
        preds_dtype = str(self.preds_train.dtype)
        if isinstance(self.preds_test, Series):
            preds_test = self.preds_test.to_list()
        else:
            preds_test = self.preds_test.tolist()
        if isinstance(self.preds_train, Series):
            preds_train = self.preds_train.to_list()
        else:
            preds_train = self.preds_train.tolist()
        selector = (
            None if self.embed_select_model is None else self.embed_select_model.value
        )
        obj = {
            "selection": self.selection,
            "selected_cols": self.selected_cols,
            "n_selected_features": len(self.selected_cols),
            "embed_select_model": selector,
            "model_cls": self.model_cls.__name__,
            "params": str(self.params),
            "metric": self.metric.name,
            "score": self.score,
            "device": self.device,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "torch_device": self.torch_device,
            "accelerated": self.accelerated,
            "backend": self.backend,
            "model_version": self.model_version,
            "tune_seconds": self.tune_seconds,
            "eval_seconds": self.eval_seconds,
            "total_seconds": self.total_seconds,
            "preds_test": preds_test,
            "preds_train": preds_train,
            "probs_test": probs_test,
            "probs_train": probs_train,
            "preds_dtype": preds_dtype,
            "probs_dtype": probs_dtype,
        }
        return json.dumps(obj)

    def to_preds(self) -> PredResult:
        embed_model = (
            self.embed_select_model.value
            if self.embed_select_model is not None
            else "none"
        )
        return PredResult(
            selection=self.selection,
            selected_cols=self.selected_cols,
            embed_select_model=embed_model,
            model_cls=str(self.model_cls.__name__),
            params=str(self.params),
            metric=self.metric.name,
            score=self.score,
            device=self.device,
            requested_device=self.requested_device,
            resolved_device=self.resolved_device,
            torch_device=self.torch_device,
            accelerated=self.accelerated,
            backend=self.backend,
            model_version=self.model_version,
            tune_seconds=self.tune_seconds,
            eval_seconds=self.eval_seconds,
            total_seconds=self.total_seconds,
            preds_test=self.preds_test,
            preds_train=self.preds_train,
            probs_test=self.probs_test,
            probs_train=self.probs_train,
        )

    def to_summary_dict(self) -> dict[str, Any]:
        selector = (
            None if self.embed_select_model is None else self.embed_select_model.value
        )
        return {
            "selection": self.selection,
            "selected_cols": self.selected_cols,
            "n_selected_features": len(self.selected_cols),
            "embed_select_model": selector,
            "model_cls": self.model_cls.__name__,
            "model": self.model.shortname,
            "params": str(self.params),
            "metric": self.metric.name,
            "score": self.score,
            "device": self.device,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "torch_device": self.torch_device,
            "accelerated": self.accelerated,
            "backend": self.backend,
            "model_version": self.model_version,
            "tune_seconds": self.tune_seconds,
            "eval_seconds": self.eval_seconds,
            "total_seconds": self.total_seconds,
        }

    @staticmethod
    def all_from_json(root: Path) -> HtuneResult:
        raise NotImplementedError(
            "This should be handled by src.df_analyze.hypertune.py::EvaluationResults"
        )

    @staticmethod
    def random(
        ds: TestDataset,
        y_test: Series,
        selection: Optional[Literal["none", "assoc", "pred", "embed", "wrap"]] = None,
        selected_cols: Optional[list[str]] = None,
        model_cls: Optional[Type[DfAnalyzeModel]] = None,
    ) -> HtuneResult:
        from df_analyze.models.base import DfAnalyzeModel

        n_cols = ds.shape[1]
        is_cls = ds.is_classification
        n_cls = 1 if not ds.is_classification else len(np.unique(y_test))
        n_samp = ds.shape[0]
        n_sel = randint(1, n_cols)

        selection = selection or choice(["none", "assoc", "pred", "embed", "wrap"])
        selected = (
            selected_cols
            or np.random.choice([f"f{i}" for i in range(n_cols)], size=n_sel).tolist()
        )
        model_cls = model_cls or DfAnalyzeModel.random(ds, n=1)
        if _model_shortname(model_cls) in {"mlp", "gandalf"}:
            prep = ds.prepared(load_cached=True)
            model = model_cls(num_classes=prep.num_classes)  # type: ignore
        else:
            model = model_cls()
        params = {**model.fixed_args, **model.default_args}
        # AUROC is handled automatically and changed to BalancedAcc with a warning
        metric = ClassifierScorer.random() if is_cls else RegressorScorer.random()
        score = (
            np.random.uniform(0, 1) if ds.is_classification else np.random.uniform(0, 10)
        )
        n_samp_test = len(y_test)
        n_samp_train = n_samp - n_samp_test
        if is_cls:
            probs_train = softmax(
                np.random.standard_normal([n_samp_train, n_cls]), axis=1
            )
            probs_test = softmax(np.random.standard_normal([n_samp_test, n_cls]), axis=1)
            preds_test = Series(np.argmax(probs_test, axis=1))
            preds_train = Series(np.argmax(probs_train, axis=1))
        else:
            probs_test = None
            probs_train = None
            preds_test = Series(np.random.standard_normal([n_samp_test]))
            preds_train = Series(np.random.standard_normal([n_samp_train]))

        return HtuneResult(
            selection=selection,  # type: ignore
            selected_cols=selected,
            embed_select_model=EmbedSelectionModel.random(),
            model_cls=model_cls,
            model=model,
            params=params,
            metric=metric,
            score=score,
            preds_test=preds_test,
            preds_train=preds_train,
            probs_test=probs_test,
            probs_train=probs_train,
            **_device_result_fields(model, params),
            model_version=model.model_version(),
            tune_seconds=0.0,
            eval_seconds=0.0,
            total_seconds=0.0,
        )


@dataclass
class PredResult:
    selection: Literal["none", "assoc", "pred", "embed", "wrap"]
    selected_cols: list[str]
    embed_select_model: str
    model_cls: str
    params: str  # optuna best_params
    metric: str
    score: float
    preds_test: Union[Series, ndarray]
    preds_train: Union[Series, ndarray]
    probs_test: Optional[ndarray]
    probs_train: Optional[ndarray]
    device: str = "cpu"
    requested_device: str = "cpu"
    resolved_device: str = "cpu"
    torch_device: str = "cpu"
    accelerated: bool = False
    backend: str = "sklearn-cpu"
    model_version: str = ""
    tune_seconds: float = 0.0
    eval_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass
class EvaluationResults:
    """
    df: DataFrame
        Has columns: "model", "selection"
    """

    df: DataFrame
    X_train: DataFrame
    y_train: Series
    X_test: DataFrame
    y_test: Series
    results: list[HtuneResult]
    is_classification: bool

    @staticmethod
    def random(ds: TestDataset, options: ProgramOptions) -> EvaluationResults:
        from df_analyze.models.base import DfAnalyzeModel

        prep = ds.prepared(load_cached=True)
        prep_train, prep_test = prep.split()

        # Build up list of results with random HtuneResult objects
        selectors = ["assoc", "pred", "embed", "wrap"]
        n_select = np.random.randint(1, len(selectors))
        selected = np.random.choice(selectors, replace=False, size=n_select).tolist()
        selected = ["none"] + selected

        n_model = np.random.randint(2, 5, size=1).item()
        dummy = DummyClassifier if ds.is_classification else DummyRegressor
        rands = DfAnalyzeModel.random(ds, n=n_model)
        models_clses = list(set([dummy, *rands]))

        features = prep.X.columns.to_list()

        results = []
        dfs = []
        for selection in selected:
            for model_cls in models_clses:
                p = prep.X.shape[1]
                n_sel = np.random.randint(1, p)
                selected_cols = np.random.choice(features, size=n_sel, replace=False)
                selected_cols = selected_cols.tolist()
                result = HtuneResult.random(
                    ds=ds,
                    y_test=prep_test.y,
                    selection=selection,
                    selected_cols=selected_cols,
                    model_cls=model_cls,
                )
                df = result.to_row()
                results.append(result)
                dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)

        return EvaluationResults(
            df=df,
            X_train=prep_train.X,
            y_train=prep_train.y,
            X_test=prep_test.X,
            y_test=prep_test.y,
            results=results,
            is_classification=ds.is_classification,
        )

    def hp_table(self, fold_idx: Optional[int] = None) -> DataFrame:
        dfs = []
        for res in self.results:
            dfs.append(res.to_row())
        df = pd.concat(dfs, axis=0, ignore_index=True)
        if fold_idx is not None:
            df["test_idx"] = fold_idx
        return df

    def wide_table(
        self, valset: Literal["5-fold", "trainset", "holdout"] = "5-fold"
    ) -> DataFrame:
        cols = ["trainset", "holdout", "5-fold"]
        cols.remove(valset)
        col = valset
        idx = ~self.df.mean(axis=1, numeric_only=True).isna()
        if "metric" in self.df.columns:
            idx = idx & self.df["metric"].notna()
        df = (
            self.df.loc[idx]
            .drop(columns=cols)
            .pivot(
                columns="metric",
                values=col,
                index=["model", "selection", "embed_selector"],
            )
            .reset_index()
        )
        sorter = "acc" if self.is_classification else "mae"
        ascending = not self.is_classification
        if "embed_selector" in df.columns:
            df.drop(columns="embed_selector")
        return df.sort_values(by=sorter, ascending=ascending)

    def to_markdown(self) -> str:
        df_train = self.wide_table("trainset")
        df_hold = self.wide_table("holdout")
        df_fold = self.wide_table("5-fold")
        tab_train = df_train.to_markdown(tablefmt="simple", floatfmt="0.3f", index=False)
        tab_hold = df_hold.to_markdown(tablefmt="simple", floatfmt="0.3f", index=False)
        tab_fold = df_fold.to_markdown(tablefmt="simple", floatfmt="0.3f", index=False)
        text = (
            "# Final Model Performances\n\n"
            "## Training set performance\n\n"
            f"{tab_train}\n\n"
            "## Holdout set performance\n\n"
            f"{tab_hold}\n\n"
            "## 5-fold performance on holdout set\n\n"
            f"{tab_fold}\n\n"
        )
        return text

    def to_json(self) -> str:
        # obj = {
        #     "df": self.df.to_json()
        #     "X_train": DataFrame
        #     "y_train": Series
        #     "X_test": DataFrame
        #     "y_test": Series
        #     "results": list[HtuneResult]
        #     "is_classification": bool
        # }
        # raise NotImplementedError()
        return str(jsonpickle.encode(self))

    def save(self, root: Path, fold_idx: Optional[int]) -> None:
        fix = partial(add_fold_idx, fold_idx=fold_idx)
        self.df.to_csv(fix(root / "performance_long_table.csv"), index=False)
        self.X_train.to_csv(fix(root / "X_train.csv"), index=False)
        self.X_test.to_csv(fix(root / "X_test.csv"), index=False)
        self.y_train.to_frame().to_csv(fix(root / "y_train.csv"), index=True)
        self.y_test.to_frame().to_csv(fix(root / "y_test.csv"), index=True)

        prediction_payload = {
            "predictions": [
                json.loads(result.to_preds_json()) for result in self.results
            ]
        }
        results_json = json.dumps(prediction_payload)
        try:
            fix(root / "prediction_results.json").write_text(results_json)
        except Exception as e:
            traceback.print_exc()
            print(e)
            print(
                "Error saving manual results .json. Information above.", file=sys.stderr
            )

        remain = {
            "results": [result.to_summary_dict() for result in self.results],
            "is_classification": self.is_classification,
        }
        try:
            enc = str(jsonpickle.encode(remain, unpicklable=True))
            fix(root / "eval_htune_results_jsonpickle.json").write_text(enc)
        except Exception as e:
            try:
                enc = str(
                    jsonpickle.encode(
                        remain,
                        unpicklable=False,
                        fail_safe=str,
                    )
                )
                fix(root / "eval_htune_results_jsonpickle.json").write_text(enc)
            except Exception as fallback_error:
                traceback.print_exc()
                print(fallback_error)
                print(
                    "Error saving lightweight evaluation metadata. "
                    "Prediction JSON was written first when possible.",
                    file=sys.stderr,
                )
            else:
                print(e, file=sys.stderr)

        for result in self.results:
            result.model.model = None
            result.model.tuned_model = None
            cleanup_accelerator_memory(result.model.device_info)

    @classmethod
    def load(cls, root: Path) -> EvaluationResults:
        # TODO: handle fold_idx
        df = pd.read_csv(root / "performance_long_table.csv")
        X_train = pd.read_csv(root / "X_train.csv", engine="python")
        X_test = pd.read_csv(root / "X_test.csv", engine="python")
        df_y_tr = pd.read_csv(root / "y_train.csv", index_col=0)
        df_y_test = pd.read_csv(root / "y_test.csv", index_col=0)

        y_train = Series(data=df_y_tr.values.ravel(), name=df_y_tr.columns.to_list()[0])
        y_test = Series(
            data=df_y_test.values.ravel(), name=df_y_test.columns.to_list()[0]
        )
        enc = (root / "eval_htune_results_jsonpickle.json").read_text()
        remain = cast(dict[str, Any], jsonpickle.decode(enc))
        return EvaluationResults(
            df=df,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            results=remain["results"],
            is_classification=remain["is_classification"],
        )

    @classmethod
    def load_preds(cls, root: Path) -> list[PredResult]:
        path = root / "prediction_results.json"
        results: dict[Literal["predictions"], list[dict[str, Any]]] = json.loads(
            path.read_text()
        )
        pred_results = []
        for result in results["predictions"]:
            preds_dtype = result["preds_dtype"]
            probs_dtype = result["probs_dtype"]
            backend = result.get("backend", "sklearn-cpu")
            resolved_device = str(
                result.get("resolved_device", result.get("device", "cpu"))
            )
            requested_device = str(result.get("requested_device", resolved_device))
            torch_device = str(result.get("torch_device", result.get("device", "cpu")))
            if result["probs_test"] is None:
                probs_test = None
            else:
                probs_test = np.array(result["probs_test"], dtype=probs_dtype)
            if result["probs_train"] is None:
                probs_train = None
            else:
                probs_train = np.array(result["probs_train"], dtype=probs_dtype)

            pred_result = PredResult(
                selection=result["selection"],
                selected_cols=result["selected_cols"],
                embed_select_model=result["embed_select_model"],
                model_cls=result["model_cls"],
                params=result["params"],
                metric=result["metric"],
                score=result["score"],
                preds_test=Series(result["preds_test"], dtype=preds_dtype),
                preds_train=Series(result["preds_train"], dtype=preds_dtype),
                probs_test=probs_test,
                probs_train=probs_train,
                device=result.get("device", resolved_device),
                requested_device=requested_device,
                resolved_device=resolved_device,
                torch_device=torch_device,
                accelerated=bool(
                    result.get("accelerated", _backend_looks_accelerated(backend))
                ),
                backend=backend,
                model_version=str(result.get("model_version", "")),
                tune_seconds=float(result.get("tune_seconds", 0.0)),
                eval_seconds=float(result.get("eval_seconds", 0.0)),
                total_seconds=float(result.get("total_seconds", 0.0)),
            )
            pred_results.append(pred_result)
        return pred_results


def _get_cols(
    selection: Union[Literal["none", "assoc", "pred", "embed", "wrap"], str],
    selected: Optional[list[str]],
) -> Union[list[str], slice]:
    if selection == "none" or (selected is None):
        return slice(None)

    if selection == "assoc":
        # dealing with level-specific metrics
        selected = sorted(set([re.sub("__.*", "", str(c)) for c in selected]))
    return selected


def _get_splits(
    prep_train: PreparedData,
    prep_test: PreparedData,
    selection: str,
    cols: Union[list[str], slice],
) -> tuple[DataFrame, DataFrame]:
    return _subset_splits(prep_train.X, prep_test.X, selection, cols)


def _subset_splits(
    train: DataFrame,
    test: DataFrame,
    selection: str,
    cols: Union[list[str], slice],
) -> tuple[DataFrame, DataFrame]:
    if selection == "none" or isinstance(cols, slice):
        X_train = train
        X_test = test
    else:
        # clunky way to deal with renames and indicators
        # should be safe since we sanitize names
        try:
            X_train = train.filter(regex="|".join(cols))
            X_test = test.filter(regex="|".join(cols))
        except Exception:
            matched = []
            for col in cols:  # may be e.g. colname__target.1
                col_name = str(col)
                for c in train.columns:
                    c_name = str(c)
                    cname = os.path.commonprefix([col_name, c_name])
                    if cname == "":
                        continue
                    if cname in [col_name, c_name] and c not in matched:
                        matched.append(c)
            X_train = train.loc[:, matched]
            X_test = test.loc[:, [c for c in matched if c in test.columns]]

    return X_train, X_test


def _tabpfn_continuous_features(prep: PreparedData) -> DataFrame:
    empty = DataFrame(index=prep.X.index)
    cont_cols = [] if prep.X_cont is None else prep.X_cont.columns.to_list()
    indicator_cols = [
        col
        for col in prep.X.columns
        if str(col).endswith("_NAN") and col not in cont_cols
    ]
    ordered_cols = [*cont_cols, *indicator_cols]
    if len(ordered_cols) == 0:
        return empty
    # Prefer raw continuous columns from prep.X because prep.X_cont is normalized.
    # TabPFN should not receive scaled continuous features when raw values exist.
    raw_cols = [col for col in cont_cols if col in prep.X.columns]
    missing_cols = [col for col in cont_cols if col not in prep.X.columns]
    parts = []
    if len(raw_cols) > 0:
        parts.append(prep.X.loc[:, raw_cols].copy())
    if len(missing_cols) > 0 and prep.X_cont is not None:
        parts.append(prep.X_cont.loc[:, missing_cols].copy())
    if len(indicator_cols) > 0:
        parts.append(prep.X.loc[:, indicator_cols].copy())
    if len(parts) == 0:
        return empty
    return pd.concat(parts, axis=1).loc[:, ordered_cols]


def _tabpfn_feature_matrices(
    prep_train: PreparedData, prep_test: PreparedData
) -> tuple[DataFrame, DataFrame, list[str]]:
    train_parts = [_tabpfn_continuous_features(prep_train)]
    test_parts = [_tabpfn_continuous_features(prep_test)]
    cat_cols: list[str] = []

    if prep_train.X_cat is not None and prep_test.X_cat is not None:
        cat_cols = [
            col for col in prep_train.X_cat.columns if col in prep_test.X_cat.columns
        ]
        if len(cat_cols) > 0:
            train_cat = prep_train.X_cat.loc[:, cat_cols].copy()
            test_cat = prep_test.X_cat.loc[:, cat_cols].copy()
            for col in cat_cols:
                train_values = _tabpfn_categorical_values(train_cat[col])
                test_values = _tabpfn_categorical_values(test_cat[col])

                categories = list(pd.unique(train_values))
                if TABPFN_UNKNOWN_CATEGORY not in categories:
                    categories.append(TABPFN_UNKNOWN_CATEGORY)

                unseen = ~test_values.isin(categories)
                test_values.loc[unseen] = TABPFN_UNKNOWN_CATEGORY

                train_cat[col] = pd.Categorical(
                    train_values,
                    categories=categories,
                )
                test_cat[col] = pd.Categorical(
                    test_values,
                    categories=categories,
                )
            train_parts.append(train_cat)
            test_parts.append(test_cat)

    X_train = pd.concat(train_parts, axis=1)
    X_test = pd.concat(test_parts, axis=1)
    return X_train, X_test, cat_cols


def _tabpfn_categorical_values(values: Series) -> Series:
    clean = values.astype("object")
    clean = clean.where(~clean.isna(), TABPFN_UNKNOWN_CATEGORY)
    return clean.map(
        lambda value: TABPFN_UNKNOWN_CATEGORY if pd.isna(value) else str(value)
    )


def _tabpfn_selected_cols(
    all_cols: list[str],
    cat_cols: list[str],
    cols: Union[list[str], slice],
) -> Union[list[str], slice]:
    if isinstance(cols, slice):
        return cols

    selected = set()
    ordered_cats = sorted(cat_cols, key=len, reverse=True)
    for col in cols:
        col = str(col)
        if col in all_cols:
            selected.add(col)
            continue
        for cat in ordered_cats:
            if col == cat or col.startswith(f"{cat}_"):
                selected.add(cat)
                break

    return [col for col in all_cols if col in selected]


def _get_tabpfn_splits(
    prep_train: PreparedData,
    prep_test: PreparedData,
    selection: str,
    cols: Union[list[str], slice],
) -> tuple[DataFrame, DataFrame, list[int]]:
    X_train, X_test, cat_cols = _tabpfn_feature_matrices(prep_train, prep_test)
    selected = _tabpfn_selected_cols(X_train.columns.to_list(), cat_cols, cols)
    if not isinstance(selected, slice):
        X_train = X_train.loc[:, selected]
        X_test = X_test.loc[:, selected]
    cat_indices = [X_train.columns.get_loc(col) for col in cat_cols if col in X_train]
    return X_train, X_test, [int(idx) for idx in cat_indices]


def evaluate_tuned(
    prepared: PreparedData,
    prep_train: PreparedData,
    prep_test: PreparedData,
    assoc_filtered: FilterSelected,
    pred_filtered: Optional[FilterSelected],
    model_selected: ModelSelected,
    options: ProgramOptions,
) -> EvaluationResults:
    model_cls: Type[DfAnalyzeModel]
    results: list[HtuneResult]
    dfs: list[DataFrame]

    selections: dict[str, Optional[list[str]]] = {"none": None}
    if FeatureSelection.Filter in options.feat_select:
        selections["assoc"] = assoc_filtered.selected
        if pred_filtered is not None:
            selections["pred"] = pred_filtered.selected

    embed_models: dict[str, EmbedSelectionModel] = {}
    if model_selected.embed_selected is not None:
        for selected in model_selected.embed_selected:
            selections[f"embed_{selected.model.value}"] = selected.selected
            embed_models[f"embed_{selected.model.value}"] = selected.model
    if model_selected.wrap_selected is not None:
        selections["wrap"] = model_selected.wrap_selected.selected

    if prepared.is_classification:
        metric = options.htune_cls_metric
    else:
        metric = options.htune_reg_metric

    dfs, results = [], []
    model_setup_failures: dict[Type[Any], Exception] = {}
    for model_cls in options.models:
        model_name = _model_shortname(model_cls)
        for selection, cols in selections.items():
            selected_cols = _get_cols(selection=selection, selected=cols)
            tabpfn_cat_indices: list[int] = []
            if model_name == "tabpfn":
                X_train, X_test, tabpfn_cat_indices = _get_tabpfn_splits(
                    prep_train, prep_test, selection, selected_cols
                )
            else:
                X_train, X_test = _get_splits(
                    prep_train, prep_test, selection, selected_cols
                )
            if X_train.empty or X_test.empty:
                raise ValueError(
                    f"Error when subsetting features for model '{model_cls.shortname}'. Got:\n"
                    f"cols: {cols}\n"
                    f"selected_cols: {selected_cols}\n"
                )
            if X_train.isna().any().any() or X_test.isna().any().any():
                raise ValueError(
                    f"Got NaNs when subsetting features for model '{model_cls.shortname}'. Got:\n"
                    f"cols: {cols}\n"
                    f"selected_cols: {selected_cols}\n"
                )

            if model_name in {"mlp", "gandalf"}:
                model = model_cls(num_classes=prepared.num_classes)  # type: ignore
            else:
                model = model_cls()  # type: ignore
            if model_name == "tabpfn":
                model.set_tabpfn_version(options.tabpfn_version)
                model.set_tune_folds(options.tabpfn_tune_folds)
                model.set_categorical_feature_indices(tabpfn_cat_indices)
            model.set_device_info(options.device_info)

            is_embed = "embed" in selection
            embed_model = embed_models[selection] if is_embed else None
            tune_seconds = 0.0
            eval_seconds = 0.0
            params: dict[str, Any] = {}
            setup_failure = model_setup_failures.get(model_cls)
            if setup_failure is not None:
                print(
                    f"Skipping {model.longname} for selection={selection}: "
                    f"{setup_failure}"
                )
                if prepared.is_classification:
                    nulls = Series(ClassifierScorer.null_scores(), name="metric")
                else:
                    nulls = Series(RegressorScorer.null_scores(), name="metric")
                df = DataFrame(
                    {
                        "metric": nulls.index,
                        "trainset": nulls.values,
                        "holdout": nulls.values,
                        "5-fold": nulls.values,
                    }
                )
                result = HtuneResult(
                    selection="embed" if is_embed else selection,  # type: ignore
                    selected_cols=X_train.columns.to_list(),
                    embed_select_model=embed_model,
                    model_cls=model_cls,
                    model=model,
                    params=params,
                    score=np.nan,
                    metric=metric,
                    preds_train=Series(),
                    preds_test=Series(),
                    probs_train=None,
                    probs_test=None,
                    **_device_result_fields(model, params),
                    model_version=model.model_version(),
                    tune_seconds=tune_seconds,
                    eval_seconds=eval_seconds,
                    total_seconds=tune_seconds + eval_seconds,
                )
                results.append(result)
            else:
                print(f"Tuning {model.longname} for selection={selection}")
                phase = "tuning"
                phase_start = perf_counter()
                try:
                    study = model.htune_optuna(
                        X_train=X_train,
                        y_train=prep_train.y,
                        g_train=prep_train.groups,
                        n_trials=options.htune_trials,
                        metric=metric,  # type: ignore
                        n_jobs=-1,
                        verbosity=optuna.logging.ERROR,
                    )
                    tune_seconds = perf_counter() - phase_start
                    params = dict(study.best_params)
                    phase = "evaluation"
                    phase_start = perf_counter()
                    df, preds_train, preds_test, probs_train, probs_test = (
                        model.htune_eval(
                            X_train=X_train,
                            y_train=prep_train.y,
                            g_train=prep_train.groups,
                            X_test=X_test,
                            y_test=prep_test.y,
                            g_test=prep_test.groups,
                            seed=options.seed,
                        )
                    )
                    eval_seconds = perf_counter() - phase_start
                    result = HtuneResult(
                        selection="embed" if is_embed else selection,  # type: ignore
                        selected_cols=X_train.columns.to_list(),
                        embed_select_model=embed_model,
                        model_cls=model_cls,
                        model=model,
                        params=params,
                        metric=metric,
                        score=study.best_value,
                        preds_train=preds_train,
                        preds_test=preds_test,
                        probs_train=probs_train,
                        probs_test=probs_test,
                        **_device_result_fields(model, params),
                        model_version=model.model_version(),
                        tune_seconds=tune_seconds,
                        eval_seconds=eval_seconds,
                        total_seconds=tune_seconds + eval_seconds,
                    )
                    results.append(result)
                except Exception as e:
                    if is_fatal_accelerator_error(e):
                        raise
                    elapsed = perf_counter() - phase_start
                    if phase == "tuning":
                        tune_seconds = elapsed
                    else:
                        eval_seconds = elapsed
                    if prepared.is_classification:
                        nulls = Series(ClassifierScorer.null_scores(), name="metric")
                    else:
                        nulls = Series(RegressorScorer.null_scores(), name="metric")
                    is_tabpfn_weights_error = _is_tabpfn_weights_error(e)
                    if is_tabpfn_weights_error:
                        model_setup_failures[model_cls] = e
                    message = (
                        f"\nGot exception when trying to tune and evaluate "
                        f"{model.shortname}:\n{e}"
                    )
                    if not is_tabpfn_weights_error:
                        message = f"{message}\n{traceback.format_exc()}"
                    warn(message)
                    df = DataFrame(
                        {
                            "metric": nulls.index,
                            "trainset": nulls.values,
                            "holdout": nulls.values,
                            "5-fold": nulls.values,
                        }
                    )
                    result = HtuneResult(
                        selection="embed" if is_embed else selection,  # type: ignore
                        selected_cols=X_train.columns.to_list(),
                        embed_select_model=embed_model,
                        model_cls=model_cls,
                        model=model,
                        params=params,
                        score=np.nan,
                        metric=metric,
                        preds_train=Series(),
                        preds_test=Series(),
                        probs_train=None,
                        probs_test=None,
                        **_device_result_fields(model, params),
                        model_version=model.model_version(),
                        tune_seconds=tune_seconds,
                        eval_seconds=eval_seconds,
                        total_seconds=tune_seconds + eval_seconds,
                    )
                    results.append(result)
            df["model"] = model.shortname
            df["selection"] = selection
            df["embed_selector"] = (
                embed_model.value if embed_model is not None else "none"
            )
            df["device"] = result.device
            df["requested_device"] = result.requested_device
            df["resolved_device"] = result.resolved_device
            df["torch_device"] = result.torch_device
            df["accelerated"] = result.accelerated
            df["backend"] = result.backend
            df["model_version"] = result.model_version
            df["tune_seconds"] = result.tune_seconds
            df["eval_seconds"] = result.eval_seconds
            df["total_seconds"] = result.total_seconds
            dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return EvaluationResults(
        df=df,
        X_train=prep_train.X,
        X_test=prep_test.X,
        y_train=prep_train.y,
        y_test=prep_test.y,
        results=results,
        is_classification=prepared.is_classification,
    )


@dataclass(init=True, repr=True, eq=True, frozen=True)
class HtuneResultLegacy:
    estimator: Estimator
    mode: EstimationMode
    n_trials: int
    cv_method: CVMethod
    val_acc: float = np.nan
    best_params: Dict = field(default_factory=dict)


def train_val_splits(
    df: DataFrame, is_classification: bool, val_size: float = VAL_SIZE
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Wrapper around `sklearn.model_selection.train_test_split` to return splits as `DataFrame`s
    instead of numpy arrays.

    Parameters
    ----------
    df: DataFrame
        Data with target in column named "target".

    mode: Literal["classify", "regress"]
        What kind of model we are running.

    val_size: float = 0.2
        Percent of data to reserve for validation

    Returns
    -------
    splits: [X_train, X_val, y_train, y_val]
    """
    train, val = train_test_split(
        df, test_size=val_size, random_state=SEED, shuffle=True, stratify=df.target
    )
    X_train = train.drop(columns="target")
    X_val = val.drop(columns="target")
    y_train = train.target
    y_val = val.target
    if is_classification:
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
    return X_train, X_val, y_train, y_val


def cv_desc(cv_method: CVMethod, n_folds: Optional[int] = None) -> str:
    """Helper for logging a readable description of the CVMethod to stdout"""
    if isinstance(cv_method, int):
        return f"stratified {cv_method}-fold"
    if isinstance(cv_method, float):  # stratified holdout
        if cv_method <= 0 or cv_method >= 1:
            raise ValueError("Holdout test_size must be in (0, 1)")
        perc = int(100 * cv_method)
        return f"stratified {perc}% holdout"
    cv_method = str(cv_method).lower()  # type: ignore

    if n_folds is None:
        n_folds = 5
    if cv_method == "loocv":
        return "LOOCV"
    if cv_method in ["kfold", "k-fold"]:
        return f"{n_folds}-fold"
    if cv_method == "mc":
        return "stratified Monte-Carlo (20 random 20%-sized test sets)"
    raise ValueError(f"Invalid `cv_method`: {cv_method}")


def package_classifier_cv_scores(
    scores: Dict[str, ndarray],
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    result = dict(
        htuned=htuned,
        cv_method=htuned.cv_method,
        acc=np.mean(scores["test_acc"]),
        auc=np.mean(scores["test_auroc"]),
        sens=np.mean(scores["test_sens"]),
        spec=np.mean(scores["test_spec"]),
        acc_sd=np.std(scores["test_acc"], ddof=1),
        auc_sd=np.std(scores["test_auroc"], ddof=1),
        sens_sd=np.std(scores["test_sens"], ddof=1),
        spec_sd=np.std(scores["test_spec"], ddof=1),
    )
    if not log:
        return result

    acc_mean = float(np.mean(scores["test_acc"]))
    auc_mean = float(np.mean(scores["test_auroc"]))
    sens_mean = float(np.mean(scores["test_sens"]))
    spec_mean = float(np.mean(scores["test_spec"]))
    acc_sd = float(np.std(scores["test_acc"], ddof=1))
    auc_sd = float(np.std(scores["test_auroc"], ddof=1))
    sens_sd = float(np.std(scores["test_sens"], ddof=1))
    spec_sd = float(np.std(scores["test_spec"], ddof=1))
    desc = cv_desc(cv_method)
    # fmt: off
    print(f"Testing validation: {desc}")
    print(f"Accuracy:           μ = {np.round(acc_mean, 3):0.3f} (sd = {np.round(acc_sd, 4):0.4f})")  # noqa
    print(f"AUC:                μ = {np.round(auc_mean, 3):0.3f} (sd = {np.round(auc_sd, 4):0.4f})")  # noqa
    print(f"Sensitivity:        μ = {np.round(sens_mean, 3):0.3f} (sd = {np.round(sens_sd, 4):0.4f})")  # noqa
    print(f"Specificity:        μ = {np.round(spec_mean, 3):0.3f} (sd = {np.round(spec_sd, 4):0.4f})")  # noqa
    # fmt: on
    return result


def package_classifier_scores(
    y_test: ndarray,
    y_pred: ndarray,
    y_score: ndarray,
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    sens = sensitivity(y_test, y_pred)
    spec = specificity(y_test, y_pred)
    percent = int(100 * float(cv_method))
    scores = dict(
        test_accuracy=np.array([acc]).ravel(), test_roc_auc=np.array([auc]).ravel()
    )
    result = dict(
        htuned=htuned,
        cv_method=cv_method,
        acc=np.mean(scores["test_acc"]),
        auc=np.mean(scores["test_auroc"]),
        sens=sens,
        spec=spec,
        acc_sd=np.nan,
        auc_sd=np.nan,
        sens_sd=np.nan,
        spec_sd=np.nan,
    )
    if not log:
        return result
    print(f"Testing validation: {percent}% holdout")
    print(
        f"          Accuracy: μ = {np.round(acc, 3):0.3f} (sd = {np.round(acc, 4):0.4f})"
    )
    print(
        f"               AUC: μ = {np.round(auc, 3):0.3f} (sd = {np.round(auc, 4):0.4f})"
    )
    return result


def package_regressor_cv_scores(
    scores: Dict[str, ndarray],
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    result = dict(
        htuned=htuned,
        cv_method=htuned.cv_method,
        mae=np.mean(scores["test_mae"]),
        msqe=np.mean(scores["test_msqe"]),
        mdae=np.mean(scores["test_mdae"]),
        mape=np.mean(scores["test_mape"]),
        r2=np.mean(scores["test_r2"]),
        var_exp=np.mean(scores["test_var-exp"]),
        mae_sd=np.std(scores["test_mae"], ddof=1),
        msqe_sd=np.std(scores["test_msqe"], ddof=1),
        mdae_sd=np.std(scores["test_mdae"], ddof=1),
        mape_sd=np.std(scores["test_mape"], ddof=1),
        r2_sd=np.std(scores["test_r2"], ddof=1),
        var_exp_sd=np.std(scores["test_var-exp"], ddof=1),
    )
    if not log:
        return result

    mae = np.mean(scores["test_mae"])
    msqe = np.mean(scores["test_msqe"])
    mdae = np.mean(scores["test_mdae"])
    # mape = np.mean(scores["test_mape"])
    r2 = np.mean(scores["test_r2"])
    var_exp = np.mean(scores["test_var-exp"])
    mae_sd = np.std(scores["test_mae"], ddof=1)
    msqe_sd = np.std(scores["test_msqe"], ddof=1)
    mdae_sd = np.std(scores["test_mdae"], ddof=1)
    # mape_sd = np.std(scores["test_mape"], ddof=1)
    r2_sd = np.std(scores["test_r2"], ddof=1)
    var_exp_sd = np.std(scores["test_var exp"], ddof=1)

    desc = cv_desc(cv_method)
    # fmt: off
    print(f"Testing validation: {desc}")
    print(f"MAE             μ = {np.round(mae, 3):0.3f} (sd = {np.round(mae_sd, 4):0.4f})")  # noqa
    print(f"MSqE:           μ = {np.round(msqe, 3):0.3f} (sd = {np.round(msqe_sd, 4):0.4f})")  # noqa
    print(f"Median Abs Err: μ = {np.round(mdae, 3):0.3f} (sd = {np.round(mdae_sd, 4):0.4f})")  # noqa
    # print(f"MAPE:           μ = {np.round(mape, 3):0.3f} (sd = {np.round(mape_sd, 4):0.4f})")  # noqa
    print(f"R-squared:      μ = {np.round(r2, 3):0.3f} (sd = {np.round(r2_sd, 4):0.4f})")  # noqa
    print(f"Var explained:  μ = {np.round(var_exp, 3):0.3f} (sd = {np.round(var_exp_sd, 4):0.4f})")  # noqa
    # fmt: on
    return result


def package_regressor_scores(
    y_test: ndarray,
    y_pred: ndarray,
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    scores = {
        "test_MAE": mean_absolute_error(y_test, y_pred),
        "test_MSqE": mean_squared_error(y_test, y_pred),
        "test_MdAE": median_absolute_error(y_test, y_pred),
        "test_MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "test_R2": r2_score(y_test, y_pred),
        "test_Var exp": explained_variance_score(y_test, y_pred),
    }
    if log:
        percent = int(100 * float(cv_method))
        print(f"Testing validation: {percent}% holdout")
    with warnings.catch_warnings():
        # suppress warnings for taking std of length 1 ndarray:
        # RuntimeWarning: Degrees of freedom <= 0 for slice
        # RuntimeWarning: invalid value encountered in double_scalars
        warnings.simplefilter("ignore", RuntimeWarning)
        return package_regressor_cv_scores(scores, htuned, cv_method, log)


"""See Optuna docs (https://optuna.org/#code_ScikitLearn) for the motivation behond the closures
below. Currently I am using closures, but this might be a BAD IDEA in parallel contexts. In any
case, they do seem to suggest this is OK https://optuna.readthedocs.io/en/stable/faq.html
#how-to-define-objective-functions-that-have-own-arguments, albeit by using classes or lambdas. """


def hypertune_classifier(
    classifier: Classifier,
    X_train: DataFrame,
    y_train: DataFrame,
    n_trials: int = 200,
    cv_method: CVMethod = 5,
    verbosity: int = optuna.logging.ERROR,
) -> HtuneResultLegacy:
    """Core function. Uses Optuna base TPESampler (Tree-Parzen Estimator Sampler) to perform
    Bayesian hyperparameter optimization via Gaussian processes on the classifier specified in
    `classifier`.

    Parameters
    ----------
    classifier: Classifier
        Classifier to tune.

    X_train: DataFrame
        DataFrame with no target value (features only). Shape (n_samples, n_features)

    y_train: DataFrame
        Target values. Shape (n_samples,).

    n_trials: int = 200
        Number of trials to use with Optuna.

    cv_method: CVMethod = 5
        How to evaluate accuracy during tuning.

    verbosity: int = optuna.logging.ERROR
        See https://optuna.readthedocs.io/en/stable/reference/logging.html. Most useful other option
        is `optuna.logging.INFO`.

    Returns
    -------
    htuned: HtuneResultLegacy
        See top of this file.
    """
    OBJECTIVES: Dict[str, Callable] = {
        "rf": rf_classifier_objective(X_train, y_train, cv_method),
        "svm": svm_classifier_objective(X_train, y_train, cv_method),
        "dtree": dtree_classifier_objective(X_train, y_train, cv_method),
        "mlp": mlp_classifier_objective(X_train, y_train, cv_method),
        "bag": bagging_classifier_objective(X_train, y_train, cv_method),
    }
    # HYPERTUNING
    objective = OBJECTIVES[classifier]
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )
    optuna.logging.set_verbosity(verbosity)
    if classifier == "mlp":
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = (
            "ignore"  # can't kill ConvergenceWarning any other way
        )
        study.optimize(objective, n_trials=n_trials)
        os.environ["PYTHONWARNINGS"] = before
    else:
        study.optimize(objective, n_trials=n_trials)

    val_method = cv_desc(cv_method)
    acc = np.round(study.best_value, 3)
    if verbosity != optuna.logging.ERROR:
        print(f"\n{' Tuning Results ':=^80}")
        print("Best params:")
        pprint(study.best_params, indent=4, width=80)
        print(f"\nTuning validation: {val_method}")
        print(f"Best accuracy:      μ = {acc:0.3f}")
        # print("=" * 80, end="\n")

    return HtuneResultLegacy(
        estimator=classifier,
        mode="classify",
        n_trials=n_trials,
        cv_method=cv_method,
        val_acc=study.best_value,
        best_params=study.best_params,
    )
