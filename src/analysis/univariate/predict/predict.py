from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Optional,
    cast,
)
from warnings import WarningMessage

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame, Series
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from tqdm import tqdm

from src._constants import N_CAT_LEVEL_MIN, UNIVARIATE_PRED_MAX_N_SAMPLES
from src.analysis.univariate.predict.models import (
    CLS_MODELS,
    REG_MODELS,
)
from src.preprocessing.prepare import PreparedData


@dataclass
class PredFiles:
    conts = "cont.parquet"
    cats = "cat.parquet"
    idx = "subsample.npy"


class PredResults:
    def __init__(
        self,
        conts: Optional[DataFrame],
        cats: Optional[DataFrame],
        is_classification: bool,
        idx: Optional[ndarray] = None,
        errs: Optional[list[BaseException]] = None,
        warns: Optional[list[WarningMessage]] = None,
    ) -> None:
        self.conts = conts
        self.cats = cats
        self.is_classification = is_classification
        self.errs = errs
        self.warns = warns
        self.idx_subsample = idx
        self.files = PredFiles()

    def save(self, cachedir: Path) -> None:
        if self.conts is not None:
            self.conts.to_parquet(cachedir / self.files.conts)
        else:
            DataFrame().to_parquet(cachedir / self.files.conts)
        if self.cats is not None:
            self.cats.to_parquet(cachedir / self.files.cats)
        else:
            DataFrame().to_parquet(cachedir / self.files.conts)
        if self.idx_subsample is not None:
            with open(self.files.idx, "wb") as handle:
                np.save(handle, self.idx_subsample, allow_pickle=False, fix_imports=False)

    @staticmethod
    def is_saved(cachedir: Path) -> bool:
        files = PredFiles()
        conts = cachedir / files.conts
        cats = cachedir / files.cats
        return conts.exists() and cats.exists()

    @staticmethod
    def load(cachedir: Path, is_classification: bool) -> PredResults:
        preds = PredResults(conts=None, cats=None, is_classification=is_classification)
        try:
            conts = pd.read_parquet(cachedir / preds.files.conts)
            cats = pd.read_parquet(cachedir / preds.files.cats)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing cached predictions at {cachedir}")

        preds.conts = None if conts.empty else conts
        preds.cats = None if cats.empty else cats

        npy_file = cachedir / preds.files.idx
        if npy_file.exists():
            preds.idx_subsample = np.load(npy_file, allow_pickle=False, fix_imports=False)
        return preds

    def to_markdown(self, path: Path) -> None:
        sorter = "acc" if self.is_classification else "Var exp"
        if self.conts is not None:
            conts = self.conts.sort_values(by=sorter, ascending=False).to_markdown(floatfmt="0.4f")
            cont_table = f"# Continuous predictions\n\n{conts}\n\n"
        else:
            cont_table = ""
        if self.cats is not None:
            cats = self.cats.sort_values(by=sorter, ascending=False).to_markdown(floatfmt="0.4f")
            cats_table = f"# Categorical prediction:\n\n{cats}"
        else:
            cats_table = ""

        tables = cont_table + cats_table
        path.write_text(tables)


def continuous_feature_target_preds(
    continuous: DataFrame,
    column: str,
    target: Series,
    is_classification: bool,
) -> tuple[DataFrame, Optional[Exception], list[WarningMessage]]:
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("once")
        try:
            x = continuous[column].to_numpy().reshape(-1, 1)
            y = target
            models = CLS_MODELS if is_classification else REG_MODELS
            scores = []
            # pbar = tqdm(
            #     models, total=len(models), desc=models[0].__class__.__name__, leave=False, position=1
            # )
            for model_cls in models:
                # pbar.set_description(f"Tuning {model_cls.__name__}")
                model = model_cls()
                score, spam = model.evaluate(x, y)
                score.insert(0, "model", model.short)
                score.index = pd.Index([column], name="feature")
                scores.append(score)
                # pbar.update()
            # pbar.clear()
            return pd.concat(scores, axis=0), None, warns
        except Exception as e:
            traceback.print_exc()
            print(f"Got error generating predictions for column {column}: {e}")
            return DataFrame(), e, warns


def categorical_feature_target_preds(
    categoricals: DataFrame,
    column: str,
    target: Series,
    is_classification: bool,
) -> tuple[DataFrame, Optional[Exception], list[WarningMessage]]:
    """Must be UN-ENCODED categoricals"""
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("once")
        try:
            X = pd.get_dummies(categoricals[column], dummy_na=True, dtype=float).to_numpy()
            y = target
            if is_classification:
                y = Series(data=LabelEncoder().fit_transform(target), name=target.name)  # type: ignore
            models = CLS_MODELS if is_classification else REG_MODELS
            scores = []
            # pbar = tqdm(
            #     models, total=len(models), desc=models[0].__class__.__name__, leave=False, position=1
            # )
            for model_cls in models:
                # pbar.set_description(f"Tuning {model_cls.__name__}")
                model = model_cls()
                score, spam = model.evaluate(X, y)
                score.insert(0, "model", model.short)
                score.index = pd.Index([column], name="feature")
                scores.append(score)
                # pbar.update()
            # pbar.clear()
            return pd.concat(scores, axis=0), None, warns
        except Exception as e:
            traceback.print_exc()
            print(f"Got error generating predictions for column {column}: {e}")
            return DataFrame(), e, warns


def feature_target_predictions(
    categoricals: DataFrame,
    continuous: DataFrame,
    target: Series,
    is_classification: bool,
) -> tuple[Optional[DataFrame], Optional[DataFrame], list[BaseException], list[WarningMessage]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df_conts: list[DataFrame]
        df_cats: list[DataFrame]
        results: list[tuple[DataFrame, Optional[Exception], list[WarningMessage]]]

        results = Parallel(n_jobs=-1)(
            delayed(continuous_feature_target_preds)(
                continuous=continuous,
                column=col,
                target=target,
                is_classification=is_classification,
            )
            for col in tqdm(
                continuous.columns,
                desc="Predicting continuous features",
                total=continuous.shape[1],
                leave=True,
                position=0,
            )
        )  # type: ignore
        if len(results) > 0:
            df_conts, cont_errors, cont_warns = list(zip(*results))
        else:
            df_conts, cont_errors, cont_warns = [], [], []

        results = Parallel(n_jobs=-1)(
            delayed(categorical_feature_target_preds)(
                categoricals=categoricals,
                column=col,
                target=target,
                is_classification=is_classification,
            )
            for col in tqdm(
                categoricals.columns,
                desc="Predicting categorical features",
                total=categoricals.shape[1],
                leave=True,
                position=0,
            )
        )  # type: ignore

        if len(results) > 0:
            df_cats, cat_errors, cat_warns = list(zip(*results))
        else:
            df_cats, cat_errors, cat_warns = [], [], []

        df_cont = pd.concat(df_conts, axis=0) if len(df_conts) != 0 else None
        df_cat = pd.concat(df_cats, axis=0) if len(df_cats) != 0 else None
        errs = [*cont_errors, *cat_errors]
        errs = [e for e in errs if e is not None]
        all_warns = []
        if isinstance(cont_warns, tuple):
            cont_warns = list(cont_warns)
        if isinstance(cat_warns, tuple):
            cat_warns = list(cat_warns)
        for warns in cont_warns + cat_warns:
            all_warns.extend(warns)
        all_warns = [w for w in all_warns if w is not None]

    return df_cont, df_cat, errs, all_warns


def viable_subsample(
    df: DataFrame,
    target: Series,
    n_sub: int = 2000,
    rng: Optional[Generator] = None,
) -> ndarray:
    rng = rng or np.random.default_rng()
    unqs, cnts = np.unique(target, return_counts=True)
    n_min = N_CAT_LEVEL_MIN
    idx_count = cnts < n_min
    idx_final = np.arange(len(target))
    drop_vals = unqs[idx_count]
    unqs = unqs[~idx_count]
    idx_keep = ~target.isin(drop_vals)
    X: ndarray
    y: ndarray
    idx_final = idx_final[idx_keep]
    X = df.copy(deep=True).values[idx_keep]
    y = target.copy(deep=True).values[idx_keep]
    assert np.bincount(y).min() >= N_CAT_LEVEL_MIN, "Keep fail"

    # shuffle once to allow getting random first n of each class later
    idx_shuffle = rng.permutation(len(y))
    idx_final = idx_final[idx_shuffle]
    X = X[idx_shuffle]
    y = y[idx_shuffle]
    assert np.bincount(y).min() >= N_CAT_LEVEL_MIN, "Shuffle fail"
    #
    idx = np.argsort(y)
    idx_final = idx_final[idx]
    X = X[idx]
    y = y[idx]
    assert np.bincount(y).min() >= N_CAT_LEVEL_MIN, "Argsort fail"
    # this gives e.g. stops[i]:stops[i+1] are class i
    stops = np.searchsorted(y, unqs)
    cls_idxs = []
    for i in range(len(stops) - 1):
        start, stop = stops[i], stops[i + 1]
        shuf = rng.permutation(stop - start)
        cls_idxs.append(np.arange(start, stop)[shuf][:n_min])
    shuf = rng.permutation(len(y) - stops[-1])
    cls_idxs.append(np.arange(stops[-1], len(y))[shuf][:n_min])

    idx_required = np.concatenate(cls_idxs).ravel()
    idx_final_req = idx_final[idx_required]
    y_req = y[idx_required]
    assert np.bincount(y_req).min() >= N_CAT_LEVEL_MIN, "collect fail"

    n_remain = n_sub - len(y_req)
    if n_remain <= 0:
        return idx_final_req

    idx_remain = np.ones_like(y, dtype=bool)
    idx_remain[idx_required] = False
    idx_final_rem = idx_final[idx_remain]
    X_remain = X[idx_remain]
    n_remain = min(n_remain, len(y))
    if n_remain >= len(y):
        idx_full = np.concatenate([idx_final_req, idx_final_rem])
        assert np.bincount(target[idx_full]).min() >= N_CAT_LEVEL_MIN, "remain fail"
        return idx_full
    else:
        ent = rng.bit_generator.seed_seq.entropy  # type: ignore
        smax = 2**32 - 1
        while ent > smax:
            ent //= 2

        ss = ShuffleSplit(
            n_splits=1,
            train_size=n_remain,
            random_state=ent,
        )
        idx = next(ss.split(X_remain))[0]
        idx_final_strat = idx_final_rem[idx]
        idx_full = np.concatenate([idx_final_req, idx_final_strat])
        assert np.bincount(target[idx_full]).min() >= N_CAT_LEVEL_MIN, "concat fail"
        return idx_full


def get_representative_subsample(
    prepared: PreparedData,
    is_classification: bool,
    rng: Optional[Generator] = None,
) -> tuple[DataFrame, DataFrame, DataFrame, Series, ndarray]:
    X, X_cont, X_cat = prepared.X, prepared.X_cont, prepared.X_cat
    y = prepared.y
    rng = rng or np.random.default_rng()

    if len(X) <= UNIVARIATE_PRED_MAX_N_SAMPLES:
        return X, X_cont, X_cat, y, np.arange(len(y), dtype=np.int64)

    if is_classification:
        strat = y
        idx = viable_subsample(df=X, target=y, n_sub=UNIVARIATE_PRED_MAX_N_SAMPLES, rng=rng)
        X = X.iloc[idx]
        X_cont = X_cont.iloc[idx]
        X_cat = X_cat.iloc[idx]
        y = y.iloc[idx]
    else:
        kb = KBinsDiscretizer(n_bins=5, encode="ordinal")
        strat = kb.fit_transform(prepared.y.to_numpy().reshape(-1, 1))
        n_train = UNIVARIATE_PRED_MAX_N_SAMPLES
        ss = StratifiedShuffleSplit(n_splits=1, train_size=n_train)
        idx = next(ss.split(strat, strat))[0]
        X = cast(DataFrame, prepared.X.iloc[idx, :].copy(deep=True))
        y = prepared.y.loc[idx].copy(deep=True)
        X_cat = prepared.X_cat.loc[idx, :].copy(deep=True)
        X_cont = prepared.X_cont.loc[idx, :].copy(deep=True)

    return X, X_cont, X_cat, y, idx


def univariate_predictions(
    prepared: PreparedData,
    is_classification: bool,
) -> PredResults:
    X, X_cont, X_cat, y, idx = get_representative_subsample(
        prepared=prepared, is_classification=is_classification
    )
    df_cont, df_cat, errs, warns = feature_target_predictions(
        categoricals=X_cat,
        continuous=X_cont,
        target=y,
        is_classification=is_classification,
    )
    return PredResults(
        conts=df_cont,
        cats=df_cat,
        is_classification=is_classification,
        idx=idx,
        errs=errs,
        warns=warns,
    )
