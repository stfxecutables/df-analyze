from __future__ import annotations

import re
import traceback
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, cast
from warnings import warn

import jsonpickle
import numpy as np
import pandas as pd
from df_analyze._constants import (
    N_CAT_LEVEL_MIN,
    N_TARG_LEVEL_MIN,
    UNIVARIATE_PRED_MAX_N_SAMPLES,
)
from df_analyze.enumerables import NanHandling
from df_analyze.preprocessing.cleaning import (
    clean_regression_target,
    deflate_categoricals,
    drop_target_nans,
    drop_unusable,
    encode_categoricals,
    encode_target,
    handle_continuous_nans,
    normalize_continuous,
)
from df_analyze.preprocessing.inspection.inspection import (
    ClsTargetInfo,
    InspectionResults,
    RegTargetInfo,
    convert_categoricals,
    inspect_target,
    unify_nans,
)
from df_analyze.timing import timed
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer


@dataclass
class PrepFiles:
    X_raw: str = "X.parquet"
    X_cont_raw: str = "X_cont.parquet"
    X_cat_raw: str = "X_cat.parquet"
    y_raw: str = "y.parquet"
    labels: str = "labels.parquet"
    info: str = "info.json"


@dataclass
class PrepFilesTrain:
    X_raw: str = "X_train.parquet"
    X_cont_raw: str = "X_train_cont.parquet"
    X_cat_raw: str = "X_train_cat.parquet"
    y_raw: str = "y_train.parquet"
    labels: str = "labels.parquet"
    info: str = "info.json"


@dataclass
class PrepFilesTest:
    X_raw: str = "X_test.parquet"
    X_cont_raw: str = "X_test_cont.parquet"
    X_cat_raw: str = "X_test_cat.parquet"
    y_raw: str = "y_test.parquet"
    labels: str = "labels.parquet"
    info: str = "info.json"


@dataclass
class PreparationInfo:
    is_classification: bool
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    n_samples_dropped_via_target_NaNs: int
    n_cont_indicator_added: int
    target_info: Union[RegTargetInfo, ClsTargetInfo]
    runtimes: dict[str, float]

    def to_markdown(self) -> str:
        sections = []
        og = self.original_shape
        fs = self.final_shape
        task = "Classification" if self.is_classification else "Regression"
        orig_shape = f"{og[0]} samples × {og[1]} features"
        final_shape = f"{fs[0]} samples × {fs[1]} features"
        n_drop = self.n_samples_dropped_via_target_NaNs
        n_ind = self.n_cont_indicator_added
        funcs, times = zip(*self.runtimes.items())

        sections.append("# Data Preparation Summary\n\n")
        sections.append(f"Task:                   {task}\n")
        sections.append(f"Data original shape:    {orig_shape}\n")
        sections.append(f"Data final shape:       {final_shape}\n")
        sections.append(f"Target feature:         {self.target_info.name}\n")
        sections.append("\n")
        sections.append(f"Samples dropped due to NaN target: {n_drop}\n")
        sections.append(f"Indicator variables added for continuous NaNs: {n_ind}\n\n")
        sections.append("# Processing Times\n\n")
        sections.append(
            DataFrame(
                data=(np.array(times) * 1000).round(),
                columns=["runtime (ms)"],
                index=Series(name="computation", data=funcs),
            ).to_markdown()
        )

        return "".join(sections)

    def to_json(self, path: Path) -> None:
        path.write_text(str(jsonpickle.encode(self)))

    @staticmethod
    def from_json(path: Path) -> Optional[PreparationInfo]:
        content = path.read_text()
        if content.strip().replace("\n", "") == "":
            return None
        return cast(PreparationInfo, jsonpickle.decode(content))


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
        smax = 2**32 - 1
        if hasattr(rng.bit_generator, "seed_seq"):
            ent = rng.bit_generator.seed_seq.entropy  # type: ignore
        elif hasattr(rng.bit_generator, "_seed_seq"):
            ent = rng.bit_generator._seed_seq.entropy  # type: ignore
        else:
            ent = np.random.randint(1, smax)
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


class PreparedData:
    def __init__(
        self,
        X: DataFrame,
        y: Series,
        is_classification: Optional[bool] = None,
        X_cont: Optional[DataFrame] = None,
        X_cat: Optional[DataFrame] = None,
        labels: Optional[dict[int, str]] = None,
        inspection: Optional[InspectionResults] = None,
        info: Optional[PreparationInfo] = None,
        phase: Optional[Literal["train", "test"]] = None,
    ) -> None:
        # Attempt to automatically infer classification problem based on y.dtype
        # https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        if is_classification is None:
            kind = y.dtype.kind
            kinds = {"f": False, "i": True, "b": True, "u": True}
            if kind not in kinds:
                raise ValueError(
                    f"Got argument for parameter `y` with unsupported data type: {y.dtype}"
                )
            if kind == "O":
                ...
                # run cleaning.encode_target with X, y
                warn(
                    f"Found 'object' dtype for argument y (name='{y.name}'). Attempting to "
                    "automatically label encode. If this is undesirable, ensure that your "
                    "target `y` has the proper dtype, e.g. `y = y.astype(np.float64)` if "
                    "regression, or `y = y.astype(np.int64)` if classification. "
                )
                X, y, labels = encode_target(X, y, _warn=True)
                self.is_classification = True
            else:
                tname = (
                    "floating point"
                    if kind == "f"
                    else "signed/unsigned integer or boolean type"
                )
                self.is_classification = kinds[kind]
                warn(
                    f"Argument `is_classification` left at default. Inferred "
                    f"`is_classification={self.is_classification}`, since y is {tname}. "
                    "If this is incorrect, or to silence this warning, specify the value for "
                    "`is_classification`"
                )
        else:
            self.is_classification = is_classification
        self.inspection: Optional[InspectionResults] = inspection
        self.info: Optional[PreparationInfo] = info
        self.files: PrepFiles = PrepFiles()
        self.phase = phase

        X, X_cont, X_cat, y = self.validate(X, X_cont, X_cat, y)

        self.X = self.rename_cols(X)
        self.X_cont: Optional[DataFrame] = (
            None if X_cont is None else self.rename_cols(X_cont)
        )
        self.X_cat: Optional[DataFrame] = (
            None if X_cat is None else self.rename_cols(X_cat)
        )
        self.y: Series = y
        self.target = self.y.name
        self.labels = labels or {}

    @property
    def num_classes(self) -> int:
        if not self.is_classification:
            return 1
        return len(np.unique(self.y))

    def split(
        self, train_size: Union[int, float] = 0.6
    ) -> tuple[PreparedData, PreparedData]:
        y = self.y
        if self.is_classification:
            ss = StratifiedShuffleSplit(
                train_size=train_size, n_splits=1, random_state=42
            )
        else:
            ss = ShuffleSplit(train_size=train_size, n_splits=1, random_state=42)
        idx_train, idx_test = next(ss.split(y, y))  # type: ignore

        prep_train = self.subsample(idx_train)
        prep_train.phase = "train"

        prep_test = self.subsample(idx_test)
        prep_train.phase = "test"

        return prep_train, prep_test

    def subsample(self, idx: ndarray) -> PreparedData:
        X_sub = self.X.iloc[idx]
        X_cont, X_cat = self.X_cont, self.X_cat
        if self.info is not None:
            info_sub = deepcopy(self.info)
            info_sub.final_shape = X_sub.shape
        else:
            info_sub = None
        return PreparedData(
            X=X_sub,
            X_cont=None if X_cont is None else X_cont.iloc[idx],
            X_cat=None if X_cat is None else X_cat.iloc[idx],
            y=self.y.iloc[idx],
            labels=self.labels,
            inspection=self.inspection,
            info=info_sub,
        )

    def representative_subsample(
        self,
        n_sub: int = UNIVARIATE_PRED_MAX_N_SAMPLES,
        rng: Optional[Generator] = None,
    ) -> tuple[PreparedData, ndarray]:
        X, X_cont, X_cat = self.X, self.X_cont, self.X_cat
        y = self.y
        rng = rng or np.random.default_rng()

        if len(X) <= UNIVARIATE_PRED_MAX_N_SAMPLES:
            return self, np.arange(len(X), dtype=np.int64)

        if self.is_classification:
            strat = y
            idx = viable_subsample(df=X, target=y, n_sub=n_sub, rng=rng)
            X = X.iloc[idx]
            if X_cont is not None:
                X_cont = X_cont.iloc[idx]
            if X_cat is not None:
                X_cat = X_cat.iloc[idx]
            y = y.iloc[idx]
        else:
            kb = KBinsDiscretizer(n_bins=5, encode="ordinal")
            strat = kb.fit_transform(self.y.to_numpy().reshape(-1, 1))
            n_train = UNIVARIATE_PRED_MAX_N_SAMPLES
            ss = StratifiedShuffleSplit(n_splits=1, train_size=n_train)
            idx = next(ss.split(strat, strat))[0]
            X = cast(DataFrame, self.X.iloc[idx, :].copy(deep=True))
            y = self.y.loc[idx].copy(deep=True)
            if X_cont is not None:
                X_cont = X_cont.loc[idx, :].copy(deep=True)
            if X_cat is not None:
                X_cat = X_cat.loc[idx, :].copy(deep=True)

        return PreparedData(
            X=X,
            X_cont=X_cont,
            X_cat=X_cat,
            y=y,
            labels=self.labels,
            inspection=self.inspection,
            info=self.info,
        ), idx

    def validate(
        self,
        X: DataFrame,
        X_cont: Optional[DataFrame],
        X_cat: Optional[DataFrame],
        y: Series,
    ) -> tuple[DataFrame, Optional[DataFrame], Optional[DataFrame], Series]:
        n_samples = len(X)
        if X_cont is not None and len(X_cont) != n_samples:
            raise ValueError(
                f"Continuous data number of samples ({len(X_cont)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )
        if X_cat is not None and len(X_cat) != n_samples:
            raise ValueError(
                f"Categorical data number of samples ({len(X_cat)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )
        if len(y) != n_samples:
            raise ValueError(
                f"Target number of samples ({len(X)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )

        if self.is_classification and np.bincount(y).min() < N_TARG_LEVEL_MIN:
            raise ValueError(f"Target '{y.name}' has undersampled levels")

        # Handle some BS due to stupid Pandas index behaviour
        X.reset_index(drop=True, inplace=True)
        if X_cont is not None:
            X_cont.index = X.index.copy(deep=True)
        if X_cat is not None:
            X_cat.index = X.index.copy(deep=True)
        y.index = X.index.copy(deep=True)
        return X, X_cont, X_cat, y

    def rename_cols(self, df: DataFrame) -> DataFrame:
        # see https://github.com/microsoft/LightGBM/issues/6202#issuecomment-1820286842
        # for LightGBM disallowed characters
        df = df.rename(
            columns=lambda col: re.sub(r"[\[\]\{\},:\"]+", "", str(col)),
        )
        dupe_cols = set(df.columns[df.columns.duplicated()])
        counts = {col: 0 for col in dupe_cols}
        new_cols = []
        for col in df.columns:
            if col not in dupe_cols:
                new_cols.append(col)
                continue
            if counts[col] == 0:  # leave name unchanged
                new_cols.append(col)
            else:
                new_cols.append(f"{col}_{counts[col]}")
            counts[col] += 1
        df.columns = new_cols
        return df

    def to_markdown(self) -> Optional[str]:
        if self.info is not None:
            return self.info.to_markdown()
        warn("No preparation info found, no Markdown report to make")

    def save_raw(self, root: Path) -> None:
        try:
            self.X.to_parquet(root / self.files.X_raw)
            self.X_cont.to_parquet(root / self.files.X_cont_raw)
            self.X_cat.to_parquet(root / self.files.X_cat_raw)
            self.y.to_frame().to_parquet(root / self.files.y_raw)
            if self.labels is not None:
                Series(self.labels).to_frame().to_parquet(root / self.files.labels)
            if self.info is not None:
                self.info.to_json(root / self.files.info)
        except Exception as e:
            warn(
                f"Exception while saving {self.__class__.__name__} to {root}."
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    @staticmethod
    def from_saved(root: Path, inspection: InspectionResults) -> PreparedData:
        files = PrepFiles()
        X = pd.read_parquet(root / files.X_raw)
        X_cont = pd.read_parquet(root / files.X_cont_raw)
        X_cat = pd.read_parquet(root / files.X_cat_raw)
        y_raw = pd.read_parquet(root / files.y_raw)
        y = Series(name=y_raw.columns[0], data=y_raw.values.ravel(), index=y_raw.index)
        labelpath = root / files.labels
        labels: Optional[dict[int, str]]
        if labelpath.exists():
            labels = pd.read_parquet(labelpath).to_dict()  # type: ignore
        else:
            labels = None
        info = PreparationInfo.from_json(root / files.info)
        return PreparedData(
            X=X,
            X_cont=X_cont,
            X_cat=X_cat,
            y=y,
            labels=labels,
            inspection=inspection,
            info=info,
        )


def prepare_target(
    df: DataFrame,
    target: str,
    is_classification: bool,
    _warn: bool = True,
) -> tuple[DataFrame, Series, Optional[dict[int, str]]]:
    y = df[target]
    df = df.drop(columns=target)
    if is_classification:
        df, y, labels = encode_target(df, y, _warn=_warn)
    else:
        labels = None
        df, y = clean_regression_target(df, y)
    return df, y, labels


def prepare_data(
    df: DataFrame,
    target: str,
    results: InspectionResults,
    is_classification: bool,
    _warn: bool = True,
) -> PreparedData:
    """
    Returns
    -------
    X_encoded: DataFrame
        All encoded and processed predictors.

    X_cat: DataFrame
        The categorical variables remaining after processing (no encoding,
        for univariate metrics and the like).

    X_cont: DataFrame
        The continues variables remaining after processing (no encoding,
        for univariate metrics and the like).

    y: Series
        The regression or classification target, also encoded.

    info: dict[str, str]
        Other information regarding warnings and cleaning effects.

    """
    times: dict[str, float] = {}
    timer = partial(timed, times=times)
    orig_shape = (df.shape[0], df.shape[1] - 1)

    df = timer(unify_nans)(df)
    df = timer(convert_categoricals)(df, target)
    info = timer(inspect_target)(df, target, is_classification=is_classification)
    df, n_targ_drop = timer(drop_target_nans)(df, target)
    if is_classification:
        df, y, labels = timer(encode_target)(df, df[target])
    else:
        df, y = timer(clean_regression_target)(df, df[target])
        labels = None

    df = timer(drop_unusable)(df, results, _warn=_warn)
    df, X_cont, n_ind_added = handle_continuous_nans(
        df=df, target=target, results=results, nans=NanHandling.Median
    )
    X_cont = normalize_continuous(X_cont, robust=True)

    df = timer(deflate_categoricals)(df, results, _warn=_warn)
    df, X_cat = timer(encode_categoricals)(
        df, target, results=results, warn_explosion=_warn
    )

    X = df.drop(columns=target).reset_index(drop=True)
    return PreparedData(
        X=X,
        X_cont=X_cont,
        X_cat=X_cat,
        y=y,
        labels=labels,
        info=PreparationInfo(
            original_shape=orig_shape,
            final_shape=X.shape,
            n_samples_dropped_via_target_NaNs=n_targ_drop,
            n_cont_indicator_added=n_ind_added,
            target_info=info,
            runtimes=times,
            is_classification=is_classification,
        ),
        inspection=results,
    )
