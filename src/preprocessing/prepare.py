from __future__ import annotations

import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional, Tuple, Union, cast
from warnings import warn

import jsonpickle
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer  # noqa

from src._constants import N_TARG_LEVEL_MIN
from src.enumerables import NanHandling
from src.preprocessing.cleaning import (
    clean_regression_target,
    deflate_categoricals,
    drop_target_nans,
    drop_unusable,
    encode_categoricals,
    encode_target,
    handle_continuous_nans,
    normalize_continuous,
)
from src.preprocessing.inspection.inspection import (
    ClsTargetInfo,
    InspectionResults,
    RegTargetInfo,
    convert_categoricals,
    inspect_target,
    unify_nans,
)
from src.timing import timed


@dataclass
class PrepFiles:
    X_raw: str = "X.parquet"
    X_cont_raw: str = "X_cont.parquet"
    X_cat_raw: str = "X_cat.parquet"
    y_raw: str = "y.parquet"
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
    def from_json(path: Path) -> PreparationInfo:
        return cast(PreparationInfo, jsonpickle.decode(path.read_text()))


class PreparedData:
    def __init__(
        self,
        X: DataFrame,
        X_cont: DataFrame,
        X_cat: DataFrame,
        y: Series,
        labels: Optional[dict[int, str]],
        inspection: InspectionResults,
        info: PreparationInfo,
    ) -> None:
        self.is_classification: bool = info.is_classification
        self.inspection: InspectionResults = inspection
        self.info: PreparationInfo = info
        self.files: PrepFiles = PrepFiles()

        X, X_cont, X_cat, y = self.validate(X, X_cont, X_cat, y)
        self.X: DataFrame = X
        self.X_cont: DataFrame = X_cont
        self.X_cat: DataFrame = X_cat
        self.y: Series = y
        self.target = self.y.name
        self.labels = labels

    def validate(
        self, X: DataFrame, X_cont: DataFrame, X_cat: DataFrame, y: Series
    ) -> tuple[DataFrame, DataFrame, DataFrame, Series]:
        n_samples = len(X)
        if len(X_cont) != n_samples:
            raise ValueError(
                f"Continuous data number of samples ({len(X_cont)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )
        if len(X_cat) != n_samples:
            raise ValueError(
                f"Categorical data number of samples ({len(X_cat)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )
        if len(y) != n_samples:
            raise ValueError(
                f"Target number of samples ({len(X_cat)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )

        if self.is_classification and np.bincount(y).min() < N_TARG_LEVEL_MIN:
            raise ValueError(f"Target {y} has undersampled levels")

        # Handle some BS due to stupid Pandas index behaviour
        X.reset_index(drop=True, inplace=True)
        X_cont.index = X.index.copy(deep=True)
        X_cat.index = X.index.copy(deep=True)
        y.index = X.index.copy(deep=True)
        return X, X_cont, X_cat, y

    def to_markdown(self) -> Optional[str]:
        return self.info.to_markdown()

    def save_raw(self, root: Path) -> None:
        try:
            self.X.to_parquet(root / self.files.X_raw)
            self.X_cont.to_parquet(root / self.files.X_cont_raw)
            self.X_cat.to_parquet(root / self.files.X_cat_raw)
            self.y.to_frame().to_parquet(root / self.files.y_raw)
            if self.labels is not None:
                DataFrame(self.labels).to_parquet(root / self.files.labels)
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
    df, X_cat = timer(encode_categoricals)(df, target, results=results, warn_explosion=_warn)

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
