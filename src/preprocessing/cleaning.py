import re
import sys
from pathlib import Path
from shutil import get_terminal_size
from typing import Optional
from warnings import warn

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from tqdm import tqdm

from src._constants import MAX_PERF_N_FEATURES, N_TARG_LEVEL_MIN, NAN_STRINGS
from src.enumerables import NanHandling
from src.loading import load_spreadsheet
from src.preprocessing.inspection.inspection import (
    InspectionInfo,
    InspectionResults,
    convert_categoricals,
    messy_inform,
    unify_nans,
)


class DataCleaningWarning(UserWarning):
    """For when df-analyze significantly and automatically alters input data"""

    def __init__(self, message: str) -> None:
        cols = get_terminal_size((81, 24))[0]
        sep = "=" * cols
        underline = "." * (len(self.__class__.__name__) + 1)
        self.message = (
            f"\n{sep}\n{self.__class__.__name__}\n{underline}\n{message}\n{sep}"
        )

    def __str__(self) -> str:
        return str(self.message)


class RenameInfo:
    def __init__(self, renames: list[tuple[str, str]]) -> None:
        self.renames = renames
        self.changed: list[tuple[str, str]] = []
        for original, renamed in self.renames:
            if original != renamed:
                self.changed.append((original, renamed))

    def to_markdown(self) -> Optional[str]:
        if len(self.changed) == 0:
            return None
        lines = []
        lines.append("# Feature Renames\n\n")
        lines.append(
            "Some of your feature names contained characters that are problems\n"
            "for either regular expressions or the LightGBM estimator, or\n"
            "which are duplicated. These have been renamed to resolve this\n"
            "issue. Duplicated column renames are given an integer suffix based\n"
            "on the order in which they appeared in the original data, except\n"
            "for the first instance of a duplicated name, which is left as is.\n"
            "\n\n"
        )
        originals, renamed = list(zip(*self.changed))
        table = DataFrame({"New name": renamed, "Original name": originals})
        lines.append(table.to_markdown(index=False))

        return "".join(lines)

    def rename_columns(self, cols: list[str]) -> list[str]:
        renamed = []
        for col in cols:
            if col in self.changed:
                renamed.append(self.changed[col])
            else:
                renamed.append(col)
        return renamed


def dedup_names(df: DataFrame, target: str) -> tuple[DataFrame, list[tuple[str, str]]]:
    y = df[target]
    df = df.drop(columns=target)
    dupe_cols = set(df.columns[df.columns.duplicated()])
    counts = {col: 0 for col in dupe_cols}
    renames: list[tuple[str, str]] = []
    for col in df.columns:
        if col not in dupe_cols:
            renames.append((col, col))
            continue
        # have to rename first, because if you have e.g. "F" with
        # values "1", "2", "NaN", and a second "F" with the same set of
        # values, then one-hot encoding will make "F_1", "F_2", "F_NaN",
        # and the rename here would have created "F_1", so we get a
        # duplicate. Renaming with a 0 (and double underscores) prevents
        # this
        renames.append((col, f"{col}__{counts[col]}"))  # even have to rem
        counts[col] += 1
    newnames = [new for old, new in renames]
    df.columns = newnames
    df[target] = y
    return df, renames


def sanitize_names(df: DataFrame, target: str) -> tuple[DataFrame, RenameInfo]:
    """Rename trashy feature names (e.g. with regex characters, spaces)
    to be less problematic

    Returns
    -------
    df: DataFrame
        Renamed DataFrame

    info: RenameInfo
        Container holding tuples of renamed columns. Does NOT include target
        (which is never renamed).
    """

    names = df.columns.to_series().apply(str).to_list()
    # https://docs.python.org/3/library/re.html
    if target not in names:
        raise RuntimeError(
            f"Unrecoverable error. Specified target: `{target}` not found in "
            f"data with feature names: {names}."
        )
    names.remove(target)

    # if `target` still in names, there was a dupe
    if target in names:
        raise RuntimeError(
            f"Unrecoverable error. Specified target: `{target}` appears "
            f"multiple times as a column name. Column names: {names}."
        )

    trash = r"[\\\.\^\$\*\+\?\{\}\[\]\(\)\| ]"
    lgbm = r"[,:\"]"

    renames = {}
    for name in names:
        renamed = re.sub(trash, "_", name)
        renamed = re.sub(lgbm, "_", renamed)
        renamed = re.sub(r"[_]+", "_", renamed)  # readability
        if renamed[-1] == "_":  # ugly dangling
            renamed = renamed[:-1]
        renames[name] = renamed

    # At this point we have a serious problem if any features have the same
    # name as the target. We have to decide between renaming the target or
    # renaming the conflicting features. Obviously we rename the features.
    if target in renames.values():
        for original, renamed in renames.items():
            if renamed != target:
                continue
            renames[original] = f"feat_{renamed}"

    df = df.rename(columns=renames)
    df, renames = dedup_names(df, target=target)
    info = RenameInfo(renames)
    return df, info


def restore_names(df: DataFrame, renames: list[tuple[str, str]]) -> DataFrame:
    for original, renamed in renames:
        raise NotImplementedError()


def cleaning_inform(message: str) -> None:
    cols = get_terminal_size((81, 24))[0]
    sep = "=" * cols
    title = "Cleaning Data Features"
    underline = "." * (len(title) + 1)
    message = f"\n{sep}\n{title}\n{underline}\n{message}\n{sep}"
    print(message, file=sys.stderr)


def normalize(df: DataFrame, target: Optional[str], robust: bool = True) -> DataFrame:
    """
    Clip data to within twice of its "robust range" (range of 90% of the data),
    and then min-max normalize.

    Notes
    -----
    Since df-analyze accepts arbitrary data, it needs to be somewhat robust to
    extreme values, and so a naive MinMax normalization seems unwise as a
    default (i.e. a single outlier can essentially "destroy" the information
    value of a feature for classifiers that are sensitive to scale, or in
    general). An ideal automated method might use more advanced outlier
    detection methods for each feature (e.g. sklearn's OneClassSVM or
    https://scikit-learn.org/stable/modules/outlier_detection.html).

    Robust, percentile-based normalization (as opposed to quantile-based
    normalization or normalization based on the probability integral transform)
    is powerful, generally not being obviously harmful to regular data, but
    being much better on highly skewed data or data with outliers.

    However, the actual impact of outliers (or rather, anchor points) surely in
    general depends more on the cumulative magnitudes of those outliers. For
    example, a feature value that is *double* the maximum value of all other
    values from the same feature is probably a clear outlier, but is not
    necessarily going to meaningfully affect the fits.

    Likewise, being an "outlier" may be a useful feature for prediction (e.g.
    by NOT normalizing or clipping, an estimator like a decision tree may learn
    an outlier threshold internally, and use this threshold effectively in
    prediction).

    NOTE: Thus, the optimal normalization strategy depends on the classier.
    Estimators highly sensitive to feature magnitudes, like linear or
    piecewise-linear classifers including LinearSVM, ElasticNet, or in theory
    shallow neural networks, and also support vector machines (for a visual
    proof of the sensitivity to scale see
    https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
    perhaps benefit immensely from robust normalization.

    Estimators involving decision trees could be harmed by any robust
    normalization that employs clipping. A nearest-neighbors-based estimator
    will be harmed by robust normalization and/or clipping if "being an
    outlier" is something that is correlated across features, i.e. if a sample
    that is an outlier on one feature also tends to be an outlier on most other
    features. However, if outlier samples are identified as such by just a
    small number of features relative to the total number of featurs, then
    normalization will mask their outlier status. (By contrast, for a
    neighbors-based estimator, if there are non-predictive features that
    contain outliers, these will skew the computed distances in a way that is
    not useful for prediction).



    """
    if (target in df.columns) and (target is not None):
        X = df.drop(columns=target)
    else:
        X = df

    cols = X.columns

    # need to not normalize one-hot columns...

    if robust:
        medians = np.nanmedian(X, axis=0)
        X = X - medians  # robust center

        # clip values 2 times more extreme than 95% of the data
        rmins = np.nanpercentile(X, 5, axis=0)
        rmaxs = np.nanpercentile(X, 95, axis=0)
        rranges = rmaxs - rmins
        rmins -= 2 * rranges
        rmaxs += 2 * rranges
        X = np.clip(X, a_min=rmins, a_max=rmaxs)

    X_norm = DataFrame(data=MinMaxScaler().fit_transform(X), columns=cols)
    if (target in df.columns) and (target is not None):
        X_norm = pd.concat([X, df[target]], axis=1)
    return X_norm


def normalize_continuous(X_cont: DataFrame, robust: bool = True) -> DataFrame:
    if X_cont.empty:
        return X_cont
    return normalize(df=X_cont, target=None, robust=robust)


def drop_target_nans(
    df: DataFrame,
    target: str,
) -> tuple[DataFrame, int]:
    y_str = df[target].copy(deep=True)
    y_str = y_str.apply(lambda x: np.nan if x in NAN_STRINGS else x)

    idx = ~y_str.isna()
    df = df.loc[idx]
    return df, (~idx).sum()


def handle_continuous_nans(
    df: DataFrame,
    target: str,
    results: InspectionResults,
    nans: NanHandling,
    add_indicators: bool = True,
) -> tuple[DataFrame, DataFrame, int]:
    """Impute or drop nans based on values not in `cat_cols`

    Returns
    -------
    df: DataFrame
        Imputed and indicator-augmented DataFrame with target included

    X_cont: DataFrame
        Imputed DataFrame with no target, no indicators (for used in
        univariate analyses and feature selection, etc)

    n_indicators: int
        Number of NaN indicator columns added
    """
    # drop rows where target is NaN: meaningless
    # NaNs in categoricals are handled as another dummy indicator
    cats = [*results.cats.infos.keys(), *results.binaries.infos.keys()]
    X_cat = df[cats].copy(deep=True)
    y = df[target]
    df = df.drop(columns=target)
    X = df.drop(columns=results.drop_cols(), errors="ignore")
    X = X.drop(columns=cats, errors="ignore")  # now only cats and ords

    # construct NaN indicators
    if add_indicators:
        X_nan = X.isna().astype(float)
        X_nan.rename(columns=lambda s: f"{s}_NAN", inplace=True)
        # remove constant indicators
        X_nan = X_nan.loc[:, X_nan.sum(axis=0) != 0]

    if nans is NanHandling.Drop:
        warn(
            "Dropping NaNs is currently not implemented as it can render "
            "usable data unusable, and is unnecessarily destructive of "
            "potential information relative to the use of indicator features "
            "and/or imputation strategies. NaN handling will be set to "
            f"'{NanHandling.Median.value}' instead."
        )
        nans = NanHandling.Median

    if X.empty:
        X_cont = X
    elif nans is NanHandling.Drop:
        raise NotImplementedError("Not implemented currently due to shape changes.")
        X_cont = X.dropna(axis="columns", how="any").dropna(axis="index", how="any")
        if 0 in X_cont.shape:
            others = [na.value for na in NanHandling if na is not NanHandling.Drop]
            raise RuntimeError(
                "Dropping NaNs resulted in either no remaining samples or features. "
                "Consider changing the option `--nan` to another valid option. Other "
                f"valid options: {others}"
            )
    elif nans in [NanHandling.Mean, NanHandling.Median]:
        strategy = "mean" if nans is NanHandling.Mean else "median"
        imputer = SimpleImputer(strategy=strategy, keep_empty_features=True)
        X_fitted = imputer.fit_transform(X)
        X_cont = DataFrame(data=X_fitted, columns=X.columns)
    elif nans is NanHandling.Impute:
        warn(
            "Using experimental multivariate imputation. This could take a very "
            "long time for even tiny (<500 samples, <30 features) datasets."
        )
        imputer = IterativeImputer(verbose=2, keep_empty_features=True)
        X_cont = DataFrame(data=imputer.fit_transform(X), columns=X.columns)
    else:
        raise NotImplementedError(f"Unhandled enum case: {nans}")

    if add_indicators:
        X = pd.concat([X_nan, X_cont, X_cat, y], axis=1)  # type: ignore
        return X, X_cont, int(X_nan.shape[1])  # type: ignore

    X = pd.concat([X_cont, X_cat, y], axis=1)
    return X, X_cont, 0


def encode_target(
    df: DataFrame, target: Series, _warn: bool = False
) -> tuple[DataFrame, Series, dict[int, str]]:
    unqs, cnts = np.unique(unify_nans(target).astype(str), return_counts=True)
    if len(unqs) <= 1:
        raise ValueError(f"Target variable {target.name} is constant.")
    idx = cnts <= N_TARG_LEVEL_MIN
    n_cls = len(unqs)
    if np.sum(idx).item() > 0:
        if _warn:
            cleaning_inform(
                "The target variable has a number of class labels "
                f"({unqs[idx]}) with less than {N_TARG_LEVEL_MIN} members. This "
                "will cause problems with splitting in various nested k-fold "
                "procedures used in `df-analyze`. In addition, any estimates "
                "or metrics produced for such a class will not be "
                "statistically meaningful (i.e. the uncertainty on those "
                "metrics or estimates will be exceedingly large). We thus "
                "remove all samples that belong to these labels, bringing the "
                f"total number of classes down to {n_cls - np.sum(idx).item()}"
            )
        idx_drop = ~target.isin(unqs[idx])
        df = df.copy().loc[idx_drop].reset_index(drop=True)
        # reset index extremely important for later concats
        target = target[idx_drop].reset_index(drop=True)

    # drop NaNs: Makes no sense to count correct NaN predictions toward
    # classification performance
    idx_drop = ~target.isna()
    df = df.copy().loc[idx_drop].reset_index(drop=True)
    target = target[idx_drop].reset_index(drop=True)

    enc = LabelEncoder()
    encoded = np.array(enc.fit_transform(target))
    classes = enc.classes_.tolist()
    ints = np.asarray(enc.transform(classes)).tolist()
    return (
        df,
        Series(encoded, name=target.name),
        {i: cls for i, cls in zip(ints, classes)},
    )


def clean_regression_target(df: DataFrame, target: Series) -> tuple[DataFrame, Series]:
    """NaN targets cannot be predicted. Remove them, and then robustly
    normalize target to facilitate convergence and interpretation
    of metrics
    """
    idx_drop = ~target.isna()
    # reset index extremely important for later concats
    df = df.loc[idx_drop].reset_index(drop=True)
    target = target[idx_drop].reset_index(drop=True)

    y = (
        RobustScaler(quantile_range=(2.5, 97.5))
        .fit_transform(target.to_numpy().reshape(-1, 1))
        .ravel()
    )
    target = Series(y, name=target.name)

    return df, target


def drop_cols(
    df: DataFrame, kind: str, *col_dicts: InspectionInfo, _warn: bool = False
) -> DataFrame:
    cols = set()
    cols_descs = []
    for d in col_dicts:
        for col, desc in d.infos.items():
            if col in df:
                cols.add(col)
                cols_descs.append((col, desc))
    cols_descs = sorted(cols_descs, key=lambda pair: pair[0])

    if len(cols) <= 0:  # nothing to drop
        return df

    w = max(len(col) for col in cols) + 2
    info = "\n".join([f"{col:<{w}} {desc}" for col, desc in cols_descs])
    if _warn:
        cleaning_inform(
            f"Dropping features that appear to be {kind}. Additional information "
            "should be available above.\n\n"
            f"Dropped features:\n{info}"
        )
    drops = list(cols)

    df = df.drop(columns=drops, errors="ignore")
    return df


def floatify(df: DataFrame) -> DataFrame:
    df = df.copy()
    cols = df.select_dtypes(include=["object", "string[python]"]).columns.tolist()
    for col in cols:
        try:
            df[col] = df[col].astype(float)
        except Exception:
            pass
    return df


def drop_unusable(
    df: DataFrame, results: InspectionResults, _warn: bool = False
) -> DataFrame:
    """Drops identifiers, datetime, constants"""
    df = drop_cols(df, "identifiers", results.ids, _warn=_warn)
    df = drop_cols(df, "datetime data", results.times, _warn=_warn)
    df = drop_cols(df, "constant", results.consts, _warn=_warn)
    return df


def deflate_categoricals(
    df: DataFrame,
    results: InspectionResults,
    _warn: bool = True,
) -> DataFrame:
    df = df.copy()

    infos = results.inflation
    infos = sorted(infos, key=lambda info: info.n_total, reverse=True)

    for info in tqdm(
        infos, desc="Deflating categoricals", total=len(infos), disable=len(infos) < 50
    ):
        col = info.col
        nan = df[col].isna()
        df[col] = df[col].astype(str)
        idx = df[col].isin(info.to_deflate) | nan
        df.loc[idx, col] = np.nan

    if len(infos) > 0:
        w = max(len(info.col) for info in infos) + 2
    else:
        w = 0
    if _warn and len(infos) > 0:
        message = "\n".join(
            [f"{info.col:<{w}} {info.n_total} --> {info.n_keep}" for info in infos]
        )
        messy_inform(
            "Found and 'deflated' a number of categorical variables with "
            "levels that are unlikely to be reliable or useful in prediction. "
            "For each level of categorical variable to be predictively useful, "
            "there must be enough samples to be statistically meaningful (or to "
            "allow some reasonable generalization) in each fold used for fitting "
            "or analyses. Roughly, this means each fold needs to see at least "
            "10-20 samples of each level (depending on how strongly / cleanly "
            "the level relates to other features - ultimately this is just a "
            "heuristic). Assuming k-fold is used for validation, then this means "
            "about k*10 samples per categorical level would a reasonable default "
            "minimum requirement one might use for culling categorical levels. "
            "Under the typical assumption of k=5, this means we require useful / "
            "reliable categorical levels to have 50 samples each.\n\n"
            "Deflated categorical variables (before --> after):\n"
            f"{message}"
        )

    return df


def encode_categoricals(
    df: DataFrame,
    target: str,
    results: InspectionResults,
    warn_explosion: bool = True,
) -> tuple[DataFrame, DataFrame]:
    """

    Returns
    -------
    encoded: DataFrame
        Pandas DataFrame with categorical variables one-hot encoded

    unencoded: DataFrame
        Pandas DataFrame with original categorical variables
    """

    y = df[target]
    df = df.drop(columns=target)
    df = convert_categoricals(df, target)
    df = unify_nans(df)
    df = drop_unusable(df, results, _warn=False)
    df = deflate_categoricals(df, results, _warn=warn_explosion)
    cats = [*results.cats.infos.keys(), *results.binaries.infos.keys()]
    X_cat = df.loc[:, cats].copy(deep=True)
    to_convert = cats

    # below will FAIL if we didn't remove timestamps or etc.
    try:
        bins = sorted(set(results.binaries.cols).intersection(to_convert))
        multis = sorted(set(results.multi_cats).intersection(to_convert))
        new = df
        # note `bins` includes variables that are (1) "constant-binary" (i.e.
        # either a constant value or NaN), (2) true binary (i.e. two unique
        # non-NaN values), or (3) binary plus NaN (two unique non-NaN values
        # and a NaN). In all these cases there is no intepretive confusion or
        # information loss when representing them as:
        #
        # (1) {0, 1}
        # (2) {0, 1}
        # (3) {0, 1} + {0, 1} NaN indicator
        #
        # i.e. by using pd.get_dummies(..., dummy_na=True, drop_first=True)
        new = pd.get_dummies(new, columns=bins, dummy_na=True, drop_first=True)
        new = pd.get_dummies(new, columns=multis, dummy_na=True, drop_first=False)
        if new.columns.has_duplicates:
            dupes = new.columns[new.columns.duplicated()]
            raise ValueError(f"pd.get_dummies created duplicates: {dupes}")
        new = convert_categoricals(new, target=target)
        new = new.astype(float)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            "Could not convert data to floating point after cleaning. Some "
            "messy data must be removed or cleaned before `df-analyze` can "
            "continue. See information above."
        ) from e

    # warn about large number of features
    old_feats = df.shape[1]
    enc_feats = new.shape[1]
    big_cats = results.big_cats
    n_orig = len(to_convert)
    n_cont = old_feats - n_orig
    n_enc_total = enc_feats - n_cont
    n_big_orig = len(big_cats)
    n_small_orig = n_orig - n_big_orig
    n_big_enc = sum(big_cats.values(), start=0)
    n_small_enc = n_enc_total - n_big_enc
    n_small_added = n_small_enc - n_small_orig
    n_big_added = n_big_enc - n_big_orig

    if enc_feats < MAX_PERF_N_FEATURES:
        new = pd.concat([new, y], axis=1)
        return new, df.loc[:, to_convert]

    names = reversed(sorted(big_cats.keys(), key=lambda col: big_cats[col]))
    if n_big_enc > 0:
        info = "\n".join([f"{cat}: {big_cats[cat]} levels" for cat in names])
        big_message = (
            "\n\n"
            "Some of the increase in data size was due to encoding categorical "
            "features with a large (20+) number of levels / categories. These "
            f"features in total added {n_big_added} additional features to "
            "the data. Often, a categorical variable with a large number of "
            "levels will not contain much predictive information in most of "
            "the levels (i.e. only a few classes will be useful). Inspect "
            "df-analyze's univariate association and prediction reports and "
            "consider whether either manually removing some of these features "
            "using the `--drops` option is warranted, or otherwise specify "
            "a filter-based automatic selection method.\n\n"
            f"Large categoricals:\n{info}"
        )
    else:
        big_message = ""

    small_names = sorted(set(to_convert).difference(names))
    if len(small_names) > 0:
        small_message = (
            "\n\n"
            "Some of the increase in data size was due to encoding categorical "
            f"features. These were the features: {small_names}, which in "
            f"total added {n_small_added} additional features to the data. "
            "Inspect df-analyze's univariate association and prediction "
            "reports and consider whether either manually removing some of "
            "these features using the `--drops` option is warranted, or "
            "otherwise specify a filter-based automatic selection method. "
        )
    else:
        small_message = ""

    # handle data explosion due to one-hot encoding
    if (n_enc_total / n_orig) > 10 and warn_explosion:
        messy_inform(
            "Encoding the categoricals of your data has increased the number "
            "of effective features in your data by an order of magnitude, and "
            f"your data also has now over {MAX_PERF_N_FEATURES} effective "
            "features. This will cause performance issues if using stepwise "
            "feature selection methods, or if you do not use any filter-based "
            "selection methods."
            f"{big_message}"
            f"{small_message}"
        )

    new = pd.concat([new, y], axis=1)
    return new, X_cat


def load_as_df(path: Path, spreadsheet: bool) -> DataFrame:
    FILETYPES = [".json", ".csv", ".npy", "xlsx", ".parquet"]
    if path.suffix not in FILETYPES:
        raise ValueError(f"Invalid data file. Currently must be one of: {FILETYPES}")
    if path.suffix == ".json":
        df = pd.read_json(str(path))
    elif path.suffix == ".csv":
        if spreadsheet:
            df = load_spreadsheet(path)[0]
        else:
            df = pd.read_csv(str(path))
    elif path.suffix == ".parquet":
        df = pd.read_parquet(str(path))
    elif path.suffix == ".xlsx":
        if spreadsheet:
            df = load_spreadsheet(path)[0]
        else:
            df = pd.read_excel(str(path))
    elif path.suffix == ".npy":
        arr: ndarray = np.load(str(path), allow_pickle=False)
        if arr.ndim != 2:
            raise RuntimeError(
                f"Invalid NumPy data in {path}. NumPy array must be two-dimensional."
            )
        cols = [f"c{i}" for i in range(arr.shape[1])]
        df = DataFrame(data=arr, columns=cols)
    else:
        raise RuntimeError("Unreachable!")
    return df
