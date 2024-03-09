from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from math import ceil
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
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from lightgbm.callback import log_evaluation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from typing_extensions import Literal

from data.snp.simulate_snp import simulate_snp_data
from src.analysis.metrics import cramer_v
from src.cli.cli import ProgramOptions
from src.enumerables import ClassifierScorer, WrapperSelection, WrapperSelectionModel
from src.models.dummy import DummyClassifier
from src.models.lgbm import LightGBMClassifier
from src.models.linear import SGDClassifierSelector as SGDLinear
from src.preprocessing.inspection.inspection import ClsTargetInfo
from src.preprocessing.prepare import PreparationInfo, PreparedData
from src.selection.stepwise import StepwiseSelector

DATA = Path(__file__).resolve().parent

AFRICAN = DATA / "EthA2_data.csv.gz"
ESTONIAN = DATA / "EthE2_data.csv.gz"
MERGED = DATA / "merged_estonian_african.parquet"
SNPS = DATA / "estonian_african_snps.parquet"
ENCODED = DATA / "estonian_african_snps_encoded.parquet"
DROPS = ["Name", "ParentM", "ParentF", "EthA", "EthE", "EthK", "EthP"]


def load_raw() -> DataFrame:
    if MERGED.exists():
        print("Loading raw data...")
        return pd.read_parquet(MERGED)

    df1 = pd.read_csv(ESTONIAN)
    df2 = pd.read_csv(AFRICAN)
    df1["target"] = 0
    df2["target"] = 1
    df = pd.concat([df1, df2], axis=0)
    df.drop(columns=DROPS, inplace=True)
    df.to_parquet(MERGED)
    return df


def get_snp(df: DataFrame, i: int) -> Optional[Series]:
    a1 = df["Allele1"].apply(lambda s: s[i])
    a2 = df["Allele2"].apply(lambda s: s[i])
    snp = a1 + a2  # concat
    snp.name = f"{i}"
    if len(snp.unique()) > 1:
        return snp.astype("category")


def make_snps(subset: Optional[int] = None) -> DataFrame:
    if SNPS.exists() and subset is None:
        df = pd.read_parquet(SNPS)
        if "M" in df["sex"].unique():
            df["sex"] = df["sex"].apply(lambda x: 0 if x == "F" else 1)
        return df

    print("Loading raw data...")
    df = load_raw()
    P = len(df.iloc[0].Allele1)  # 39_108
    if subset is not None:
        idx = np.random.permutation(len(df))[:subset]
        df = df.iloc[idx]

    all_snps = Parallel()(
        delayed(get_snp)(df=df, i=i) for i in tqdm(range(P), desc="Converting to SNPs")
    )
    snps = [snp for snp in tqdm(all_snps, desc="Filtering...") if snp is not None]

    print("Concatenating")
    snps = pd.concat(snps, axis=1)
    snps.insert(0, "sex", df["Gender"].apply(lambda x: 0 if x == "F" else 1))
    snps["target"] = df["target"]
    snps.to_parquet(SNPS)

    return snps


def load_label_encoded() -> tuple[DataFrame, Series]:
    if ENCODED.exists():
        print("Loading Encoded SNP data...")
        df = pd.read_parquet(ENCODED)
        X = df.drop(columns="target")
        y = df["target"]
        return X, y.reset_index(drop=True)

    print("Loading SNP data...")
    df = make_snps()
    print("Label-encoding categoricals...")
    cats = df.select_dtypes(["category"]).columns
    df[cats] = df[cats].apply(lambda x: x.cat.codes)
    df.to_parquet(ENCODED)

    X = df.drop(columns="target")
    y = df["target"]
    return X, y


def random_feature_subset(
    X: DataFrame, n_min: int = 2, n_max: int = 7
) -> tuple[DataFrame, list[str]]:
    n_feat = np.random.randint(n_min, n_max + 1)
    cols = np.random.choice(X.columns.to_list(), size=n_feat, replace=False).tolist()
    return X[cols].reset_index(drop=True), cols


def test_actual_data() -> None:
    N_sub = 5000
    X, y = load_label_encoded()

    idx = np.random.randint(0, 2, size=len(X), dtype=bool)
    X_tr, y_tr, X_ts, y_ts = X.loc[idx], y.loc[idx], X.loc[~idx], y.loc[~idx]
    model = LGBMClassifier(verbosity=1, n_jobs=-1, force_col_wise=True)
    print("Using LightGBM for RFE...")
    rfe = RFE(model, n_features_to_select=200, step=0.25)
    rfe.fit(X_tr, y_tr)
    cols = rfe.get_feature_names_out(X.columns.to_list())
    base = rfe.score(X_ts, y_ts)
    X = X[cols]
    # print("Fitting LightGBM...")
    # model.fit(X_tr, y_tr, callbacks=[log_evaluation()])
    # base = model.score(X_ts, y_ts)

    # base = 0.9998673212153376  # base LGBMClassifier score, no tuning
    print("Base performance:", base)

    idx = np.random.permutation(len(X))[:N_sub]
    X = X.iloc[idx]
    y = y.iloc[idx]

    N = 1000

    # dummy = DummyClassifier()
    # study = dummy.htune_optuna(
    #     X_train=X,
    #     y_train=y,
    #     metric=ClassifierScorer.BalancedAccuracy,
    #     n_trials=20,
    #     n_jobs=-1,
    #     verbosity=0,
    # )
    # dumb = study.best_value
    dumb = 0.5102
    print("Dummy performance:", dumb)

    # model = SGDLinear()
    # study = model.htune_optuna(
    #     X_train=X,
    #     y_train=y,
    #     metric=ClassifierScorer.BalancedAccuracy,
    #     n_trials=20,
    #     n_jobs=-1,
    #     verbosity=2,
    # )
    # base = study.best_value
    # print("Base performance:", base)

    feats, accs = [], []
    scores = []
    for i in range(10):
        X_sub_all = random_feature_subset(X)
        idx = np.random.permutation(len(X_sub_all))[:2000]
        X_sub = X_sub_all.iloc[idx].reset_index(drop=True)
        y_sub = y.iloc[idx].reset_index(drop=True)
        model = LightGBMClassifier()
        study = model.htune_optuna(
            X_train=X_sub,
            y_train=y_sub,
            metric=ClassifierScorer.PPV,
            n_trials=20,
            n_jobs=-1,
            verbosity=0,
        )
        score = model.htune_eval(
            X_train=X_sub,
            y_train=y_sub,
            X_test=X_sub_all,
            y_test=y.reset_index(drop=True),
        )[0]
        score.index = score["metric"]
        score = score["holdout"].to_frame().T.copy()
        cols = X_sub.columns.to_list()
        feats.append(cols)
        accs.append(study.best_value)
        scores.append(score)
        print(cols, study.best_value)

    score = pd.concat(scores, axis=0, ignore_index=True)
    score["feats"] = [str(feat) for feat in feats]
    score["n_feat"] = [len(feat) for feat in feats]
    print(score.to_markdown(tablefmt="simple", index=False))

    # print("Dummy performance:", dumb)
    # print("Base performance:", base)
    print("Dummy performance:", dumb)


def test_feature_elim(
    n_samples=2000,
    n_snp=30_000,
    n_predictive_sets=2,
    predictive_set_min_size=3,
    predictive_set_max_size=8,
    min_n_variants=2,
    max_n_variants=8,
    predictiveness=0.9,
    n_predictive_combinations=2,
    min_variant_samples=50,
) -> None:
    print("Generating data ... ", end="", flush=True)
    df, df_pred, y = simulate_snp_data(
        n_samples=n_samples * 2,
        n_snp=n_snp,
        n_predictive_sets=n_predictive_sets,
        predictive_set_min_size=predictive_set_min_size,
        predictive_set_max_size=predictive_set_max_size,
        min_n_variants=min_n_variants,
        max_n_variants=max_n_variants,
        predictiveness=predictiveness,
        n_predictive_combinations=n_predictive_combinations,
        min_variant_samples=min_variant_samples,
    )
    print("done")
    # shuffle feature orders to prevent first-feature advantage in algorithms
    df = df.iloc[:, np.random.permutation(df.shape[1])]

    # normalize for ridge regression
    # df = df.astype(np.float64)
    # df[:] = MinMaxScaler().fit_transform(df.values)

    opts = ProgramOptions.random(ds=None, outdir=DATA / "selection")
    opts.wrapper_model = WrapperSelectionModel.Linear
    opts.wrapper_select = WrapperSelection.StepUp

    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    idx_tr, idx_ts = next(ss.split(y, y))
    X_tr, X_ts, y_tr, y_ts = (
        df.iloc[idx_tr],
        df.iloc[idx_ts],
        y.iloc[idx_tr],
        y.iloc[idx_ts],
    )

    unqs, cnts = np.unique(y_tr, return_counts=True)
    info = PreparationInfo(
        is_classification=True,
        original_shape=X_tr.shape,
        final_shape=X_tr.shape,
        n_samples_dropped_via_target_NaNs=0,
        n_cont_indicator_added=0,
        runtimes={},
        target_info=ClsTargetInfo(
            name="target",
            inflation=None,
            unqs=unqs,
            cnts=cnts,
            p_max_cls=np.max(cnts) / cnts.sum(),
            p_min_cls=np.min(cnts) / cnts.sum(),
            p_nan=0.0,
        ),
    )
    prep = PreparedData(
        X=X_tr,
        X_cont=DataFrame(np.empty([X_tr.shape[0], 1])),
        X_cat=X_tr,
        y=y_tr,
        labels=None,
        inspection=None,
        info=info,
        phase="train",
    )
    selector = StepwiseSelector(
        prep_train=prep, options=opts, n_features=20, direction="forward"
    )
    selector.fit()
    selector.scores

    selected = df.loc[:, np.ravel(selector.support_)].columns.to_list()
    phonies = [s for s in selected if "_" not in s]
    corrs = np.empty([len(phonies), df_pred.shape[1]], dtype=np.float64)
    tagged = set()
    for i, phony in enumerate(phonies):
        for j, predcol in enumerate(df_pred.columns):
            corr = cramer_v(df[phony], df_pred[predcol])
            corrs[i, j] = corr
            if abs(corr) > 0.9:
                tagged.add(predcol)
    corrs = np.max(corrs, axis=0)  # largest correlations with predictors

    n_true = len([s for s in selected if "_" in s])
    n_phony = len(selected) - n_true
    n_possible = df_pred.shape[1]
    n_tagged = len(tagged)

    print(
        f"StepUp: True features selected: {n_true} / {n_possible} ({n_true / n_possible:0.4f})"
    )
    print(
        f"StepUp: True / phony selected: {n_true} / {n_phony} ({n_true / len(selected):0.4f})"
    )
    print(f"StepUp scores: {selector.scores}")
    print(
        f"StepUp Selected feature Cramer V correlations with predictive features: {DataFrame(corrs.ravel()).describe()}"
    )
    print(
        f"StepUp Selected Features with Cramer V > 0.90: {(np.abs(corrs) >= 0.90).sum()}"
    )
    print(
        f"StepUp True Predictive Features Tagged: {n_tagged} / {df_pred.shape[1]} ({n_tagged / df_pred.shape[1]})"
    )

    model = LGBMClassifier(verbosity=-1, n_jobs=-1, force_col_wise=True)
    model.fit(X_tr.loc[:, selected], y_tr)
    score = model.score(X_ts.loc[:, selected], y_ts)
    print(f"StepUp Selection Test performance: {score}")

    # NOTE
    # LASSO (L1 regularized LR-CV), and Relief (MultiSURF) methods fail
    # beyond horribly here, they fail to select true features. LightGBM and
    # RidgeClassifierCV do though.

    ss = StratifiedKFold(n_splits=3)
    importances = np.zeros(shape=[df.shape[1]], dtype=np.uint64)
    for idx_tr, idx_ts in tqdm(ss.split(y_tr, y_tr), desc="Fitting k-fold LGBM"):
        model = LGBMClassifier(verbosity=-1, n_jobs=-1, force_col_wise=True)
        model.fit(X_tr.iloc[idx_tr], y_tr.iloc[idx_tr])
        importances = importances + model.feature_importances_

    idx = importances > 0
    selected = df.loc[:, idx].columns.to_list()
    phonies = [s for s in selected if "_" not in s]
    corrs = np.empty([len(phonies), df_pred.shape[1]], dtype=np.float64)
    tagged = set()
    for i, phony in enumerate(phonies):
        for j, predcol in enumerate(df_pred.columns):
            corr = cramer_v(df[phony], df_pred[predcol])
            corrs[i, j] = corr
            if abs(corr) > 0.9:
                tagged.add(predcol)
    corrs = np.max(corrs, axis=0)  # largest correlations with predictors

    n_true = len([s for s in selected if "_" in s])
    n_phony = len(selected) - n_true
    n_possible = df_pred.shape[1]
    print(
        f"LGBM: True features selected: {n_true} / {n_possible} ({n_true / n_possible:0.4f})"
    )
    print(
        f"LGBM: True / phony selected: {n_true} / {n_phony} ({n_true / len(selected):0.4f})"
    )
    model = LGBMClassifier(verbosity=-1, n_jobs=-1, force_col_wise=True)
    model.fit(X_tr.loc[:, selected], y_tr)
    score = model.score(X_ts.loc[:, selected], y_ts)
    print(f"LGBM Selection Test performance: {score}")
    print(
        f"LGBM Selected Cramer V correlations with predictive features: {DataFrame(corrs.ravel()).describe()}"
    )
    print(f"LGBM Selected Features with Cramer V > 0.90: {(np.abs(corrs) >= 0.90).sum()}")
    print(
        f"LGBM True Predictive Features Tagged: {n_tagged} / {df_pred.shape[1]} ({n_tagged / df_pred.shape[1]})"
    )

    # ss = StratifiedKFold(n_splits=3)
    # importances = np.zeros(shape=[df.shape[1]], dtype=np.float64)
    # for idx_tr, idx_ts in tqdm(
    #     ss.split(y_tr, y_tr), desc="Fitting k-fold RidgeClassifer-cv"
    # ):
    #     model = RidgeClassifierCV(
    #         alphas=[0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
    #         cv=3,
    #     )
    #     model.fit(X_tr.iloc[idx_tr], y_tr.iloc[idx_tr])
    #     importances = importances + model.coef_

    # p = 100 * (1 - 500 / n_snp)  # min 500 features
    # imin = np.percentile(importances.ravel(), p)
    # idx = np.abs(importances) > imin
    # selected = df.loc[:, np.ravel(idx)].columns.to_list()
    # n_true = len([s for s in selected if "_" in s])
    # n_phony = len(selected) - n_true
    # n_possible = df_pred.shape[1]
    # print(
    #     f"Ridge: True features selected: {n_true} / {n_possible} ({n_true / n_possible:0.4f})"
    # )
    # print(
    #     f"Ridge: True / phony selected: {n_true} / {n_phony} ({n_true / len(selected):0.4f})"
    # )


def test_simulated_data(
    n_samples=2000,
    n_snp=30_000,
    n_predictive_sets=2,
    predictive_set_min_size=3,
    predictive_set_max_size=8,
    min_n_variants=2,
    max_n_variants=8,
    predictiveness=0.9,
    n_predictive_combinations=2,
    min_variant_samples=20,
) -> None:
    df, df_pred, y = simulate_snp_data(
        n_samples=n_samples * 2,
        n_snp=n_snp,
        n_predictive_sets=n_predictive_sets,
        predictive_set_min_size=predictive_set_min_size,
        predictive_set_max_size=predictive_set_max_size,
        min_n_variants=min_n_variants,
        max_n_variants=max_n_variants,
        predictiveness=predictiveness,
        n_predictive_combinations=n_predictive_combinations,
        min_variant_samples=min_variant_samples,
    )

    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    idx_tr, idx_ts = next(ss.split(y, y))
    X_tr, X_ts, y_tr, y_ts = (
        df.iloc[idx_tr],
        df.iloc[idx_ts],
        y.iloc[idx_tr],
        y.iloc[idx_ts],
    )
    model = LGBMClassifier(verbosity=1, n_jobs=-1, force_col_wise=True)
    print("Using LightGBM for RFE...")
    # rfe = RFE(model, n_features_to_select=200, step=0.25)
    rfe = RFECV(model, min_features_to_select=200, step=0.25, cv=3)
    rfe.fit(X_tr, y_tr)
    cols = rfe.get_feature_names_out(df.columns.to_list())
    base = rfe.score(X_ts, y_ts)
    X = df[cols]
    X_tr, X_ts = X.iloc[idx_tr], X.iloc[idx_ts]
    # print("Fitting LightGBM...")
    # model.fit(X_tr, y_tr, callbacks=[log_evaluation()])
    # base = model.score(X_ts, y_ts)

    # base = 0.9998673212153376  # base LGBMClassifier score, no tuning
    print("Base performance:", base)

    dumb = max(y.mean(), 1 - y.mean())
    print("Dummy performance:", dumb)

    # model = SGDLinear()
    # study = model.htune_optuna(
    #     X_train=X,
    #     y_train=y,
    #     metric=ClassifierScorer.BalancedAccuracy,
    #     n_trials=20,
    #     n_jobs=-1,
    #     verbosity=2,
    # )
    # base = study.best_value
    # print("Base performance:", base)

    feats, accs = [], []
    scores = []
    for i in range(10):
        X_sub_all, cols = random_feature_subset(X_tr)
        X_tt = X_ts[cols].reset_index(drop=True)
        idx = np.random.permutation(len(X_sub_all))[:2000]
        X_sub = X_sub_all.iloc[idx].reset_index(drop=True)
        y_sub = y.iloc[idx].reset_index(drop=True)
        model = LightGBMClassifier()
        study = model.htune_optuna(
            X_train=X_sub,
            y_train=y_sub,
            metric=ClassifierScorer.PPV,
            n_trials=20,
            n_jobs=-1,
            verbosity=0,
        )
        score = model.htune_eval(
            X_train=X_sub,
            y_train=y_sub,
            X_test=X_tt,
            y_test=y_ts.reset_index(drop=True),
        )[0]
        score.index = score["metric"]
        score = score["holdout"].to_frame().T.copy()
        cols = X_sub.columns.to_list()
        feats.append(cols)
        accs.append(study.best_value)
        scores.append(score)
        print(cols, study.best_value)

    score = pd.concat(scores, axis=0, ignore_index=True)
    score["feats"] = [str(feat) for feat in feats]
    score["n_feat"] = [len(feat) for feat in feats]
    print(score.to_markdown(tablefmt="simple", index=False))

    print("Base performance:", base)
    print("Dummy performance:", dumb)


if __name__ == "__main__":
    ...
    N = 2000
    # P = 30_000
    P = 2000
    # test_simulated_data(
    #     n_samples=N,
    #     n_snp=P,
    #     n_predictive_sets=1,
    #     predictive_set_min_size=3,
    #     predictive_set_max_size=8,
    #     min_n_variants=2,
    #     max_n_variants=8,
    #     predictiveness=0.9,
    #     n_predictive_combinations=2,
    #     min_variant_samples=ceil(0.01 * N),
    # )
    for _ in range(10):
        test_feature_elim(
            n_samples=N,
            n_snp=P,
            n_predictive_sets=np.random.randint(2, 4),
            predictive_set_min_size=3,
            predictive_set_max_size=20,
            min_n_variants=2,
            max_n_variants=8,
            predictiveness=0.9,
            n_predictive_combinations=np.random.randint(1, 4),
            min_variant_samples=ceil(0.02 * N),
        )
