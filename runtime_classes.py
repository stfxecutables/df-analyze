from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from pandas import DataFrame, Index, Series
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from df_analyze.testing.datasets import (
    FAST_INSPECTION,
    MEDIUM_INSPECTION,
    SLOW_INSPECTION,
)

matplotlib.use("QtAgg")


def load_rich_runtime_info() -> DataFrame:
    DATASETS = FAST_INSPECTION + MEDIUM_INSPECTION + SLOW_INSPECTION
    CLASSES = (
        ["fast" for _ in FAST_INSPECTION]
        + ["med" for _ in MEDIUM_INSPECTION]
        + ["slow" for _ in SLOW_INSPECTION]
    )
    rts = pd.read_csv(ROOT / "cc_results/prev_jobs.txt")
    rts.index = Index(rts["jobid"])
    rts = rts.drop(columns="jobid")

    dfs = []
    for i, ((dsname, ds), cls) in enumerate(zip(DATASETS, CLASSES)):
        n, p = ds.shape
        is_cls = ds.is_classification
        dfs.append(
            DataFrame(
                dict(dsname=dsname, n=n, p=p, cls=cls, is_cls=is_cls),
                index=Series(name="jobid", data=[i]),
            )
        )
    df = pd.concat(dfs, axis=0)
    df = pd.concat([df, rts], axis=1)
    df["elapsed"] = pd.to_timedelta(df["elapsed"])
    df["elapsed"] = df["elapsed"].apply(lambda t: t.seconds)
    df = df.rename(columns={"elapsed": "seconds"})
    df["minutes"] = (df["seconds"] / 60).round(1)
    df["hours"] = (df["seconds"] / 3600).round(1)
    return df


def fit_runtimes() -> None:
    df = load_rich_runtime_info()
    df = df.loc[df.state == "COMPLETED"].drop(columns="state")
    print(df)
    y = df["seconds"]
    n, p = df["n"], df["p"]
    degree = 3
    pf = PolynomialFeatures(degree=degree, include_bias=True)
    df["log(n)"] = np.log(df["n"])
    df["log(p)"] = np.log(df["p"])
    cols = ["n", "log(n)", "p"]
    X = pf.fit_transform(df.loc[:, cols])
    # [1, n, p, | n^2, np, p^2 |, n^3, n^2p, np^2, p^3]
    print(pf.powers_)  # shape [m, degree]
    expressions = []
    for m in range(pf.powers_.shape[0]):
        powers = pf.powers_[m]
        terms = []
        if np.all(powers == 0):
            terms.append("1")
        else:
            for i, exp in enumerate(powers):
                if exp == 0:
                    continue
                if exp > 1:
                    terms.append(f"{cols[i]}^{exp}")
                else:
                    terms.append(f"{cols[i]}")
        term = "*".join(terms)
        expressions.append(term)
    print(" + ".join(expressions))

    linear = LinearRegression(fit_intercept=False, n_jobs=8).fit(X, y)
    y_pred = linear.predict(X)
    score = np.mean(np.abs(y_pred - y))
    coef = linear.coef_
    print(f"Linear: degree={degree}, score={score}")

    idx = np.argsort(-np.abs(coef[1:]))  # ignore intercept
    for n_sub in [1, 2, 3]:
        idx_sub = idx[:n_sub].tolist()
        X_sub = np.stack([X[:, i] for i in idx_sub], axis=1)
        linear = LinearRegression(fit_intercept=False, n_jobs=8).fit(X_sub, y)
        y_pred = linear.predict(X_sub)
        score = np.mean(np.abs(y_pred - y))
        coef = linear.coef_

        eqn = "+".join(np.array(expressions[1:])[idx_sub].tolist())
        # eqn = f"{coef[0]} + {eqn}"
        print(f"Linear sub={n_sub}: degree={degree}, score={score}, eqn={eqn}")
        print(coef)

    n_sub = 3
    idx_sub = idx[:n_sub]
    X_sub = np.stack([X[:, i] for i in idx_sub], axis=1)
    linear = LinearRegression(fit_intercept=False, n_jobs=8).fit(X_sub, y)
    coef = linear.coef_
    y_pred = linear.predict(X_sub)
    score = np.mean(np.abs(y_pred - y))
    eqn = "+".join(np.array(expressions[1:])[idx_sub].tolist())
    print(f"\nLinear sub={n_sub}: degree={degree}, score={score}, eqn={eqn}")
    print(coef)

    n = df["n"]
    p = df["p"]
    X = df.loc[:, ["n", "p"]]
    X_log = df.loc[:, ["n", "log(n)"]]
    X_n = df.loc[:, ["n"]]
    m_np = LinearRegression(fit_intercept=False, n_jobs=8).fit(X, y)
    m_nlogn = LinearRegression(fit_intercept=False, n_jobs=8).fit(X_log, y)
    m_n = LinearRegression(fit_intercept=False, n_jobs=8).fit(X_n, y)

    y_np = m_np.predict(X)
    y_nlogn = m_nlogn.predict(X_log)
    y_n = m_n.predict(X_n)
    y_h = n + 20 * p

    score_np = np.mean(np.abs(y_np - y))
    score_nlogn = np.mean(np.abs(y_nlogn - y))
    score_n = np.mean(np.abs(y_n - y))
    score_h = np.mean(np.abs(y_h - y))

    np.testing.assert_allclose(y_np, m_np.coef_[0] * n + m_np.coef_[1] * p)

    print(f"n + p: score={score_np}")
    print(m_np.coef_)
    print(f"n + log(n): score={score_nlogn}")
    print(m_nlogn.coef_)
    print(f"n: score={score_n}")
    print(m_n.coef_)
    print(f"heuristic: score={score_h}")
    print([1, 20])

    df["pred_np"] = y_np
    df["pred_nlogn"] = y_nlogn
    df["pred_n"] = y_n
    df["pred_h"] = y_h

    df["err_np"] = y - y_np
    df["err_nlogn"] = y - y_nlogn
    df["err_n"] = y - y_n
    df["err_h"] = y - y_h

    plt.hist(df["err_h"] / 3600, bins=8, color="black")
    plt.show()
    mean_np, med_np = df["err_np"].mean(), df["err_np"].median()
    mean_nlogn, med_nlogn = df["err_nlogn"].mean(), df["err_nlogn"].median()
    mean_n, med_n = df["err_n"].mean(), df["err_n"].median()

    fig, axes = plt.subplots(ncols=2)
    sbn.scatterplot(
        data=df, hue="cls", x="n", y="p", size="seconds", markers="state", ax=axes[0]
    )
    sbn.scatterplot(data=df, x="n", y="pred_np", size="p", ax=axes[1], color="orange")
    sbn.scatterplot(data=df, x="n", y="pred_nlogn", size="p", ax=axes[1], color="blue")
    sbn.scatterplot(data=df, x="n", y="pred_n", size="p", ax=axes[1], color="purple")
    sbn.scatterplot(data=df, x="n", y="pred_h", size="p", ax=axes[1], color="green")
    sbn.scatterplot(data=df, x="n", y="seconds", size="p", ax=axes[1], color="black")

    # sbn.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    axes[1].set_xscale("log")
    fig.set_size_inches(w=17, h=6)
    fig.tight_layout()

    plt.show()


def feature_classes(p: int) -> str:
    if p < 100:
        return "p < 100"
    elif p < 150:
        return "p < 150"
    else:
        return "200 <= p < 400"


if __name__ == "__main__":
    # fit_runtimes()
    pd.options.display.max_rows = 100
    df = load_rich_runtime_info()
    df = df.loc[df["state"] != "FAILED"]
    df["n + p"] = df["n"] + df["p"]
    df["n_feat"] = df["p"].apply(feature_classes)
    print(df)
    # plt.hist(df.p, bins=20, color="black")
    # plt.show()
    palette = {"COMPLETED": "black", "TIMEOUT": "red"}
    palette = {
        "p < 100": "black",
        "p < 150": "orange",
        "200 <= p < 400": "red",
    }
    markers = {"COMPLETED": ".", "TIMEOUT": "x"}
    fix, axes = plt.subplots(ncols=2)
    sbn.scatterplot(
        data=df,
        x="n",
        y="hours",
        hue="n_feat",
        palette=palette,  # type: ignore
        # palette="bright",
        style="state",
        size="p",
        ax=axes[0],
    )
    sbn.scatterplot(
        data=df,
        x="p",
        y="hours",
        hue="n_feat",
        palette=palette,  # type: ignore
        # palette="bright",
        style="state",
        size="n",
        ax=axes[1],
    )
    sbn.move_legend(axes[0], "upper left", bbox_to_anchor=(1, 1))
    sbn.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1))
    fig = plt.gcf()
    fig.set_size_inches(w=15, h=6)
    fig.suptitle(
        "df-analyze runtimes (Niagara, all options, step-up selecting 10 features)"
    )
    fig.tight_layout()
    plt.show()
