import sys
from pathlib import Path
from typing import (
    Optional,
    Union,
)

import numpy as np
import openml
import pandas as pd
from numpy import ndarray
from openml.datasets import OpenMLDataset
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

DATA = Path(__file__).resolve().parent
CLS = DATA / "testing/classification"
REG = DATA / "testing/regression"


def get_clean_list() -> DataFrame:
    ds = openml.datasets.list_datasets(output_format="dataframe")
    ds.index = ds["did"]  # type: ignore
    ds = ds[ds["status"] == "active"]
    ds = ds.drop(
        columns=["format", "uploader", "did", "status", "MaxNominalAttDistinctValues"]
    )
    # get latest versions
    ds = ds.loc[ds.groupby("name")["version"].nlargest(1).index.droplevel(0)]
    ds = ds.drop(columns="version").convert_dtypes()
    ds = ds.rename(
        columns={
            "NumberOfFeatures": "n_feat",
            "NumberOfInstances": "N",
            "NumberOfInstancesWithMissingValues": "n_nan_samp",
            "NumberOfMissingValues": "nan_total",
            "NumberOfNumericFeatures": "n_numeric",
            "NumberOfSymbolicFeatures": "n_categorical",
            "MajorityClassSize": "n_max_cls",
            "MinorityClassSize": "n_min_cls",
            "NumberOfClasses": "n_cls",
        }
    )
    return ds


def get_regression_info(ds: DataFrame) -> DataFrame:
    reg = ds[ds["n_cls"] == 0].drop(columns=["n_max_cls", "n_min_cls", "n_cls"])
    reg.insert(0, "nan_perc", reg["nan_total"] / reg["N"])
    reg.drop(columns="nan_total", inplace=True)
    reg = reg.sort_values(
        by=["N", "n_feat", "n_categorical", "nan_perc"], ascending=False
    )
    reg = reg[reg["N"] > 300]
    reg = reg[reg["N"] < 100_000]
    reg = reg[reg["n_feat"] < 1000]
    reg = reg[reg["n_categorical"] > 5]  # now we have limited to 22 datasets
    return reg


def get_classification_info(ds: DataFrame) -> DataFrame:
    cls = ds[ds["n_cls"] >= 2]
    cls.insert(0, "perc_max_cls", cls["n_max_cls"] / cls["N"])
    cls.insert(0, "perc_min_cls", cls["n_min_cls"] / cls["N"])
    cls.insert(0, "nan_perc", cls["n_nan_samp"] / cls["N"])
    cls.insert(0, "nan_ratio", cls["nan_total"] / cls["N"])
    cls.drop(
        columns=["n_max_cls", "n_min_cls", "n_nan_samp", "nan_total"],
        inplace=True,
        errors="ignore",
    )

    cls = cls[~cls.name.str.contains("seed_")]

    cls = cls[cls["N"] > 300]
    cls = cls[cls["N"] < 100_000]
    cls = cls[cls["n_feat"] < 1000]
    cls = cls[cls["n_cls"] < 100]
    cls = cls[cls["n_categorical"] > 5]  # now we have limited to 252 datasets
    cls = cls[cls["nan_perc"] > 0.0]  # now down to 39

    cls = cls.sort_values(by=["n_cls", "nan_ratio", "n_feat", "N"], ascending=False)

    return cls


def to_df(
    X: Union[DataFrame, ndarray, csr_matrix],
    y: Optional[Union[DataFrame, Series, ndarray]],
    target: str,
    cols: list[str],
) -> Optional[DataFrame]:
    if isinstance(X, ndarray):
        df = DataFrame(X, columns=cols)
    elif isinstance(X, DataFrame):
        df = X
    elif isinstance(X, csr_matrix):
        df = DataFrame(np.asarray(X), columns=cols)
    else:
        df = DataFrame(np.asarray(X), columns=cols)

    if "target" in df.columns:
        df.rename(columns={"target": "ds_target"}, inplace=True)

    if y is None:
        if target not in df.columns:
            return None

        df.rename(columns={target: "target"}, inplace=True)
        return df

    if not isinstance(y, Series):
        y = Series(data=np.asarray(y).ravel(), name="target")
    else:
        y.name = "target"

    df = pd.concat([df, y], axis=1)
    return df


def save_parquet(dataset: OpenMLDataset, is_cls: bool) -> None:
    X, y, cats, cols = dataset.get_data(dataset_format="dataframe")
    target = dataset.default_target_attribute
    if target is None:
        return
    for i, col in enumerate(cols):
        if col == target:
            cols.pop(i)
            cats.pop(i)
            break

    df = to_df(X, y, target, cols)
    if df is None:
        return
    if "target" not in df.columns:
        raise KeyError(f"Missing target column for {dataset.name}")

    out_parent = CLS if is_cls else REG
    dsname = str(dataset.name)
    if dsname == "Midwest_survey":
        dsname = "Midwest_survey2"
    outdir = out_parent / dsname
    outdir.mkdir(parents=True, exist_ok=True)
    pq = outdir / f"{dsname}.parquet"
    typs = outdir / "types.csv"

    cats, cols = np.asarray(cats), np.asarray(cols)
    df_types = DataFrame({"feature_name": cols, "type": cats})
    if "midwest_survey" in dsname.lower():
        df_types.loc[:, "type"] = True  # are in fact all categorical
    df_types["type"] = df_types["type"].apply(
        lambda x: "categorical" if x else "continuous"
    )

    df_types.to_csv(typs, index=False)
    df.to_parquet(pq)
    size = pq.stat().st_size
    readable = size / 1024
    unit = "KB"
    if readable > 1024:
        readable /= 1024
        unit = "MB"

    print(f"[{readable:>8.1f}{unit}] saved to {pq.relative_to(DATA)}")


if __name__ == "__main__":
    ds = get_clean_list()
    benchmark = openml.study.get_suite(293)
    tasks = openml.tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
    df = openml.evaluations.list_evaluations(
        # function="area_under_roc_curve", tasks=benchmark.tasks, output_format="dataframe"
        function="predictive_accuracy",
        tasks=benchmark.tasks,
        output_format="dataframe",
        size=None,
    )
    reg = get_regression_info(ds)
    cls = get_classification_info(ds)
    sys.exit()

    reg = get_regression_info(ds)
    cls = get_classification_info(ds)

    dids = cls.index.to_series().apply(str).to_list()
    for did in dids:
        dataset = openml.datasets.get_dataset(
            dataset_id=did,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
        save_parquet(dataset=dataset, is_cls=True)

    dids = reg.index.to_series().apply(str).to_list()
    for did in dids:
        dataset = openml.datasets.get_dataset(
            dataset_id=did,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
        if dataset.name == "Midwest_Survey":
            continue
        save_parquet(dataset=dataset, is_cls=False)

    print(reg)
