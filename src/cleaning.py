from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.io import loadmat

from src._constants import CLEAN_JSON, DATA_JSON, DATAFILE
from src.data import DataResource


class MCIC:
    """Functions for cleaning MCIC Freesurfer data. Remnants from first version.
    TODO: DEPRECATE
    """

    @staticmethod
    def clean_fs_label(s: str) -> str:
        """Clean the FreeSurfer column labels to ensure brevity.

        Notes
        -----
        The strings are labels of the form:

            "\\stats\\wmparc.stats XXXX XXXX                        "
            "\\stats\\aseg.stats XXXX XXXX                        "

        i.e. there are large amounts of trailing spaces and there may or may not be
        a comma preceding those trailing spaces.
        """
        # get to e.g. 'aseg Brain Segmentation Volume, '
        shorter = s.replace("\\stats\\", "").replace(".stats", "").replace("  ", "")
        while shorter[-1] in [" ", ","]:
            shorter = shorter[:-1]
        return shorter

    @staticmethod
    def reformat_matlab_schizo(path: PathLike) -> DataFrame:
        """Generate a clean DataFrame from the .mat data

        Notes
        -----
        The data saved in the .mat file looks like this:

            [0] ClinicalVariableLabels              (1, 93)     dtype=object
            [1] thisSubjectLabels                   (4784,)     dtype=<U98
            [2] SchizophreniaMeasurements           (99, 4784)  dtype=float64
            [3] SchizophreniaAges                   (99, 1)     dtype=uint8
            [4] SchizophreniaGender                 (99, 1)     dtype=uint8
            [5] SchizophreniaClinicalVariablesCell  (99, 93)    dtype=object
            [6] HealthyMeasurements                 (77, 4784)  dtype=float64
            [7] HealthyAges                         (77, 1)     dtype=float64
            [8] HealthyGender                       (77, 1)     dtype=uint8

        Gender variables are all either 0 or 1 only. Two healthy subjects are missing
        age information.
        """
        p = Path(path)
        file = str(p.resolve())
        data = loadmat(file)
        varnames = [key for key in list(data.keys()) if "__" not in key]
        for i, v in enumerate(varnames):
            print(f"[{i}] {v:<35} {str(data[v].shape):<11} dtype={data[v].dtype}")

        # clinical_target_names = [a[0] for a in data["ClinicalVariableLabels"][0].tolist()]
        X_health_freesurfer = data["HealthyMeasurements"]
        X_schizo_freesurfer = data["SchizophreniaMeasurements"]
        age_health = data["HealthyAges"].astype(np.float64)
        age_schizo = data["SchizophreniaAges"].astype(np.float64)
        sex_health = data["HealthyGender"].astype(np.float64)
        sex_schizo = data["SchizophreniaGender"].astype(np.float64)

        X_health = np.concatenate([X_health_freesurfer, age_health, sex_health], axis=1)
        X_schizo = np.concatenate([X_schizo_freesurfer, age_schizo, sex_schizo], axis=1)
        y_health = np.zeros(X_health.shape[0]).ravel()
        y_schizo = np.ones(X_schizo.shape[0]).ravel()

        X = np.concatenate([X_health, X_schizo], axis=0)
        y = np.concatenate([y_health, y_schizo], axis=0)
        feature_names = list(map(MCIC.clean_fs_label, data["thisSubjectLabels"].tolist())) + [
            "Age",
            "Sex",
        ]
        # there are duplicate column names for some reason...
        feature_names = [f"{i}__{fname}" for fname, i in enumerate(feature_names)]
        df = DataFrame(data=X, columns=feature_names)
        df["target"] = y
        df.to_json(DATA_JSON)
        print(f"Saved processed DataFrame to:\r\n{DATA_JSON}")
        return df

    @staticmethod
    def load_data() -> DataFrame:
        if DATA_JSON.exists():
            return pd.read_json(DATA_JSON)
        return MCIC.reformat_matlab_schizo(DATAFILE)

    @staticmethod
    def get_clean_data() -> DataFrame:
        """Perform minimal cleaning, like removing NaN features"""
        if CLEAN_JSON.exists():
            return pd.read_json(CLEAN_JSON)
        df = MCIC.load_data()
        print("Shape before dropping:", df.shape)
        inspect_data(df)
        df = remove_nan_features(df)
        df = remove_nan_samples(df)
        print("Shape after dropping:", df.shape)
        inspect_data(df)
        df.to_json(CLEAN_JSON)
        return df


def load_as_df(path: Path) -> Union[DataFrame, ndarray]:
    FILETYPES = [".json", ".csv", ".npy"]
    if path.suffix not in FILETYPES:
        raise ValueError("Invalid data file. Currently must be one of ")
    if path.suffix == ".json":
        df = pd.read_json(str(path))
    elif path.suffix == ".csv":
        df = pd.read_csv(str(path))
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


def remove_nan_features(df: DataFrame) -> DataFrame:
    """Remove columns (features) that are ALL NaN"""
    return df.dropna(axis=1, how="any").dropna(axis=1, how="any")


def remove_nan_samples(df: DataFrame) -> DataFrame:
    """Remove rows (samples) that have ANY NaN"""
    return df.dropna(axis=0, how="any").dropna(axis=0, how="any")


def get_clean_data(resource: DataResource) -> DataFrame:
    """Perform minimal cleaning, like removing NaN features"""
    if CLEAN_JSON.exists():
        return pd.read_json(CLEAN_JSON)
    df = load_as_df(path)
    print("Shape before dropping:", df.shape)
    inspect_data(df)
    df = remove_nan_features(df)
    df = remove_nan_samples(df)
    print("Shape after dropping:", df.shape)
    inspect_data(df)
    df.to_json(CLEAN_JSON)
    return df


def inspect_data(df: DataFrame) -> DataFrame:
    A = df.to_numpy()
    nan_rows = [i for i in range(A.shape[0]) if np.sum(np.isnan(A[i])) > 0]
    print("NaN rows:\n", nan_rows)
    nan_cols = {}
    for i in range(A.shape[1]):
        nan_count = np.sum(np.isnan(A[:, i]))
        if nan_count > 0:
            nan_cols[i] = nan_count
    print("NaN cols:")
    for idx, count in nan_cols.items():
        print(f"{idx}: {count}", end=", ")
    print("")


if __name__ == "__main__":
    df = get_clean_data()
