from scipy.io import loadmat
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series

from os import PathLike


DATAFILE = Path(__file__).resolve().parent.parent / "data/MCICFreeSurfer.mat"


def clean_fs_label(s: str) -> str:
    """
    The strings are labels of the form:

    "\\stats\\wmparc.stats XXXX XXXX                        "
    "\\stats\\aseg.stats XXXX XXXX                        "

    """
    # get to e.g. 'aseg Brain Segmentation Volume, '
    shorter = s.replace("\\stats\\", "").replace(".stats", "").replace("  ", "")
    while shorter[-1] in [" ", ","]:
        shorter = shorter[:-1]
    return shorter


def load_matlab_schizo(path: PathLike) -> DataFrame:
    """
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

    Gender variables are all either 0 or 1 only.
    """
    file = str(Path(path).resolve())
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
    feature_names = list(map(clean_fs_label, data["thisSubjectLabels"].tolist())) + ["Age", "Sex"]
    df = DataFrame(data=X, columns=feature_names)
    df["target"] = y
    return df


if __name__ == "__main__":
    load_matlab_schizo(DATAFILE)
