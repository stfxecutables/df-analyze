from math import isnan
from pathlib import Path
from typing import (
    Union,
)


def resolved_path(p: Union[str, Path]) -> Path:
    try:
        path = Path(p)
    except Exception as e:
        raise ValueError(f"Could not interpret string {p} as path") from e
    try:
        path = path.resolve()
    except Exception as e:
        raise ValueError(f"Could not resolve path {path} to valid path.") from e
    return path


def cv_size(cv_str: str) -> Union[float, int]:
    try:
        cv = float(cv_str)
    except Exception as e:
        raise ValueError(
            "Could not convert a `... -size` argument (e.g. --htune-val-size) value to float"
        ) from e
    # validate
    if isnan(cv):
        raise ValueError("NaN is not a valid size")
    if cv <= 0:
        raise ValueError("`... -size` arguments (e.g. --htune-val-size) must be positive")
    if cv == 1:
        raise ValueError(
            "'1' is not a valid value for `... -size` arguments (e.g. --htune-val-size)."
        )
    if (cv > 1) and not cv.is_integer():
        raise ValueError(
            "Passing a float greater than 1.0 for `... -size` arguments "
            "(e.g. --htune-val-size) is not valid. See documentation for "
            "`--htune-val-size` or `--test-val-sizes`."
        )
    if cv > 1:
        return int(cv)
    return cv


def separator(s: str) -> str:
    if s.lower().strip() == "tab":
        return "\t"
    if s.lower().strip() == "newline":
        return "\n"
    return s
