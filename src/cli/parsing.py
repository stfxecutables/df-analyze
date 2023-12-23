from math import isnan
from pathlib import Path
from typing import Callable, TypeVar, Union, overload
from warnings import warn


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


@overload
def int_or_percent(default: int) -> Callable[[str], int]:
    ...


@overload
def int_or_percent(default: float) -> Callable[[str], float]:
    ...


def int_or_percent_parser(default: Union[int, float]) -> Callable[[str], Union[int, float]]:
    def parser(arg: str) -> Union[int, float]:
        message = f"Setting argument to default value: {default}"
        try:
            parsed = float(arg)
        except Exception as e:
            warn(f"Could not convert argument `{arg}` to float: {e}. {message}")
            parsed = float(default)
        # validate
        if isnan(parsed):
            warn(f"NaN is not a valid size. {message}")
            parsed = float(default)
        if parsed < 0:
            warn(
                "Got a float or integer less than 0 for a percent argument. "
                f"Percentages must be in [0.0, 1.0], and integers must be "
                f"positive or zero. {message}"
            )
            parsed = float(default)
        if (parsed > 1) and not parsed.is_integer():
            warn(
                "Got a float greater than 1.0 for a percent argument. Percentages "
                f"must be in [0.0, 1.0]. {message}"
            )
            parsed = float(default)
        if parsed > 1:
            return int(parsed)
        return parsed

    return parser


def separator(s: str) -> str:
    if s.lower().strip() == "tab":
        return "\t"
    if s.lower().strip() == "newline":
        return "\n"
    return s
