from __future__ import annotations

import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import ctime
from typing import (
    Optional,
    Tuple,
)

from pandas import DataFrame

from src._types import FeatureCleaning, FeatureSelection
from src.utils import Debug

JOBLIB = "__JOBLIB_CACHE__"


class FileType(Enum):
    Interim = "interim"
    Final = "final"
    Feature = "feature"
    Params = "params"
    Univariate = "univariatae"


@dataclass
class ProgramDirs(Debug):
    """Container for various output and caching directories"""

    joblib_cache: Path
    univariate: Path
    feature_selection: Path
    interim_results: Path
    final_results: Path


def setup_io(outdir: Path, add_timestamp: bool = False) -> ProgramDirs:
    out = outdir
    if add_timestamp:
        timestamp = ctime().replace(":", "-").replace("  ", " ").replace(" ", "_")
        out = out / f"{timestamp}"

    dirs = ProgramDirs(
        joblib_cache=out / JOBLIB,
        univariate=out / "univariate",
        feature_selection=out / "selected_features",
        interim_results=out / "interim_results",
        final_results=out / "final_results",
    )
    directory: Path
    for directory in dirs.__dict__.values():
        if directory.name in [JOBLIB, "selected_features"]:
            continue  # joblib will handle creation
        if not directory.exists():
            json = directory / "json"
            csv = directory / "csv"
            json.mkdir(parents=True, exist_ok=True)
            csv.mkdir(parents=True, exist_ok=True)
    return dirs


def try_save(
    program_dirs: ProgramDirs,
    df: DataFrame,
    file_stem: str,
    file_type: FileType,
    selection: Optional[FeatureSelection] = None,
    cleaning: Tuple[FeatureCleaning] = (),
) -> None:
    if file_type is FileType.Interim:
        outdir = program_dirs.interim_results
        desc = "interim results"
    elif file_type is FileType.Final:
        outdir = program_dirs.final_results
        desc = "final results for all options"
    elif file_type is FileType.Feature:
        if selection is None or selection == "none":
            # feature cleaning may have removed features and the cleaned
            # set of features is still worth saving
            selection = "no-selection"
        else:
            selection += "-selection"
        if len(cleaning) > 0:
            clean = f"_cleaning={'-'.join([method for method in cleaning])}"
        else:
            clean = ""
        outdir = program_dirs.feature_selection / f"{selection}{clean}"
        desc = "selected/cleaned features"
    elif file_type is FileType.Params:
        if selection is None or selection == "none":
            selection = "no-selection"
        else:
            selection += "-selection"
        if len(cleaning) > 0:
            clean = f"_cleaning={'-'.join([method for method in cleaning])}"
        else:
            clean = ""
        outdir = program_dirs.feature_selection / f"{selection}{clean}"
        desc = "selected/cleaned features"
    else:
        raise ValueError("Invalid FileType for manual saving")

    json = outdir / f"json/{file_stem}.json"
    csv = outdir / f"csv/{file_stem}.csv"
    json.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_json(json)
        print(f"Saved {desc} to {json}")
    except Exception:
        traceback.print_exc()
        print(f"Failed to save following results to {json}")
        df.to_markdown(tablefmt="simple", floatfmt="0.5f")
    try:
        df.to_csv(csv)
        print(f"Saved {desc} to {csv}")
    except Exception:
        traceback.print_exc()
        print(f"Failed to save {desc} results to {csv}:")
        df.to_markdown(tablefmt="simple", floatfmt="0.5f")
