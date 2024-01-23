from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from copy import deepcopy
from dataclasses import is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Tuple,
)

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

from src.cli.cli import ProgramOptions
from src.hypertune import EvaluationResults
from src.preprocessing.inspection.inspection import InspectionResults
from src.selection.embedded import EmbedSelected
from src.selection.wrapper import WrapperSelected
from src.testing.datasets import TestDataset, all_ds


@all_ds
def test_random_options(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    for _ in range(100):
        ProgramOptions.random(ds)


@all_ds
def test_hashing(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    for _ in range(100):
        opts1 = ProgramOptions.random(ds)
        opts2 = ProgramOptions.random(ds)
        opts11 = deepcopy(opts1)
        opts22 = deepcopy(opts2)
        hsh1 = opts1.hash()
        hsh2 = opts2.hash()
        hsh11 = opts11.hash()
        hsh22 = opts22.hash()
        assert hsh1 != hsh2
        assert hsh1 == hsh11
        assert hsh2 == hsh22


def are_equal(obj1: Any, obj2: Any) -> bool:
    if is_dataclass(obj1) and is_dataclass(obj2):
        if obj1.__dict__.keys() != obj2.__dict__.keys():
            return False

        for key in obj1.__dict__.keys():
            attr1 = obj1.__dict__[key]
            attr2 = obj2.__dict__[key]
            eq = are_equal(attr1, attr2)
            if not eq:
                return False
        return True
    elif isinstance(obj1, ndarray) and isinstance(obj2, ndarray):
        return (obj1.ravel().round(8) == obj2.ravel().round(8)).all()
    elif isinstance(obj1, (DataFrame, Series)) and isinstance(obj2, (DataFrame, Series)):
        return np.all((obj1.round(8).values == obj2.round(8).values)).item()
    else:
        return obj1 == obj2


@all_ds
def test_json(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    try:
        with TemporaryDirectory() as tempdir:
            for _ in range(10):
                outdir = Path(tempdir)
                opts = ProgramOptions.random(ds, outdir=outdir)
                opts.to_json()
                assert outdir.exists()
                assert opts.program_dirs.options is not None
                assert opts.program_dirs.options.exists()
                opts2 = ProgramOptions.from_json(opts.program_dirs.root)
                for attr in opts.__dict__:
                    attr1 = getattr(opts, attr)
                    attr2 = getattr(opts2, attr)
                    if not are_equal(attr1, attr2):
                        raise ValueError(
                            f"Saved attribute {attr}: {attr2} does not equal initial "
                            f"value of: {attr1}"
                        )
    except Exception as e:
        raise e


@all_ds
def test_inspection_json(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    with TemporaryDirectory() as tempdir:
        inspections = ds.inspect(load_cached=True, force=False)
        outdir = Path(tempdir)
        outfile = outdir / "inspections.json"
        inspections.to_json(outfile)
        loaded = InspectionResults.from_json(outfile)
        assert isinstance(loaded, InspectionResults)
        for attr in inspections.__dict__:
            attr1 = getattr(inspections, attr)
            attr2 = getattr(loaded, attr)
            if not are_equal(attr1, attr2):
                raise ValueError(
                    f"Saved attribute {attr}: {attr2} does not equal initial value of: {attr1}"
                )


@all_ds
def test_embed_json(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    tempdir = TemporaryDirectory()
    selected = EmbedSelected.random(ds)
    outdir = Path(tempdir.name)
    outfile = outdir / "embed.json"
    outfile.write_text(selected.to_json())
    loaded = EmbedSelected.from_json(outfile)
    assert isinstance(loaded, EmbedSelected)
    for attr in selected.__dict__:
        attr1 = getattr(selected, attr)
        attr2 = getattr(loaded, attr)
        if not are_equal(attr1, attr2):
            raise ValueError(
                f"Saved attribute {attr}: {attr2} does not equal initial value of: {attr1}"
            )


@all_ds
def test_wrap_json(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    tempdir = TemporaryDirectory()
    selected = WrapperSelected.random(ds)
    outdir = Path(tempdir.name)
    outfile = outdir / "wrap.json"
    outfile.write_text(selected.to_json())
    loaded = WrapperSelected.from_json(outfile)
    assert isinstance(loaded, WrapperSelected)
    for attr in selected.__dict__:
        attr1 = getattr(selected, attr)
        attr2 = getattr(loaded, attr)
        if not are_equal(attr1, attr2):
            raise ValueError(
                f"Saved attribute {attr}: {attr2} does not equal initial value of: {attr1}"
            )


@all_ds
def test_eval_save(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname in ["dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc", "internet_usage"]:
        return  # defective target
    tempdir = TemporaryDirectory()
    outdir = Path(tempdir.name)
    try:
        for _ in range(10):
            options = ProgramOptions.random(ds, outdir=outdir)
            selected = EvaluationResults.random(ds, options)

            selected.save(root=outdir)
            loaded = EvaluationResults.load(root=outdir)

            assert isinstance(loaded, EvaluationResults)
            for attr in selected.__dict__:
                attr1 = getattr(selected, attr)
                attr2 = getattr(loaded, attr)
                if not are_equal(attr1, attr2):
                    raise ValueError(
                        f"Saved attribute {attr}: {attr2} does not equal initial value of: {attr1}"
                    )
    except Exception as e:
        raise ValueError(f"Got error saving eval results for data {dsname}:\n{e}")
    finally:
        tempdir.cleanup()
