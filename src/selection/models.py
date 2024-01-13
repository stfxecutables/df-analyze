from __future__ import annotations

import traceback
from dataclasses import dataclass
from random import choice
from typing import TYPE_CHECKING, Optional
from warnings import warn

if TYPE_CHECKING:
    from src.cli.cli import ProgramOptions
from src.preprocessing.prepare import PreparedData
from src.selection.embedded import EmbedSelected, embed_select_features
from src.selection.filter import FilterSelected
from src.selection.wrapper import WrapperSelected, wrap_select_features
from src.testing.datasets import TestDataset


@dataclass
class ModelSelected:
    embed_selected: Optional[EmbedSelected]
    wrap_selected: Optional[WrapperSelected]

    @staticmethod
    def random(ds: TestDataset) -> ModelSelected:
        embed_selected = choice([None, *[EmbedSelected.random(ds) for _ in range(4)]])
        wrap_selected = choice([None, *[WrapperSelected.random(ds) for _ in range(4)]])
        return ModelSelected(embed_selected=embed_selected, wrap_selected=wrap_selected)


def model_select_features(
    prep_train: PreparedData,
    filtered: FilterSelected,
    options: ProgramOptions,
) -> ModelSelected:
    embed_selected = None
    wrap_selected = None
    try:
        if options.embed_select is not None:
            embed_selected = embed_select_features(prep_train=prep_train, options=options)
    except Exception as e:
        warn(
            f"Got error when attempting embedded feature selection:\n{e}\n"
            f"{traceback.format_exc()}"
        )

    try:
        if options.wrapper_select is not None:
            wrap_selected = wrap_select_features(prep_train=prep_train, options=options)
    except Exception as e:
        warn(
            f"Got error when attempting wrapped-based feature selection:\n{e}\n"
            f"{traceback.format_exc()}"
        )
    return ModelSelected(embed_selected=embed_selected, wrap_selected=wrap_selected)
