from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from df_analyze.embedding.cli import EmbeddingOptions, make_parser
from df_analyze.embedding.datasets import EmbeddingDataset
from df_analyze.embedding.download import (
    dl_models_from_opts,
    error_if_download_needed,
)


def main() -> None:
    """Do embedding logic here"""
    parser = make_parser()
    opts = EmbeddingOptions.from_parser(parser)
    error_if_download_needed(opts)
    dl_models_from_opts(opts)
    ds

    print(opts)
