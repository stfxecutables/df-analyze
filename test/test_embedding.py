from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path

from pytest import CaptureFixture

from src.df_analyze.embedding.download import download_models
from src.df_analyze.embedding.testing import (
    cluster_nlp_sanity_check,
    cluster_vision_sanity_check,
    vision_padding_check,
)

INTFLOAT_MULTILINGUAL_MODEL = ROOT / "downloaded_models/intfloat_multi_large/model"
INTFLOAT_MULTILINGUAL_TOKENIZER = (
    ROOT / "downloaded_models/intfloat_multi_large/tokenizer"
)
INTFLOAT_MULTILINGUAL_MODEL.mkdir(exist_ok=True, parents=True)
INTFLOAT_MULTILINGUAL_TOKENIZER.mkdir(exist_ok=True, parents=True)

SIGLIP_MODEL = ROOT / "downloaded_models/siglip_so400m_patch14_384/model"
SIGLIP_PREPROCESSOR = ROOT / "downloaded_models/siglip_so400m_patch14_384/preprocessor"

MACOS_NLP_RUNTIMES = ROOT / "nlp_embed_runtimes.parquet"
MACOS_VISION_RUNTIMES = ROOT / "vision_embed_runtimes.parquet"
NIAGARA_NLP_RUNTIMES = ROOT / "nlp_embed_runtimes_niagara.parquet"
NIAGARA_VISION_RUNTIMES = ROOT / "vision_embed_runtimes_niagara.parquet"


def test_download_models(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        download_models()


def test_vision_padding(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        vision_padding_check()


def test_cluster_sanity_vision(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        cluster_vision_sanity_check(n_samples=16)


def test_cluster_sanity_nlp(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        cluster_nlp_sanity_check(n_samples=16)


if __name__ == "__main__":
    download_models(force=True)
