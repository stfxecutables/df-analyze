from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pytest import CaptureFixture
from tqdm import tqdm
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import (
    XLMRobertaTokenizerFast,
)

from df_analyze.embedding.cli import EmbeddingModality, EmbeddingOptions
from df_analyze.embedding.datasets import (
    NLPDataset,
    VisionDataset,
    dataset_from_opts,
)
from df_analyze.embedding.download import (
    dl_models_from_opts,
    download_models,
    error_if_download_needed,
)
from df_analyze.embedding.embed import (
    get_embeddings,
    get_model,
    get_nlp_embeddings,
    get_vision_embeddings,
)
from df_analyze.embedding.testing import (
    NLPTestingDataset,
    VisionTestingDataset,
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


def test_vision_random(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        for _ in tqdm(range(10), desc="Creating random Vision data"):
            with TemporaryDirectory() as tempdir:
                VisionDataset.random(tempdir=tempdir)


@pytest.mark.fast
def test_main_ds_nlp_loading(capsys: CaptureFixture) -> None:
    test_dses = NLPTestingDataset.get_all()
    dses = [ds.to_embedding_dataset() for ds in test_dses]
    for ds in dses:
        if ds.name == "go_emotions":
            continue  # multilabel
        try:
            ds.load()
        except Exception as e:
            raise ValueError(f"Got error for ds: {ds.name} @ {ds.datapath}") from e


@pytest.mark.fast
def test_main_ds_vision_loading(capsys: CaptureFixture) -> None:
    test_dses = VisionTestingDataset.get_all_cls()
    dses = [ds.to_embedding_dataset() for ds in test_dses]
    for ds in dses:
        try:
            ds.load()
        except Exception as e:
            raise ValueError(f"Got error for ds: {ds.name} @ {ds.datapath}") from e


def test_nlp_embed(capsys: CaptureFixture) -> None:
    model, tokenizer = get_model(EmbeddingModality.NLP)
    assert isinstance(model, XLMRobertaModel)
    assert isinstance(tokenizer, XLMRobertaTokenizerFast)
    with capsys.disabled():
        for ds in tqdm(NLPTestingDataset.get_all(), desc="Embedding NLP data..."):
            if ds.name == "go_emotions":
                continue  # multilabel
            ds = ds.to_embedding_dataset()
            df = get_nlp_embeddings(
                ds=ds,
                tokenizer=tokenizer,
                model=model,
                batch_size=2,
                num_texts=4,
                load_limit=1000,
            )
            assert len(df) == 4
            assert df.shape[1] == 1024 + 1


def test_vision_embed(capsys: CaptureFixture) -> None:
    model, processor = get_model(EmbeddingModality.Vision)
    assert isinstance(model, SiglipModel)
    assert isinstance(processor, SiglipProcessor)
    with capsys.disabled():
        for ds in tqdm(VisionTestingDataset.get_all_cls(), desc="Embedding vision data"):
            # if ds.name != "nsfw_detect":
            #     continue
            if ds.name == "rare-species":
                continue  # text labels
            ds = ds.to_embedding_dataset()
            df = get_vision_embeddings(
                ds=ds,
                processor=processor,
                model=model,
                batch_size=2,
                num_imgs=4,
                load_limit=1000,
            )
            assert len(df) == 4
            assert df.shape[1] == 1152 + 1


def test_main_vision(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        dses = VisionTestingDataset.get_all_cls()
        dses = [ds.to_embedding_dataset() for ds in dses]
        with TemporaryDirectory() as tempdir:
            dses = dses + [VisionDataset.random(tempdir=tempdir) for _ in range(10)]
            for ds in tqdm(dses, desc="Processing vision datasets"):
                if ds.name == "Anime-dataset":  # working...
                    continue
                if ds.name == "rare-species":
                    continue  # string labels
                out = Path(tempdir) / f"{ds.name}_test_embed_out.parquet"
                opts = EmbeddingOptions(
                    datapath=ds.datapath,
                    modality=EmbeddingModality.Vision,
                    name=ds.name,
                    outpath=out,
                    limit_samples=8,
                    batch_size=2,
                )
                assert opts.outpath is not None
                error_if_download_needed(opts)
                dl_models_from_opts(opts)
                ds = dataset_from_opts(opts)
                model, processor = get_model(opts.modality)
                df = get_embeddings(
                    ds=ds,  # type: ignore
                    processor=processor,  # type: ignore
                    model=model,  # type: ignore
                    batch_size=opts.batch_size,
                    load_limit=opts.limit_samples,
                )

                # print(df)
                # print(opts)
                # df.to_parquet(opts.outpath)
                # print(f"Saved embeddings to {opts.outpath}")


def test_main_nlp(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        for ds in tqdm(NLPTestingDataset.get_all(), desc="Processing NLP datasets"):
            if ds.name == "go_emotions":
                continue  # multilabel
            ds = ds.to_embedding_dataset()
            with TemporaryDirectory() as tempdir:
                out = Path(tempdir) / f"{ds.name}_test_embed_out.parquet"
                opts = EmbeddingOptions(
                    datapath=ds.datapath,
                    modality=EmbeddingModality.NLP,
                    name=ds.name,
                    outpath=out,
                    limit_samples=8,
                    batch_size=2,
                )
                assert opts.outpath is not None
                error_if_download_needed(opts)
                dl_models_from_opts(opts)
                ds = dataset_from_opts(opts)
                model, processor = get_model(opts.modality)
                df = get_embeddings(
                    ds=ds,  # type: ignore
                    processor=processor,  # type: ignore
                    model=model,  # type: ignore
                    batch_size=opts.batch_size,
                    load_limit=opts.limit_samples,
                )

                # print(df)
                # print(opts)
                df.to_parquet(opts.outpath)
                # print(f"Saved embeddings to {opts.outpath}")


def test_download_models(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        download_models()


@pytest.mark.med
def test_vision_padding(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        vision_padding_check()


@pytest.mark.slow
def test_cluster_sanity_vision(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        cluster_vision_sanity_check(n_samples=16)


@pytest.mark.slow
def test_cluster_sanity_nlp(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        cluster_nlp_sanity_check(n_samples=16)


if __name__ == "__main__":
    download_models(force=True)
