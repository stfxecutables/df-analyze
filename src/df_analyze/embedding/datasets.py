from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import traceback
from io import BytesIO
from PIL import Image
import torch
from time import perf_counter
from itertools import islice
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import (
    XLMRobertaTokenizerFast,
)
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from transformers.models.siglip.image_processing_siglip import SiglipImageProcessor
from transformers.feature_extraction_utils import BatchFeature

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import KBinsDiscretizer
import json
import os
import sys
from functools import cache
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from sklearn.model_selection import StratifiedShuffleSplit
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from typing_extensions import Literal

from df_analyze.embedding.dataset_files import (
    CLS_DATAFILES,
    NLP_CLS,
    NLP_REG,
    NLP_ROOT,
    REG_DATAFILES,
    VISION_CLS,
    VISION_REG,
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


def batched(iterable, n):
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := list(tuple(islice(iterator, n))):
        yield batch


def load_json_lines(path: Path) -> DataFrame:
    text = path.read_text()
    lines = [(s.strip() + "}").replace("}}", "}") for s in text.split("}\n")]
    objs = []
    for i, line in enumerate(lines):
        if len(line) <= 1:  # handles last line, blanks
            continue
        try:
            obj = json.loads(line)
            objs.append(obj)
        except json.decoder.JSONDecodeError as e:
            ix_prv = max(0, i - 1)
            ix_nxt = min(i + 1, len(lines) - 1)
            ix_cur = i
            prv = lines[ix_prv]
            nxt = lines[ix_nxt]
            raise ValueError(
                f"Got error parsing line {i}: `{line}` of file: {path}.\n"
                f"[{ix_prv:d}] Previous line: {prv}\n"
                f"[{ix_cur:d}] Current line:  {line}\n"
                f"[{ix_nxt:d}] Next line:     {nxt}\n"
            ) from e
    df = DataFrame(objs).infer_objects().convert_dtypes()
    return df


def _load_datafile(path: Path | None) -> DataFrame | None:
    if path is None:
        return None

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".jsonl":  # json-list, it seems, each line is an object
        return load_json_lines(path)

    try:
        if path.suffix == ".json":
            text = path.read_text()
            info = json.loads(text)
            return DataFrame(info)
    except json.decoder.JSONDecodeError as e:
        return load_json_lines(path)

    raise ValueError(f"Unrecognized filetype: `{path.suffix}` from data file: {path}")


class VisionDataset:
    def __init__(
        self,
        name: str,
        root: Path,
        is_cls: bool,
    ) -> None:
        self.name = name
        self.root = root
        self.is_cls = self.is_classification = is_cls
        self._df = None

    def __str__(self) -> str:
        cls = f"{self.__class__.__name__}"
        return f"{cls}('{self.name}')"

    __repr__ = __str__

    @property
    def X(self) -> Series:
        df = self.load_raw()
        return df["image"]

    @property
    def y(self) -> Series:
        df = self.load_raw()
        if "Places_in_Japan" in self.root.name:
            raise RuntimeError(f"This dataset: {self.name}@{self.root} has no labels.")
        return df["label"]

    def load_raw(self) -> DataFrame:
        # https://github.com/python-pillow/Pillow/issues/4987#issuecomment-710994934
        #
        # "
        # To protect against potential DOS attacks caused by “decompression
        # bombs” (i.e. malicious files which decompress into a huge amount of
        # data and are designed to crash or cause disruption by using up a lot of
        # memory), Pillow will issue a DecompressionBombWarning if the number of
        # pixels in an image is over a certain limit, PIL.Image.MAX_IMAGE_PIXELS.
        #
        # This threshold can be changed by setting PIL.Image.MAX_IMAGE_PIXELS. It
        # can be disabled by setting Image.MAX_IMAGE_PIXELS = None.
        #
        # If desired, the warning can be turned into an error with
        # warnings.simplefilter('error', Image.DecompressionBombWarning) or
        # suppressed entirely with warnings.simplefilter('ignore',
        # Image.DecompressionBombWarning). See also the logging documentation to
        # have warnings output to the logging facility instead of stderr.
        #
        # If the number of pixels is greater than twice
        # PIL.Image.MAX_IMAGE_PIXELS, then a DecompressionBombError will be
        # raised instead. So:
        #
        #   from PIL import Image
        #   Image.MAX_IMAGE_PIXELS = None   # disables the warning
        #   Image.open(...)   # whatever operation you now run should work
        # "

        Image.MAX_IMAGE_PIXELS = None  # disables the warning

        if self._df is not None:
            return self._df
        pq = self.root / "all.parquet"
        # PyArrow engine needed to properly decode bytes column
        df = pd.read_parquet(pq, engine="pyarrow")
        im = df["image"].apply(lambda b: Image.open(BytesIO(b)))
        if self.name == "Handwritten-Mathematical-Expression-Convert-LaTeX":
            # TODO: these images have only two dimensions since BW, so expand
            ...
        self._df = pd.concat([df["label"], im], axis=1) if "label" in df.columns else im
        return self._df

    @classmethod
    def get_all_cls(cls) -> list[VisionDataset]:
        datas = []
        roots = sorted(VISION_CLS.glob("*"))
        for root in roots:
            data = cls(name=root.name, root=root, is_cls=True)
            datas.append(data)
        return datas


class NLPDataset:
    def __init__(
        self,
        name: str,
        root: Path,
        datafiles: dict[str, Path | list[str] | None],
        is_cls: bool,
    ) -> None:
        self.name = name
        self.root = root
        self.is_cls = self.is_classification = is_cls
        self.datafiles: dict[str, Path | list[str] | None] = deepcopy(datafiles)
        self.datafiles.pop("root")
        self.labels = self.datafiles.pop("labels", None)
        for subset, path in self.datafiles.items():  # make paths rel to root
            if path is not None and isinstance(path, Path):
                self.datafiles[subset] = self.root / path
        targets = self.datafiles.pop("targetcols")
        assert isinstance(targets, list)
        if len(targets) < 1:
            raise ValueError(f"Must specify targets for data: {name} at {root}")
        self.targets: list[str] = targets
        self.textcol = cast(str, self.datafiles.pop("textcol"))
        self.dropcols = cast(list[str], self.datafiles.pop("dropcols"))
        self.namecols = cast(list[str], self.datafiles.pop("labelnamecols"))
        self._df = None

    @property
    def X(self) -> Series:
        df = self.load_raw()
        return df[self.textcol]

    @property
    def y(self) -> Series:
        df = self.load_raw()
        return df[self.targets[0]]

    def load_raw(self, ignore_decompression_warning: bool = True) -> DataFrame:
        # https://github.com/python-pillow/Pillow/issues/4987#issuecomment-710994934
        #
        # "
        # To protect against potential DOS attacks caused by “decompression
        # bombs” (i.e. malicious files which decompress into a huge amount of
        # data and are designed to crash or cause disruption by using up a lot of
        # memory), Pillow will issue a DecompressionBombWarning if the number of
        # pixels in an image is over a certain limit, PIL.Image.MAX_IMAGE_PIXELS.
        #
        # This threshold can be changed by setting PIL.Image.MAX_IMAGE_PIXELS. It
        # can be disabled by setting Image.MAX_IMAGE_PIXELS = None.
        #
        # If desired, the warning can be turned into an error with
        # warnings.simplefilter('error', Image.DecompressionBombWarning) or
        # suppressed entirely with warnings.simplefilter('ignore',
        # Image.DecompressionBombWarning). See also the logging documentation to
        # have warnings output to the logging facility instead of stderr.
        #
        # If the number of pixels is greater than twice
        # PIL.Image.MAX_IMAGE_PIXELS, then a DecompressionBombError will be
        # raised instead. So:
        #
        #   from PIL import Image
        #   Image.MAX_IMAGE_PIXELS = None   # disables the warning
        #   Image.open(...)   # whatever operation you now run should work
        # "
        if ignore_decompression_warning:
            Image.MAX_IMAGE_PIXELS = None  # disables the warning

        if self._df is not None:
            return self._df

        if self.datafiles["all"] is not None:  # just load this file instead
            df = _load_datafile(self.datafiles["all"])
            if df is None:
                raise ValueError("Impossible!")
            self._df = df
            return df

        # If a simple "all" file is not provided, vertically concat all others
        datafiles = deepcopy(self.datafiles)
        datafiles.pop("all")
        dfs = [_load_datafile(file) for file in datafiles.values()]
        # TODO: check for duplicates
        df = pd.concat(dfs, axis=0, ignore_index=True)
        self._df = df
        return df

    def load(self) -> DataFrame:
        raw = self.load_raw()
        if len(self.targets) == 1:
            targetcol = self.targets[0]
        else:  # for now, just use first target
            targetcol = self.targets[0]

        textcol = self.textcol
        return raw.loc[:, [textcol, targetcol]].copy()

    @staticmethod
    def get_all_cls() -> list[NLPDataset]:
        datas = []
        for dsname, info in tqdm(
            CLS_DATAFILES.items(), total=len(CLS_DATAFILES), disable=True
        ):
            data = NLPDataset(name=dsname, root=info["root"], datafiles=info, is_cls=True)
            datas.append(data)
        return datas

    @staticmethod
    def get_all_reg() -> list[NLPDataset]:
        datas = []
        for dsname, info in tqdm(
            REG_DATAFILES.items(), total=len(REG_DATAFILES), disable=True
        ):
            data = NLPDataset(
                name=dsname, root=info["root"], datafiles=info, is_cls=False
            )
            datas.append(data)
        return datas

    @staticmethod
    def get_all() -> list[NLPDataset]:
        return NLPDataset.get_all_cls() + NLPDataset.get_all_reg()


def test_load_all_and_preview_sizes() -> None:
    dses = NLPDataset.get_all()
    datasets = []
    all_stats = []
    for ds in tqdm(dses):
        df = ds.load()
        dsname = ds.name
        try:
            lengths = ds.X.str.split(" ").apply(lambda s: len(s))
            stats = (
                lengths.astype(float)
                .describe(percentiles=[0.1, 0.25, 0.9, 0.95, 0.99])
                .to_frame()
                .T.rename(columns={"50%": "med"})
                .loc[:, ["min", "mean", "med", "90%", "95%", "99%", "max"]]
            )
            stats.insert(0, "ds", dsname)
            all_stats.append(stats)
            print(stats)
        except TypeError as e:
            print(e)

        print("=" * 81)
        print(f"{dsname}: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head())
        datasets.append((dsname, df))
    datasets = sorted(datasets, key=lambda pair: len(pair[1]))
    stats = pd.concat(all_stats, axis=0, ignore_index=True)
    for dsname, df in datasets:
        print(f"{dsname}: {df.shape}")
    pd.options.display.max_rows = 100
    print(stats)


def avg_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_nlp_embeddings(
    ds: NLPDataset,
    tokenizer: XLMRobertaTokenizerFast,
    model: XLMRobertaModel,
    batch_size: int = 32,
    max_texts: int = 1024,
) -> tuple[DataFrame, Series, DataFrame, float, float]:
    X, y = ds.X, ds.y

    strat = y if ds.is_cls else get_reg_stratify(y)
    ss = StratifiedShuffleSplit(n_splits=1, train_size=max_texts)
    ix_train = next(ss.split(y, strat))[0]
    X_tr, y_tr = X.iloc[ix_train], y.iloc[ix_train]
    strat = strat.iloc[ix_train]

    # Each input text should start with "query: " or "passage: ", even for
    # non-English texts. For tasks other than retrieval, you can simply use
    # the "query: " prefix. See also the discussion for this model on HF, but
    # the TL;DR is basically that it doesn't matter too much which you prepend,
    # but empirically, using "query: " seems to give better performance overall
    all_texts = X_tr.apply(lambda text: f"query: {text}").tolist()
    lengths = Series([len(s) for s in all_texts])
    stats = (
        lengths.astype(float)
        .describe(percentiles=[0.9])
        .to_frame()
        .T.rename(columns={"50%": "med"})
        .loc[:, ["min", "mean", "med", "90%", "max"]]
    )

    # Tokenize the input texts
    # batch_dict is:
    # {
    #     input_ids: Tensor[B, N],
    #     attention_mask: Tensor[B, N],
    # }
    #
    # B = batch size, i.e. number of samples
    # N = max number of tokens in longest sequence of tokens in batch
    #
    # input_ids has the tokens as integers (ones seem to be padding), i.e. is
    # the tokenization of the input. This of course will only be loosely
    # correlated with word count or etc.
    #
    # attention_mask is a boolean where mask[i, :] is such that for input i,
    # mask[i] is all ones up to number of non-masking tokens in input_ids
    all_embeddings = []
    with torch.no_grad():
        text_batches = [*batched(all_texts, batch_size)]
        start = perf_counter()
        for texts in tqdm(
            text_batches, total=len(text_batches), desc="Embedding batches"
        ):
            batch_dict = tokenizer(
                texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
            )
            mask = batch_dict["attention_mask"]
            assert isinstance(mask, Tensor)

            outputs = model(**batch_dict)
            embeddings = avg_pool(outputs.last_hidden_state, attention_mask=mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)  # shape [B, 1024]
            all_embeddings.append(embeddings)
        elapsed_s = perf_counter() - start
    samples_per_s = max_texts / elapsed_s

    start = perf_counter()
    embeddings = torch.cat(all_embeddings)
    cols = [f"embed{i+1:4d}" for i in range(embeddings.shape[1])]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols)
    df_text = X_tr.to_frame().rename(columns=lambda s: "text")
    df_targ = y_tr.to_frame().rename(columns=lambda s: "target")
    df_embed.index = df_text.index.copy()
    df = pd.concat([df_text, df_targ, df_embed], axis=1, ignore_index=False)
    elapsed_s = perf_counter() - start
    postproc_samples_per_s = max_texts / elapsed_s
    return df, strat, stats, samples_per_s, postproc_samples_per_s


def get_vision_embeddings(
    ds: VisionDataset,
    processor: SiglipProcessor,
    model: SiglipModel,
    batch_size: int = 32,
    max_imgs: int = 1024,
) -> tuple[DataFrame, Series, DataFrame, float, float]:
    X, y = ds.X, ds.y

    strat = y if ds.is_cls else get_reg_stratify(y)
    ss = StratifiedShuffleSplit(n_splits=1, train_size=max_imgs)
    ix_train = next(ss.split(y, strat))[0]
    X_tr, y_tr = X.iloc[ix_train], y.iloc[ix_train]
    strat = strat.iloc[ix_train]

    all_imgs = X_tr.tolist()
    s1 = Series([s.size[0] for s in all_imgs], name="h")
    s2 = Series([s.size[1] for s in all_imgs], name="w")
    shapes = pd.concat([s1, s2], axis=1)
    stats = (
        shapes.astype(float)
        .describe(percentiles=[0.9])
        .T.rename(columns={"50%": "med"})
        .loc[:, ["min", "mean", "med", "90%", "max"]]
    )

    all_embeddings = []
    with torch.no_grad():
        img_batches = [*batched(all_imgs, batch_size)]
        start = perf_counter()
        for img in tqdm(img_batches, total=len(img_batches), desc="Embedding batches"):
            processed: BatchFeature = processor(
                text=None, images=img, return_tensors="pt"
            )
            outputs = model.vision_model(
                **processed, output_hidden_states=False, output_attentions=False
            )
            embeddings = outputs.pooler_output

            # embeddings = avg_pool(embeddings2)
            # embeddings = F.normalize(embeddings, p=2, dim=1)  # shape [B, 1024]
            all_embeddings.append(embeddings)
        elapsed_s = perf_counter() - start
    samples_per_s = max_imgs / elapsed_s

    start = perf_counter()
    embeddings = torch.cat(all_embeddings)
    cols = [f"embed{i+1:4d}" for i in range(embeddings.shape[1])]
    df_embed = DataFrame(data=embeddings.numpy(), columns=cols)
    df_img = X_tr.to_frame().rename(columns=lambda s: "img")
    df_targ = y_tr.to_frame().rename(columns=lambda s: "target")
    df_embed.index = df_img.index.copy()
    df = pd.concat([df_img, df_targ, df_embed], axis=1, ignore_index=False)
    elapsed_s = perf_counter() - start
    postproc_samples_per_s = max_imgs / elapsed_s
    return df, strat, stats, samples_per_s, postproc_samples_per_s


def download_nlp_intfloat_ml_model() -> None:
    model = cast(
        XLMRobertaModel,
        AutoModel.from_pretrained("intfloat/multilingual-e5-large"),
    )
    model.save_pretrained(INTFLOAT_MULTILINGUAL_MODEL)
    print(f"Saved model to {INTFLOAT_MULTILINGUAL_MODEL}")

    tokenizer = cast(
        XLMRobertaTokenizerFast,
        AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large"),
    )
    tokenizer.save_pretrained(INTFLOAT_MULTILINGUAL_TOKENIZER)
    print(f"Saved tokenizer to {INTFLOAT_MULTILINGUAL_MODEL}")


def load_nlp_intfloat_ml_model_offline() -> (
    tuple[XLMRobertaModel, XLMRobertaTokenizerFast]
):
    model = cast(
        XLMRobertaModel,
        AutoModel.from_pretrained(INTFLOAT_MULTILINGUAL_MODEL, local_files_only=True),
    )
    tokenizer = cast(
        XLMRobertaTokenizerFast,
        AutoTokenizer.from_pretrained(
            INTFLOAT_MULTILINGUAL_TOKENIZER, local_files_only=True
        ),
    )
    return model, tokenizer


def download_siglip_model() -> None:
    model = cast(
        SiglipModel, AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
    )
    model.save_pretrained(SIGLIP_MODEL)
    print(f"Saved model to {SIGLIP_MODEL}")

    processor = cast(
        SiglipProcessor, AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    )
    processor.save_pretrained(SIGLIP_PREPROCESSOR)
    print(f"Saved preprocessor to {SIGLIP_PREPROCESSOR}")


def load_siglip_offline() -> tuple[SiglipModel, SiglipProcessor]:
    model = cast(
        SiglipModel,
        AutoModel.from_pretrained(SIGLIP_MODEL, local_files_only=True),
    )
    tokenizer = cast(
        SiglipProcessor,
        AutoProcessor.from_pretrained(SIGLIP_PREPROCESSOR, local_files_only=True),
    )
    return model, tokenizer


def estimate_nlp_embedding_times() -> None:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    N = 256 if ON_CLUSTER else 32
    BATCHES = [8, 16, 40] if ON_CLUSTER else [2, 4, 8]
    OUT = NIAGARA_NLP_RUNTIMES if ON_CLUSTER else MACOS_NLP_RUNTIMES
    # on Macbook, batch=2 seems fastest (4 very close, 1 by far too slow)

    model, tokenizer = load_nlp_intfloat_ml_model_offline()
    dses = NLPDataset.get_all_cls()
    results = []

    for ds in dses:
        print("=" * 81)
        print(ds.name)
        N_samples = len(ds.y)
        for B in BATCHES:
            try:
                start = perf_counter()
                embeds, strat, stats, proc_speed, postproc_speed = get_nlp_embeddings(
                    ds=ds, tokenizer=tokenizer, model=model, batch_size=B, max_texts=N
                )
                total = round(perf_counter() - start, 3)
                proc_speed = round(proc_speed, 3)
                embed_time = round(N / proc_speed, 1)
                post = f"{round(postproc_speed / 1000, 0)}K"
            except Exception as e:
                print(e)
                continue

            print(
                f"At batch = {B}, total time = {total}; embed time = {embed_time}; embedding speed = {proc_speed} samples/s; postproc speed = {post} samples/s"
            )
            row = DataFrame(
                {
                    "ds": ds.name,
                    "n_samp": N_samples,
                    "batch": B,
                    "total_s": total,
                    "est_h": ((total / N) * N_samples) / 3600,
                    "embed_s": embed_time,
                    "embed_speed": proc_speed,
                },
                index=[0],
            )
            row = pd.concat([row, stats], axis=1, ignore_index=False)
            results.append(row)
            print(row)

    runtimes = pd.concat(results, axis=0, ignore_index=True)
    pd.options.display.max_rows = 500
    runtimes.to_parquet(OUT)
    print(runtimes)
    print(
        runtimes.groupby(["ds", "n_samp"])
        .apply(lambda grp: grp.nlargest(1, "total_s"), include_groups=False)
        .droplevel(2)
        .sort_values(by="embed_speed")
    )
    print(f"Saved runtimes to {OUT}")


def get_optimal_nlp_batch(ds: NLPDataset) -> int:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    OUT = NIAGARA_NLP_RUNTIMES if ON_CLUSTER else MACOS_NLP_RUNTIMES
    DEFAULT = 8 if ON_CLUSTER else 2
    runtimes = pd.read_parquet(OUT)
    runtimes = (
        runtimes.groupby(["ds", "n_samp"])
        .apply(lambda grp: grp.nsmallest(1, "total_s"), include_groups=False)
        .droplevel(2)
        .reset_index()
    )
    if ds.name not in runtimes["ds"].values:
        return DEFAULT

    return runtimes[runtimes["ds"] == ds.name]["batch"].item()


def get_optimal_vision_batch(ds: VisionDataset) -> int:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    OUT = NIAGARA_VISION_RUNTIMES if ON_CLUSTER else MACOS_VISION_RUNTIMES
    DEFAULT = 8 if ON_CLUSTER else 2
    runtimes = pd.read_parquet(OUT)
    runtimes = (
        runtimes.groupby(["ds", "n_samp"])
        .apply(lambda grp: grp.nsmallest(1, "total_s"), include_groups=False)
        .droplevel(2)
        .reset_index()
    )
    if ds.name not in runtimes["ds"].values:
        return DEFAULT

    return runtimes[runtimes["ds"] == ds.name]["batch"].item()


def estimate_vision_embedding_times() -> None:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    N = 256 if ON_CLUSTER else 32
    BATCHES = [8, 16, 40] if ON_CLUSTER else [2, 4, 8]
    OUT = NIAGARA_VISION_RUNTIMES if ON_CLUSTER else MACOS_VISION_RUNTIMES
    # on Macbook, batch=2 seems fastest (4 very close, 1 by far too slow)

    model, processor = load_siglip_offline()
    dses = VisionDataset.get_all_cls()

    results = []

    for ds in dses:
        if "Places_in_Japan" in ds.name:  # no targets to stratify on
            continue
        if "rare-species" != ds.name:
            continue
        print("=" * 81)
        print(ds.name)
        N_samples = len(ds.y)
        for B in BATCHES:
            try:
                start = perf_counter()
                embeds, strat, stats, proc_speed, postproc_speed = get_vision_embeddings(
                    ds=ds, processor=processor, model=model, batch_size=B, max_imgs=N
                )
                total = round(perf_counter() - start, 3)
                proc_speed = round(proc_speed, 3)
                embed_time = round(N / proc_speed, 1)
                post = f"{round(postproc_speed / 1000, 0)}K"
            except Exception as e:
                traceback.print_exc()
                print(e)
                continue

            print(
                f"At batch = {B}, total time = {total}; embed time = {embed_time}; embedding speed = {proc_speed} samples/s; postproc speed = {post} samples/s"
            )
            wide_stats = pd.concat(
                [
                    stats.loc["h"]
                    .to_frame()
                    .T.rename(columns=lambda col: f"{col}_h")
                    .reset_index(drop=True),
                    stats.loc["w"]
                    .to_frame()
                    .T.rename(columns=lambda col: f"{col}_w")
                    .reset_index(drop=True),
                ],
                axis=1,
            )
            wide_stats = wide_stats.loc[
                :,
                [
                    "min_h",
                    "min_w",
                    "mean_h",
                    "mean_w",
                    "med_h",
                    "med_w",
                    "90%_h",
                    "90%_w",
                    "max_h",
                    "max_w",
                ],
            ]
            row = DataFrame(
                {
                    "ds": ds.name,
                    "n_samp": N_samples,
                    "batch": B,
                    "total_s": total,
                    "est_h": ((total / N) * N_samples) / 3600,
                    "embed_s": embed_time,
                    "embed_speed": proc_speed,
                },
                index=[0],
            )
            row = pd.concat([row, wide_stats], axis=1, ignore_index=False)
            results.append(row)
            print(row)

    runtimes = pd.concat(results, axis=0, ignore_index=True)
    pd.options.display.max_rows = 500
    runtimes.to_parquet(OUT)
    print(runtimes)
    print(
        runtimes.groupby(["ds", "n_samp"])
        .apply(lambda grp: grp.nlargest(1, "total_s"), include_groups=False)
        .droplevel(2)
        .sort_values(by="embed_speed")
    )
    print(f"Saved runtimes to {OUT}")


def get_reg_stratify(y: Series) -> Series:
    yy = y.to_numpy().reshape(-1, 1)
    kb = KBinsDiscretizer(n_bins=5, encode="ordinal")
    strat = kb.fit_transform(yy)
    strat = strat.ravel()
    strat = Series(name=y.name, data=strat)
    return strat


def cluster_nlp_sanity_check() -> None:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    N = 256 if ON_CLUSTER else 128
    BATCHES = [8, 16, 40] if ON_CLUSTER else [2, 4, 8]
    # on Macbook, batch=2 seems fastest (4 very close, 1 by far too slow)
    OUT = NIAGARA_NLP_RUNTIMES if ON_CLUSTER else MACOS_NLP_RUNTIMES

    model, tokenizer = load_nlp_intfloat_ml_model_offline()
    dses = NLPDataset.get_all_cls()
    results = []

    for ds in dses:
        print("=" * 81)
        print(ds.name)
        batch = get_optimal_nlp_batch(ds)
        try:
            embeds, strat = get_nlp_embeddings(
                ds=ds, tokenizer=tokenizer, model=model, batch_size=batch, max_texts=N
            )[:2]
            X = embeds.drop(columns=["text", "target"])
            dfs = []
            for metricname, metric in dict(
                cosine=cosine_similarity, euclid=lambda x: 1 - euclidean_distances(x)
            ).items():
                sims = metric(X.values)
                sims = DataFrame(data=sims, index=X.index, columns=X.index)
                for clust in sorted(strat.unique()):
                    ix = strat == clust
                    df = sims.loc[ix, ix]
                    df = (
                        df.where(np.triu(np.ones(df.shape), k=1).astype(np.bool))
                        .stack()
                        .reset_index()
                    )
                    df.columns = ["x1", "x2", "sim"]
                    within = df["sim"].mean()
                    between = sims.loc[ix, ~ix].values.mean()
                    row = DataFrame(
                        {
                            "cls": clust,
                            "within": within,
                            "between": between,
                            "OK": within > between,
                            "metric": metricname,
                        },
                        index=[0],
                    )
                    dfs.append(row)
            df = pd.concat(dfs, axis=0, ignore_index=True)
            df["ds"] = ds.name
            results.append(df)
            print(df)

        except Exception as e:
            print(e)
            continue


def cluster_vision_sanity_check() -> None:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    N = 256 if ON_CLUSTER else 16

    model, processor = load_siglip_offline()
    dses = VisionDataset.get_all_cls()
    results = []

    for ds in dses:
        print("=" * 81)
        print(ds.name)
        batch = get_optimal_vision_batch(ds)
        try:
            embeds, strat = get_vision_embeddings(
                ds=ds, processor=processor, model=model, batch_size=batch, max_imgs=N
            )[:2]
            X = embeds.drop(columns=["img", "target"])
            dfs = []
            for metricname, metric in dict(
                cosine=cosine_similarity, euclid=lambda x: euclidean_distances(x)
            ).items():
                sims = metric(X.values)
                if metricname == "euclid":
                    sims = 1 - sims / sims.max()
                sims = DataFrame(data=sims, index=X.index, columns=X.index)
                for clust in sorted(strat.unique()):
                    ix = strat == clust
                    df = sims.loc[ix, ix]
                    df = (
                        df.where(np.triu(np.ones(df.shape), k=1).astype(np.bool))
                        .stack()
                        .reset_index()
                    )
                    df.columns = ["x1", "x2", "sim"]
                    within = df["sim"].mean()
                    between = sims.loc[ix, ~ix].values.mean()
                    row = DataFrame(
                        {
                            "cls": clust,
                            "within": within,
                            "between": between,
                            "OK": within > between,
                            "metric": metricname,
                        },
                        index=[0],
                    )
                    dfs.append(row)
            df = pd.concat(dfs, axis=0, ignore_index=True)
            df["ds"] = ds.name
            results.append(df)
            print(df)

        except Exception as e:
            print(e)
            continue


def detect_sus_images(ds: VisionDataset) -> None:
    """
    Surprisingly, not about image onctent, but size, since some multi-panel
    animated comics are included in the deepghs/nsfw_detect, and these images
    are large enough to trigger PIL DecompressionBombWarning:

    > `DecompressionBombWarning: Image size (115600000 pixels) exceeds limit
    > of 89478485 pixels, could be decompression bomb DOS attack.`


    """
    # https://github.com/python-pillow/Pillow/issues/4987#issuecomment-710994934
    #
    # "
    # To protect against potential DOS attacks caused by “decompression
    # bombs” (i.e. malicious files which decompress into a huge amount of
    # data and are designed to crash or cause disruption by using up a lot of
    # memory), Pillow will issue a DecompressionBombWarning if the number of
    # pixels in an image is over a certain limit, PIL.Image.MAX_IMAGE_PIXELS.
    #
    # This threshold can be changed by setting PIL.Image.MAX_IMAGE_PIXELS. It
    # can be disabled by setting Image.MAX_IMAGE_PIXELS = None.
    #
    # If desired, the warning can be turned into an error with
    # warnings.simplefilter('error', Image.DecompressionBombWarning) or
    # suppressed entirely with warnings.simplefilter('ignore',
    # Image.DecompressionBombWarning). See also the logging documentation to
    # have warnings output to the logging facility instead of stderr.
    #
    # If the number of pixels is greater than twice
    # PIL.Image.MAX_IMAGE_PIXELS, then a DecompressionBombError will be
    # raised instead. So:
    #
    #   from PIL import Image
    #   Image.MAX_IMAGE_PIXELS = None   # disables the warning
    #   Image.open(...)   # whatever operation you now run should work
    # "

    Image.MAX_IMAGE_PIXELS = None  # disables the warning
    df = ds.load_raw()
    shapes = df["image"].apply(lambda i: i.size)
    shapes = pd.concat(
        [shapes.apply(lambda s: s[0]), shapes.apply(lambda s: s[1])],
        axis=1,
        keys=["x", "y"],
    ).max(axis=1)
    shapes.index = df.index
    ix_large = (-shapes).argsort()
    large = df.iloc[ix_large[:20]]
    for img in large["image"]:
        plt.imshow(img)
        plt.show()
    print(shapes)
    raise


if __name__ == "__main__":
    # test_load_all_and_preview_sizes()
    # sys.exit()

    # download_nlp_intfloat_ml_model()
    # sys.exit()
    # load_nlp_intfloat_ml_model_offline()
    # estimate_embedding_times()
    # cluster_nlp_sanity_check()

    # model, tokenizer = load_nlp_intfloat_ml_model_offline()
    # download_siglip_model()

    # SUS ds is nsfw_detect
    # model, processor = load_siglip_offline()
    # dses = VisionDataset.get_all_cls()
    estimate_vision_embedding_times()
    # cluster_vision_sanity_check()
