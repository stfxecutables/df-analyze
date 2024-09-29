from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
import traceback
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import (
    Optional,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas import DataFrame, Series
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import (
    XLMRobertaTokenizerFast,
)

from df_analyze.embedding.dataset_files import (
    CLS_DATAFILES,
    REG_DATAFILES,
    VISION_CLS,
)
from df_analyze.embedding.datasets import NLPDataset, VisionDataset
from df_analyze.embedding.download import (
    INTFLOAT_MODEL_FILES,
    INTFLOAT_MULTILINGUAL_MODEL,
    INTFLOAT_MULTILINGUAL_TOKENIZER,
    SIGLIP_MODEL,
    SIGLIP_MODEL_FILES,
    SIGLIP_PREPROCESSOR,
    SIGLIP_PREPROCESSOR_FILES,
    load_nlp_intfloat_ml_model_offline,
    load_siglip_offline,
)
from df_analyze.embedding.loading import _load_datafile
from df_analyze.embedding.utils import batched, get_n_test_samples


def get_nlp_embeddings(
    ds: NLPTestingDataset,
    tokenizer: XLMRobertaTokenizerFast,
    model: XLMRobertaModel,
    batch_size: int = 32,
    max_texts: int = 1024,
) -> tuple[DataFrame, Series, DataFrame, float, float]:
    X, y = ds.X, ds.y

    strat = y if ds.is_cls else get_reg_stratify(y)
    ss = StratifiedShuffleSplit(n_splits=1, train_size=max_texts)
    ix_train = next(ss.split(y, strat))[0]  # type: ignore
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
                texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            mask = batch_dict["attention_mask"]
            assert isinstance(mask, Tensor)

            outputs = model(**batch_dict)
            embeddings = avg_pool(outputs.last_hidden_state, attention_mask=mask)
            # embeddings = F.normalize(embeddings, p=2, dim=1)  # shape [B, 1024]
            all_embeddings.append(embeddings)
        elapsed_s = perf_counter() - start
    samples_per_s = max_texts / elapsed_s

    start = perf_counter()
    embeddings = torch.cat(all_embeddings)
