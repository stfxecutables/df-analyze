from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
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
from pytest import CaptureFixture

from df_analyze.embedding.datasets import download_models
from df_analyze.embedding.dataset_files import (
    CLS_DATAFILES,
    NLP_CLS,
    NLP_REG,
    NLP_ROOT,
    REG_DATAFILES,
    VISION_CLS,
    VISION_REG,
)
from df_analyze.embedding.utils import batched

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
