from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from argparse import ArgumentParser, RawTextHelpFormatter
from enum import Enum
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Optional, Type, Union

from df_analyze.cli.parsing import (
    resolved_path,
)

if TYPE_CHECKING:
    pass
from df_analyze.utils import Debug

USAGE_STRING = """

The df-embed script can be used to embed image (vision) data, or texts (NLP)
which are not too long.

IMPORTANT: In order to get this working, you will first have to download the
underlying HuggingFace zero-shot models. Assuming you have used `cd` to
change to where you have cloned this repo, then this can be done by running:

    python df-embed.py --download

*once*. This will download the necessary files to ./downloaded_models. Then,
all subsequent calls will work properly, even offline without internet
access.

Good defaults are chosen so that likely the only arguments you would wish to
specify manually are:

    --data (required)
    --modality (required)

"""

USAGE_EXAMPLES = """
USAGE EXAMPLE (assumes you have run `python df-embed --download`
successfully):

    python df-embed.py --data my_data.parquet --modality nlp

"""

DATA_HELP = """
Path to the input .parquet data file to be converted to tabular format via
zero-shot embeddings.

# IMAGE DATA

  For image classification data (`--modality vision`), the file must be a
  two-column table with the columns named "image" and "label". The order of
  the columns is not important, but the "label" column must contain integers
  in {0, 1, ..., c - 1}, where `c` is the number of class labels for your
  data. The data type is not really important, however, if the table is
  loaded into a Pandas DataFrame `df`, then running `df["label"].astype(np.int64)`
  (assuming you have imported NumPy as `np`, as is convention) should not
  alter the meaning of the data.

  For image regression data (very rare), the file must be a two-column table
  with the columns named "image" and "target". The order of the columns is
  not important, but the "target" column must contain floating point values.
  The floating point data type is not really important, however, if the table
  is loaded into a Pandas DataFrame `df`, then running
  `df["label"].astype(float)` should not raise any exceptions.

  The "image" column must of `bytes` dtype, and must be readable by PIL
  `Image.open`. Internally, all we do, again assuming that the data is loaded
  into a Pandas DataFrame `df`, is run:

      images = df["image"].apply(lambda raw: Image.open(BytesIO(raw)))

  TODO: Figure out channels first vs. last and what happens with BW images.


# NLP Text Data

  For text classification data (`--modality nlp`), the file must be a
  two-column table with the columns named "text" and "label". The order of
  the columns is not important, but the "label" column must contain integers
  in {0, 1, ..., c - 1}, where `c` is the number of class labels for your
  data. The data type is not really important, however, if the table is
  loaded into a Pandas DataFrame `df`, then running `df["label"].astype(np.int64)`
  (assuming you have imported NumPy as `np`, as is convention) should not
  alter the meaning of the data.

  For text regression data (e.g. sentiment analysis, rating prediction), the
  file must be a two-column table with the columns named "text" and
  "target". The order of the columns is not important, but the "target"
  column must contain floating point values. The floating point data type is
  not really important, however, if the table is loaded into a Pandas
  DataFrame `df`, then running `df["label"].astype(float)` should not raise
  any exceptions.

  The "text" column will have "object" ("O") dtype. Assuming you have loaded
  your text data into a Pandas DataFrame `df`, then you can check that the
  data has the correct type by running:

      assert df.text.apply(lambda s: isinstance(s, str)).all()

  which will raise an AssertionError if a row has an incorrect type.

"""

OUT_HELP = """
Name of the .parquet file in which the embedded data will be saved. Can be
one a name, filename and extension, or a directory.

If passing in a name, e.g. 'my_embeddings', then the .parquet extension is
automatically appended.

Extensions other than .parquet are not valid and will raise an exception.

For safety reasons, and to avoid file access issues on e.g. compute clusters,
paths ABOVE the parent of the input data are not permitted. E.g. if your
input image data to be embedded is in:

    /path/to/my/inputs/my_images.parquet/

then you will only be able to specify outputs such as:

    /path/to/my/inputs/my_embeddings.parquet
    /path/to/my/inputs/images/my_embeddings.parquet
    /path/to/my/inputs/images/[...]/my_embeddings.parquet

and so forth. If working on an HPC cluster (especially a cluster with
different $HOME and $SCRATCH filesystems, as on e.g. SLURM clusters in
Compute Canada / DRAC, it is highly recommended that you use the `realpath`
or `readlink -f` command first to find out the actual path to your input
file. I.e. running:

    python df-embed.py --data ./my_images.parquet

tends to be a recipe for trouble. You might instead try:

    python df-embed.py --data "$(realpath ./my_images.parquet)"
    python df-embed.py --data "$(readlink -f ./my_images.parquet)"

to avoid these issues.

If you do not specify any output file, then the default is for the output to
go to a file `embeddings.parquet` in the parent directory of the input
parquet file passed to `--data`

"""

NAME_HELP = """
Unique name to label the dataset for debugging / testing purposes. Only
alters CLI outputs and / or produced test files.

"""

FORCE_DL_HELP = """
Download the NLP and vision zero-shot embedding models, overwriting existing
files if present.

"""


class EmbeddingModality(Enum):
    NLP = "nlp"
    Vision = "vision"


class EmbeddingOptions(Debug):
    def __init__(
        self,
        datapath: Path,
        modality: Union[str, EmbeddingModality],
        name: Optional[str] = None,
        outpath: Optional[Path] = None,
        download: bool = False,
        force_download: bool = False,
    ) -> None:
        # memoization-related
        # other
        self.datapath: Path = self.validate_datapath(datapath)
        self.modality = EmbeddingModality(modality)
        self.name = name or self.datapath.stem
        self.outpath = outpath or self.datapath.parent / "embedded.parquet"
        self.download: bool = download
        self.force_download: bool = force_download

    @staticmethod
    def validate_datapath(df_path: Path) -> Path:
        datapath = resolved_path(df_path)
        if not datapath.exists():
            raise FileNotFoundError(f"The specified file {datapath} does not exist.")
        if not datapath.is_file():
            raise FileNotFoundError(f"{datapath} is not a file.")
        return Path(datapath).resolve()

    @classmethod
    def from_parser(
        cls: Type[EmbeddingOptions], parser: ArgumentParser
    ) -> EmbeddingOptions:
        args = parser.parse_known_args()[0]
        return cls(
            datapath=args.data,
            modality=args.modality,
            name=args.name,
            outpath=args.out,
            download=args.download,
            force_download=args.force_download,
        )

    def __str__(self) -> str:
        cls = self.__class__.__name__
        mod = self.modality.name
        fields = {**self.__dict__}
        fields.pop("datapath")
        fields.pop("name")
        fields.pop("modality")
        info = []
        for key, value in fields.items():
            info.append(f"    '{key}': {repr(value)}")  # yes, will break on recursive
        info = "\n".join(info)
        return f"{cls}({self.name}.{mod} @ {self.datapath.parent}\n{info}\n)"

    __repr__ = __str__


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="df-embed",
        usage=USAGE_STRING,
        formatter_class=RawTextHelpFormatter,
        epilog=USAGE_EXAMPLES,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=DATA_HELP,
    )
    parser.add_argument(
        "--modality",
        choices=["vision", "nlp"],
        help="specify the input data type (modality)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        required=False,
        help=NAME_HELP,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=OUT_HELP,
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the NLP and vision zero-shot embedding models.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help=FORCE_DL_HELP,
    )
    # subparsers = parser.add_subparsers(
    #     title="embedding",
    #     description="Tools for embedding image or NLP datasets into a tabular form",
    #     dest="embedding",
    #     required=False,
    #     help="I am subcommand!",
    # )
    return parser
