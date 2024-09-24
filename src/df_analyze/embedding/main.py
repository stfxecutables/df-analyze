from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from df_analyze.embedding.cli import make_parser


def main() -> None:
    """Do embedding logic here"""
    parser = make_parser()
    args = parser.parse_known_args()[0]
    print(args)
