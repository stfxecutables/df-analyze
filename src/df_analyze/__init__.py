import os
import sys
from pathlib import Path


def _migrate_torch_alloc_conf() -> None:
    deprecated = "PYTORCH_CUDA_ALLOC_CONF"
    current = "PYTORCH_ALLOC_CONF"
    value = os.environ.pop(deprecated, None)
    if value is not None:
        os.environ.setdefault(current, value)


_migrate_torch_alloc_conf()

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.append(str(SRC))


# from df_analyze._main import main
# def main() -> int:
#     print("Hello from rye-learn!")
#     return 0
