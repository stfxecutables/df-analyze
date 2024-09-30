from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

VISION_ROOT = ROOT / "data/vision"
VISION_CLS = VISION_ROOT / "classification"
VISION_REG = VISION_ROOT / "regression"

import pandas as pd


def cleanup_parquets() -> None:
    cls_roots = sorted(VISION_CLS.glob("*"))
    reg_roots = sorted(VISION_REG.glob("*"))
    roots = cls_roots + reg_roots

    for root in roots:
        out = root / "all.parquet"
        pqs = sorted(root.glob("*.parquet*"))
        if out in pqs:
            pqs.remove(out)
        if len(pqs) == 0:
            continue

        df = pd.concat([pd.read_parquet(pq) for pq in pqs])
        df.drop(columns="image.path", inplace=True, errors="ignore")
        df.rename(columns={"image.bytes": "image"}, inplace=True)
        if isinstance(df["image"].iloc[0], dict):
            df["image"] = df["image"].apply(lambda d: d["bytes"])

        if root.name == "rare-species":
            df.rename(columns={"class": "label"}, inplace=True)
            df = df.loc[:, ["label", "image"]].copy()
        df.to_parquet(out)
        print(f"Saved single parquet file to {out}")


def inspect_cleaned() -> None:
    cls_roots = sorted(VISION_CLS.glob("*"))
    reg_roots = sorted(VISION_REG.glob("*"))
    roots = cls_roots + reg_roots
    for root in roots:
        print("=" * 81)
        print(f"{root.name}")
        try:
            # NOTE! Important to use pyarrow, fastparquet broken here
            df = pd.read_parquet(root / "all.parquet", engine="pyarrow")
            print(f"Columns: {sorted(df.columns.tolist())}")
            print(f"Image: {len(df)}")
            del df
        except Exception as e:
            print("BROKEN!")
            print(e)
            continue


if __name__ == "__main__":
    cleanup_parquets()
    inspect_cleaned()
