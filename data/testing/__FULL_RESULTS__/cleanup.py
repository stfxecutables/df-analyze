from pathlib import Path
from shutil import rmtree

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    feats = sorted(ROOT.rglob("features"))
    hashed = [feat.parent for feat in feats if feat.name == "features"]
    hashed = sorted(set(hashed))
    removes = []
    for dir in hashed:
        opts = list(dir.rglob("options.json"))
        if len(opts) > 0:
            print("Keep: ", dir)
        else:
            removes.append(dir)

    if input("Proceed with deletion? [y/N]\n").lower() == "y":
        for remove in tqdm(removes, desc="deleting"):
            rmtree(remove)
