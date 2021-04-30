from pathlib import Path

VAL_SIZE = 0.20
SEED = 69

DATADIR = Path(__file__).resolve().parent.parent / "data"
DATAFILE = DATADIR / "MCICFreeSurfer.mat"
DATA_JSON = DATAFILE.parent / "mcic.json"
CLEAN_JSON = DATAFILE.parent / "mcic_clean.json"
UNCORRELATED = DATADIR / "mcic_uncorrelated.json"
