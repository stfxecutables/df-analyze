from pathlib import Path

VAL_SIZE = 0.20
SEED = 69  # because we are all such mature people here

DATADIR = Path(__file__).resolve().parent.parent / "data"
DATAFILE = DATADIR / "MCICFreeSurfer.mat"
DATA_JSON = DATAFILE.parent / "mcic.json"
CLEAN_JSON = DATAFILE.parent / "mcic_clean.json"
UNCORRELATED = DATADIR / "mcic_uncorrelated_cols.json"

CLASSIFIERS = ["rf", "svm", "dtree", "mlp", "bag"]
FEATURE_SELECTIONS = ["step-up", "pca", "kpca", "d", "auc", "pearson"]
FEATURE_CLEANINGS = ["correlated", "constant", "lowinfo"]
HTUNE_VAL_METHODS = ["holdout", "kfold", "k-fold", "loocv", "mc", "none"]
