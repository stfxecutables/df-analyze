from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATADIR = ROOT / "data"
SIMPLE_XLSX = DATADIR / "spreadsheet.xlsx"
SIMPLE_CSV = DATADIR / "spreadsheet.csv"
SIMPLE_ODS = DATADIR / "spreadsheet.ods"

DATAFILE = DATADIR / "MCICFreeSurfer.mat"
DATA_JSON = DATAFILE.parent / "mcic.json"
CLEAN_JSON = DATAFILE.parent / "mcic_clean.json"
UNCORRELATED = DATADIR / "mcic_uncorrelated_cols.json"

CLASSIFIERS = ["rf", "svm", "dtree", "mlp", "bag"]
REGRESSORS = ["linear", "rf", "svm", "adaboost", "gboost", "mlp", "knn"]
FEATURE_SELECTIONS = ["step-up", "step-down", "pca", "kpca", "d", "auc", "pearson", "none"]
FEATURE_CLEANINGS = ["correlated", "constant", "lowinfo"]
CLEANINGS_SHORT = ["corr", "const", "info"]
HTUNE_VAL_METHODS = ["holdout", "kfold", "k-fold", "loocv", "mc", "none"]

VAL_SIZE = 0.20
SEED = 69
