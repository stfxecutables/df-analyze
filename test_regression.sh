#!/bin/bash
source .venv/bin/activate
python df-analyze.py \
    --df="data/sklearn_regression/data.json" \
    --target='target' \
    --mode=regress \
    --regressors rf adaboost gboost mlp \
    --drop-nan=rows \
    --feat-clean constant lowinfo \
    --feat-select kpca pca pearson none \
    --n-feat=10 \
    --htune \
    --htune-trials=3 \
    --test-val=kfold \
    --test-val-size=5 \
    --outdir='test_results'
#    --regressors knn svm linear rf adaboost gboost mlp \