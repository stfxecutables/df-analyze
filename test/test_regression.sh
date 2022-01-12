#!/bin/bash
source ../.venv/bin/activate
python ../df-analyze.py \
    --df="data/sklearn_regression/data.json" \
    --target='target' \
    --mode=regress \
    --regressors svm linear \
    --drop-nan=rows \
    --feat-clean constant lowinfo \
    --feat-select step-down \
    --n-feat=10 \
    --htune \
    --htune-trials=3 \
    --test-val=kfold \
    --test-val-size=5 \
    --outdir='test_results'
#    --regressors knn svm linear rf adaboost gboost mlp \
#    --regressors rf adaboost gboost mlp \