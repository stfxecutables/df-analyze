#!/bin/bash
source .venv/bin/activate
python df-analyze.py \
    --df="data/sklearn_regression/data.json" \
    --target='target' \
    --mode=regress \
    --regressors svm linear \
    --drop-nan=rows \
    --feat-clean=constant \
    --feat-select pca pearson \
    --n-feat=10 \
    --htune \
    --htune-trials=10 \
    --test-val=kfold \
    --test-val-size=5 \
    --outdir='test_results'
