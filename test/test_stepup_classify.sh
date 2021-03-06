#!/bin/bash
cd ..
source .venv/bin/activate
python df-analyze.py \
    --df="data/sklearn_classify/data.json" \
    --target='target' \
    --mode=classify \
    --classifiers svm dtree \
    --drop-nan=rows \
    --feat-clean constant lowinfo \
    --feat-select step-up \
    --n-feat=5 \
    --htune \
    --htune-trials=3 \
    --test-val=kfold \
    --test-val-size=5 \
    --outdir='test_results'
#    --regressors knn svm linear rf adaboost gboost mlp \
#    --regressors rf adaboost gboost mlp \