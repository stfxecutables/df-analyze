#!/bin/bash
python -X faulthandler df-analyze.py \
  --df Thyroid_diff.csv \
  --target Recurred \
  --outdir ./gandalf_test_results \
  --mode classify \
  --classifiers gandalf lgbm dummy \
  --feat-select none \
  --embed-select none \
  --wrapper-select none \
  --norm robust \
  --nan drop median \
  --htune-trials 8 \
  --htune-cls-metric auroc \
  --htune-reg-metric mae \
  --test-val-size 0.4