#!/bin/bash
./run_python_with_home.sh -X faulthandler "$(realpath df-analyze.py)" \
  --df "$(realpath Thyroid_diff.csv)" \
  --outdir "$(realpath ./gandalf_test_results)" \
  --target Recurred \
  --mode classify \
  --classifiers gandalf dummy \
  --feat-select none \
  --embed-select none \
  --wrapper-select none \
  --norm robust \
  --nan drop median \
  --htune-trials 50 \
  --htune-cls-metric acc \
  --htune-reg-metric mae \
  --test-val-size 0.4