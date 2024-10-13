#!/bin/bash

# Only intended for use on MacOS and/or Linux local install
ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "$ROOT"
cd "$ROOT" || exit 1
ROOT="$(realpath "$ROOT")"

NLP_DATA="$ROOT/data/testing/embedded/NLP/classification"
VISION_DATA="$ROOT/data/testing/embedded/vision/classification"
NLP_OUT="$ROOT/embed_df_results/NLP"
VISION_OUT="$ROOT/embed_df_results/vision"


DATAS=(
  "$NLP_DATA/tweet_topic_single/all_2020.parquet"
  "$NLP_DATA/tweet_topic_single/all_2021.parquet"
  "$NLP_DATA/rotten_tomatoes/all.parquet"
  "$NLP_DATA/toxic-chat/data/0124/toxic-chat_annotation_all.parquet"
  "$NLP_DATA/toxic-chat/data/1123/toxic-chat_annotation_all.parquet"
  "$VISION_DATA/Anime-dataset/all.parquet"
  "$VISION_DATA/fast_food_image_classification/all.parquet"
)
OUTS=(
  "$NLP_OUT/tweet_topic_single_2020"
  "$NLP_OUT/tweet_topic_single_2021"
  "$NLP_OUT/rotten_tomatoes"
  "$NLP_OUT/toxic-chat-0124"
  "$NLP_OUT/toxic-chat-1123"
  "$VISION_OUT/Anime-dataset"
  "$VISION_OUT/fast_food_image_classification"
)

df-analyze () {
    if [[ -z "${CC_CLUSTER}" ]]; then  # local
        "$PYTHON" "$ROOT/df-analyze.py" "$@"
    else
        bash "$ROOT/run_python_with_home.sh" "$ROOT/df-analyze.py" "$@"
    fi
}

if [[ -z "${CC_CLUSTER}" ]]; then  # local
    echo "On local machine, will use virtual environment for testing"
    VENV="$ROOT/.venv"
    PYTHON="$VENV/bin/python"
else
    # shellcheck disable=SC2016
    echo 'On Compute Canada'
    module load apptainer
    export APPTAINERENV_MPLCONFIGDIR="$(realpath .)"/.mplconfig
    export APPTAINERENV_OPENBLAS_NUM_THREADS="1"
fi

for i in "${!DATAS[@]}"; do
    OUTDIR="$(dirname "${OUTS[$i]}")"
    OUT="${OUTS[$i]}"
    if [ ! -f "$OUT" ]; then
        mkdir -p "$OUTDIR"
        echo "Processing ${DATAS[$i]}..."
        df-analyze \
            --df "${DATAS[$i]}" \
            --outdir "${OUTS[$i]}" \
            --target target \
            --mode classify \
            --classifiers dummy knn lgbm lr rf sgd \
            --feat-select filter embed wrap \
            --embed-select lgbm \
            --wrapper-select step-up \
            --wrapper-model linear \
            --norm robust \
            --nan median \
            --n-feat-filter 64 \
            --n-feat-wrapper 10 \
            --filter-method pred \
            --redundant-wrapper-selection \
            --redundant-threshold 0.01 \
            --htune-trials 50 \
            --test-val-size 0.4 || exit 1
    fi
done

