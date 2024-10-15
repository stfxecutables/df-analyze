#!/bin/bash

# Only intended for use on MacOS and/or Linux local install
ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "$ROOT"
cd "$ROOT" || exit 1
ROOT="$(realpath "$ROOT")"

DATA="$ROOT/data/testing/embedding/NLP/classification"
OUT="$ROOT/data/testing/embedded/NLP/classification"
mkdir -p "$OUT"
OUT="$(realpath "$ROOT/data/testing/embedded/NLP/classification")"


DATAS=(
  "$(realpath "$DATA/tweet_topic_single/all_2020.parquet")"
  "$(realpath "$DATA/tweet_topic_single/all_2021.parquet")"
  "$(realpath "$DATA/rotten_tomatoes/all.parquet")"
  "$(realpath "$DATA/toxic-chat/data/0124/toxic-chat_annotation_all.parquet")"
  "$(realpath "$DATA/toxic-chat/data/1123/toxic-chat_annotation_all.parquet")"
)
OUTS=(
  "$OUT/tweet_topic_single/all_2020.parquet"
  "$OUT/tweet_topic_single/all_2021.parquet"
  "$OUT/rotten_tomatoes/all.parquet"
  "$OUT/toxic-chat/data/0124/toxic-chat_annotation_all.parquet"
  "$OUT/toxic-chat/data/1123/toxic-chat_annotation_all.parquet"
)

embed () {
    if [[ -z "${CC_CLUSTER}" ]]; then  # local
        "$PYTHON" "$ROOT/df-embed.py" "$@"
    else
        bash "$ROOT/run_python_with_home.sh" "$ROOT/df-embed.py" "$@"
    fi
}

if [[ -z "${CC_CLUSTER}" ]]; then  # local
    echo "On local machine, will use virtual environment for testing"
    VENV="$ROOT/.venv"
    PYTEST="$VENV/bin/pytest"
    PYTHON="$VENV/bin/python"
    BATCHES=(
      2
      2
      2
      2
      2
    )
else
    # shellcheck disable=SC2016
    echo 'On Compute Canada, will use container-defined $PYTEST variable'
    module load apptainer
    export APPTAINERENV_MPLCONFIGDIR="$(realpath .)"/.mplconfig
    export APPTAINERENV_OPENBLAS_NUM_THREADS="1"
    BATCHES=(
      8
      8
      8
      40
      40
    )
fi

for i in "${!DATAS[@]}"; do
    OUTDIR="$(dirname "${OUTS[$i]}")"
    OUT="${OUTS[$i]}"
    if [ ! -f "$OUT" ]; then
        mkdir -p "$OUTDIR"
        echo "Processing ${DATAS[$i]}..."
        embed --data "${DATAS[$i]}" --modality nlp --out "$OUT" --batch-size "${BATCHES[$i]}"
    fi
done

