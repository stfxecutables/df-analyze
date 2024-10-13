#!/bin/bash

# Only intended for use on MacOS and/or Linux local install
ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "$ROOT"
cd "$ROOT" || exit 1
ROOT="$(realpath "$ROOT")"

DATA="$ROOT/data/testing/embedding/vision/classification"
OUT="$ROOT/data/testing/embedded/vision/classification"
mkdir -p "$OUT"
OUT="$(realpath "$ROOT/data/testing/embedded/vision/classification")"


DATAS=(
  "$DATA/Anime-dataset/all.parquet"
  "$DATA/fast_food_image_classification/all.parquet"
)
OUTS=(
  "$OUT/Anime-dataset/all.parquet"
  "$OUT/fast_food_image_classification/all.parquet"
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
    )
else
    # shellcheck disable=SC2016
    echo 'On Compute Canada, will use container-defined $PYTEST variable'
    module load apptainer
    export APPTAINERENV_MPLCONFIGDIR="$(realpath .)"/.mplconfig
    export APPTAINERENV_OPENBLAS_NUM_THREADS="1"
    BATCHES=(
      8
      16
    )
fi

for i in "${!DATAS[@]}"; do
    OUTDIR="$(dirname "${OUTS[$i]}")"
    OUT="${OUTS[$i]}"
    if [ ! -f "$OUT" ]; then
        mkdir -p "$OUTDIR"
        echo "Processing ${DATAS[$i]}..."
        embed --data "${DATAS[$i]}" --modality vision --out "$OUT" --batch-size "${BATCHES[$i]}"
    fi
done

