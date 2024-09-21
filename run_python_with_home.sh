#!/bin/bash

PROJECT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT="$(dirname "$PROJECT")"
cd "$ROOT" || exit 1
echo "Running script from: $ROOT"

# grotesque hack to deal with shitty python imports never working on CC
# file system in apptainer
DF_ANALYZE_SRC="$ROOT/src"
export PATH="$PATH:$DF_ANALYZE_SRC"
export APPTAINERENV_PATH="$PATH:$APPTAINERENV_PATH"

# silence matlplotlib warning on readonly filesystems
export APPTAINERENV_MPLCONFIGDIR="$(readlink -f .)"/.mplconfig
export APPTAINERENV_OPENBLAS_NUM_THREADS="1"
apptainer run --home $(readlink -f .) --app python df_analyze.sif $@
