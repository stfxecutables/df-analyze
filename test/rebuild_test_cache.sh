#!/bin/bash

TESTS=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT="$(dirname "$TESTS")"
echo "$ROOT"
cd "$ROOT" || exit 1


if [[ -z "${CC_CLUSTER}" ]]; then
    echo "On local machine, will use virtual environment for testing"
    VENV="$ROOT/.venv"
    PYTEST="$VENV/bin/pytest"
else
    # shellcheck disable=SC2016
    echo 'On Compute Canada, will use container-defined $PYTEST variable'
    module load apptainer
    export APPTAINERENV_MPLCONFIGDIR="$(readlink -f .)"/.mplconfig
    export APPTAINERENV_OPENBLAS_NUM_THREADS="1"
    apptainer run --home "$(readlink -f .)" df_analyze.sif "$(readlink -f test/cc_rebuild_test_cache.sh)"
    exit 0
fi


echo ""
echo "================================================================================="
echo "Testing inspection: should take about 2-4 minutes..."
echo "================================================================================="
echo ""
"$PYTEST" test/test_inspection.py -m 'regen' -x || { echo "Failed to regenerate inspections" ; exit 1; }


echo ""
echo "================================================================================="
echo "Testing data preparation: should take less than 5 minutes..."
echo "================================================================================="
echo ""
"$PYTEST" test/test_prepare.py -m 'regen' -x || { echo "Failed to generate prepared data"; exit 1; }

echo ""
echo "================================================================================="
echo "Testing association computations: should take less than 5 minutes..."
echo "================================================================================="
echo ""
"$PYTEST" test/test_associate.py -m 'regen and fast' -x || { echo "Failed to gen associations"; exit 1; }

echo ""
echo "================================================================================="
echo "Testing predictions: should take about 8-10 minutes..."
echo "================================================================================="
echo ""
"$PYTEST" test/test_predict.py::test_predict_fast -x || { echo "Failed to make predictions"; exit 1; }
"$PYTEST" test/test_predict.py::test_predict_cached_fast -x || { echo "Failed to load cached predictions"; exit 1; }

echo ""
echo "================================================================================="
echo "Testing fast feature selection method: will take hours, but this"
echo "is the last test, so if the first many pass, *probably* things are okay..."
echo "================================================================================="
echo ""
"$PYTEST" test/test_selection.py -m 'fast' -x || { echo "Failed to do selection"; exit 1; }