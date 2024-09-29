#!/bin/bash

# Only intended for use on MacOS and/or Linux local install
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
    apptainer run --home "$(readlink -f .)" df_analyze.sif "$(readlink -f test/cc_run_all_tests_no_rebuild.sh)"
    exit 0
fi


echo "================================================================================="
echo "Testing basic data inspection, cleaning, preparation, and associational stats"
echo "================================================================================="
"$PYTEST" \
    -m 'not regen' -m 'cached' -x \
    test/test_inspection.py \
    test/test_prepare.py \
    test/test_associate.py \
    test/test_name_sanitize.py \
    test/test_cleaning.py

echo "================================================================================="
echo "Testing basic CLI functionality"
echo "================================================================================="
"$PYTEST" \
    test/test_cli_parsing.py \
    test/test_cli_random.py \
    -x

echo "================================================================================="
echo "Testing test dataset IO and basics"
echo "================================================================================="
"$PYTEST" -n auto \
    test/test_loading.py \
    test/test_datasets.py \
    test/test_models.py \
    test/test_tuning_score.py \
    -x

echo "================================================================================="
echo "Testing result saving (slow)"
echo "================================================================================="
"$PYTEST" -n auto test/test_saving.py -x

echo "================================================================================="
echo "Testing prediction"
echo "================================================================================="
"$PYTEST" test/test_predict.py -m 'fast' -x

echo "================================================================================="
echo "Testing selection. This is very slow, and if the first 5-10 pass, the rest "
echo "likely will, so feel free to Ctrl+C or Ctrl+Z at this point..."
echo "================================================================================="
"$PYTEST" test/test_selection.py -m 'fast' -x

# # Done
# test_inspection.py
# test_prepare.py
# test_associate.py
# test_cleaning.py
# test_loading.py
# test_cli_random.py
# test_cli_parsing.py
# test_datasets.py
# test_saving.py
# test_models.py
# test_predict.py
# test_name_sanitize.py
# test_tuning_score.py

# # Integration
# test_main.py
# test_spreadsheets.py

# # ERRORS!
# test_mlp.py  # segmentation fault


## TODO
# test_selection.py