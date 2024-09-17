#!/bin/bash

# Only intended for use on MacOS and/or Linux local install
TESTS=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT="$(dirname "$TESTS")"
echo "$ROOT"
cd "$ROOT" || exit 1
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
PYTEST="$VENV/bin/pytest"

"$PYTEST" \
    -m 'not regen' -m 'cached' -x \
    test/test_inspection.py \
    test/test_prepare.py \
    test/test_associate.py \
    test/test_cleaning.py

"$PYTEST" \
    test/test_loading.py \
    test/test_cli_random.py \
    test/test_cli_parsing.py \
    test/test_datasets.py \
    test/test_saving.py \
    test/test_models.py \
    -x

"$PYTEST" \
    test/test_predict.py \
    test/test_selection.py \
    -m 'fast' -x


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

# # Integration
# test_main.py
# test_spreadsheets.py

# # ERRORS!
# test_mlp.py  # segmentation fault
# test_name_sanitize.py  # unclear, some simple logic error somewhere or off by one
# test_tuning_score.py  # ValueError: Generated training classification target is constant


## TODO
# test_selection.py