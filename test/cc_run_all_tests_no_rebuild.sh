#!/bin/bash

TESTS=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT="$(dirname "$TESTS")"
echo "$ROOT"
cd "$ROOT" || exit 1

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
    test/test_name_sanitize.py \
    test/test_datasets.py \
    test/test_saving.py \
    test/test_models.py \
    test/test_tuning_score.py \
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
# test_name_sanitize.py
# test_tuning_score.py

# # Integration
# test_main.py
# test_spreadsheets.py

# # ERRORS!
# test_mlp.py  # segmentation fault


## TODO
# test_selection.py