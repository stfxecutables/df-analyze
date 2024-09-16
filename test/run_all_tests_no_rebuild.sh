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
    test/test_datasets.py \
    test/test_loading.py \
    test/test_cli_random.py \
    test/test_cli_parsing.py \
    -x