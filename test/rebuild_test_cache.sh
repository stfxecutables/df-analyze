#!/bin/bash

# Only intended for use on MacOS and/or Linux local install
TESTS=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT="$(dirname "$TESTS")"
echo "$ROOT"
cd "$ROOT" || exit 1
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
PYTEST="$VENV/bin/pytest"

# "$PYTEST" test/test_inspection.py -m 'regen' -x || echo "Failed to regenerate inspections" && exit 1
# "$PYTEST" test/test_inspection.py -m 'not regen' -x || echo "Failed to load cached inspections" && exit 1
# "$PYTEST" test/test_prepare.py -m 'regen' -x || echo "Failed to generate prepared data" && exit 1
# "$PYTEST" test/test_prepare.py -m 'not regen' -x || echo "Failed to load cached prepared data" && exit 1
# "$PYTEST" test/test_associate.py -m 'regen' -x || echo "Failed to gen associations" && exit 1
# "$PYTEST" test/test_associate.py -m 'not regen' -x || echo "Failed to load cached associations" && exit 1
"$PYTEST" test/test_associate.py -m 'not regen' -x || echo "Failed to load cached associations" && exit 1