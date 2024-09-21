#!/bin/bash

# Only intended for use on MacOS and/or Linux local install
TESTS=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT="$(dirname "$TESTS")"
echo "$ROOT"
cd "$ROOT" || exit 1

echo ""
echo "================================================================================="
echo "Testing inspection: should take about 2-4 minutes..."
echo "================================================================================="
echo ""
"$PYTEST" test/test_inspection.py -m 'regen' -x
"$PYTEST" test/test_inspection.py -m 'not regen' -x


echo ""
echo "================================================================================="
echo "Testing data preparation: should take less than 10 minutes..."
echo "================================================================================="
echo ""
"$PYTEST" test/test_prepare.py -m 'regen' -x
"$PYTEST" test/test_prepare.py -m 'not regen' -x

echo ""
echo "================================================================================="
echo "Testing association computations: could take up to 45 minutes..."
echo "================================================================================="
echo ""
"$PYTEST" test/test_associate.py -m 'regen' -x
"$PYTEST" test/test_associate.py -m 'not regen' -x

echo ""
echo "================================================================================="
echo "Testing predictions: should take about 10-20 minutes..."
echo "================================================================================="
echo ""
"$PYTEST" test/test_predict.py::test_predict_fast -x
"$PYTEST" test/test_predict.py::test_predict_cached_fast -x

echo ""
echo "================================================================================="
echo "Testing fast feature selection method: could take hours, but this"
echo "is the last test, so if the first many pass, *probably* things are okay..."
echo "================================================================================="
echo ""
"$PYTEST" test/test_selection.py -m 'fast' -x