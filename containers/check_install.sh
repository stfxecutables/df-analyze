#!/bin/bash
# shellcheck disable=SC2016
# shellcheck disable=SC2005
THIS_SCRIPT_PARENT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cd "$THIS_SCRIPT_PARENT" || exit 1

CONTAINER="$(realpath "../df_analyze.sif")"
DFANALYZE="$(realpath "../df-analyze.py")"

if [ ! -f "$CONTAINER" ]; then
    echo "FAIL: Could not find container at $CONTAINER."
    exit 1;
else
    rm -f check.log
    bash ../run_python_with_home.sh "$DFANALYZE" --version 2> check.log || { echo "FAIL: Could not get df-analyze --version. See 'check.log'" ; exit 1; }
    rm -f check.log
fi
