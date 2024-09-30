#!/bin/bash

TESTS=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT="$(dirname "$TESTS")"
echo "$ROOT"
cd "$ROOT" || exit 1

bash test/rebuild_test_cache.sh && bash test/run_all_tests_no_rebuild.sh