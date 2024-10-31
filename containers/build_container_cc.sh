#!/bin/bash
# shellcheck disable=SC2016
# shellcheck disable=SC2005
THIS_SCRIPT_PARENT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cd "$THIS_SCRIPT_PARENT" || exit 1

rm -f build.txt && bash build_container_cc_alt.sh 2> build.txt | tee -a build.txt
