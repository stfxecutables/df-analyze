#!/bin/bash
set -e

TOKEN=$(cat secrets/pypi_token.txt)

rye show
rye sync
rye build --wheel
rye publish --yes -u __token__ --token "$TOKEN" dist/"df_analyze-$(rye version)-py3-none-any.whl"
