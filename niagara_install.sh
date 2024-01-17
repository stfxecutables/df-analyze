#!/bin/bash
module load gcc cmake
module load python/3.9.8
python -m venv .venv
.venv/bin/python -m pip install -r niagara_requirements.txt
