#!/bin/bash
export APPTAINERENV_MPLCONFIGDIR="$(readlink -f .)"/.mplconfig
export APPTAINERENV_OPENBLAS_NUM_THREADS="1"
apptainer run --home $(readlink -f .) --app pytest df_analyze.sif $*
