#!/bin/bash
export MLPCONFIGDIR="$(readlink -f .)"/.mplconfig
apptainer run --home $(readlink -f .) --app python debian_app.sif $@
