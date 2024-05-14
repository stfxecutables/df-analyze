#!/bin/bash
export APPTAINERENV_MPLCONFIGDIR="$(readlink -f .)"/.mplconfig
module load apptainer

PROJECT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$PROJECT" || exit 1
ROOT="$(readlink -f .)"
RESULTS="$ROOT/traffic_results"
TERM_OUT="$RESULTS/terminal_outputs.txt"
DATA="$ROOT"/traffic_data
SHEET="$DATA/traffic_data_processed.parquet"

# choices: outcome, search_conducted, search_disposition, search_type
TARGET=outcome
CATS='acc_blame,accident,alcohol,article,belts,chrg_sect_mtch,chrg_title,chrg_title_mtch,comm_license,comm_vehicle,fatal,hazmat,home_outstate,licensed_outstate,outcome,patrol_entity,pers_injury,prop_dmg,race,search_conducted,search_disposition,search_type,sex,stop_chrg_title,subagency,tech_arrest,unmarked_arrest,vehicle_color,vehicle_make,vehicle_type,violation_type,work_zone'
ORDS='year_of_stop,month_of_stop,weeknum_of_stop,weekday_of_stop'
# [OPTIONAL]
# DROPS='vehicle_make,vehicle_color,subagency,patrol_entity'
DROPS='search_conducted,search_disposition,search_type'

mkdir -p "$RESULTS"

df_analyze () {
    mkdir -p "$RESULTS"
    apptainer run --home $(readlink -f .) --app python "$ROOT/df_analyze.sif" \
        "$ROOT/df-analyze.py" \
        --df "$SHEET" \
        --mode classify \
        --target "$TARGET" \
        --categoricals "$CATS" \
        --ordinals "$ORDS" \
        --drops "$DROPS" \
        --classifiers lgbm rf sgd dummy \
        --norm robust \
        --nan median \
        --feat-select filter embed wrap \
        --embed-select lgbm linear \
        --wrapper-select step-up \
        --wrapper-model linear \
        --filter-method assoc pred \
        --filter-assoc-cont-classify mut_info \
        --filter-assoc-cat-classify mut_info \
        --filter-pred-classify acc \
        --n-feat-filter 10 \
        --n-feat-wrapper 10 \
        --redundant-wrapper-selection \
        --redundant-threshold 0.01 \
        --htune-trials 100 \
        --htune-cls-metric acc \
        --test-val-size 0.25 \
        --outdir "$RESULTS" 2>&1 | tee "$TERM_OUT"
}

df_analyze