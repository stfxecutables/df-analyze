#!/bin/bash
export APPTAINERENV_MPLCONFIGDIR="$(readlink -f .)"/.mplconfig
module load apptainer

PROJECT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$PROJECT" || exit 1
ROOT="$(readlink -f .)"
DATA="$ROOT"/cmc_data
RESULTS="$ROOT"/cmc_results
ABIDE_I="$DATA/ABIDE_I"
ABIDE_II="$DATA/ABIDE_II"
ADHD200="$DATA/ADHD200"
HCP="$DATA/HCP"

mkdir -p "$RESULTS"

ALL_DATA=(
    "$ABIDE_I/ABIDE_I_all_FS_all_CMC_all_phenotypic_reduced_TARGET__CLS__autism.parquet"
    "$ABIDE_I/ABIDE_I_all_FS_all_CMC_all_phenotypic_reduced_TARGET__CLS__dsm_iv_spectrum.parquet"
    "$ABIDE_I/ABIDE_I_all_FS_all_CMC_all_phenotypic_reduced_TARGET__MULTI__dsm_iv.parquet"
    "$ABIDE_I/ABIDE_I_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__fiq.parquet"
    "$ABIDE_I/ABIDE_I_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__int_g_like.parquet"
    "$ABIDE_I/ABIDE_I_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__piq.parquet"
    "$ABIDE_I/ABIDE_I_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__viq.parquet"
    "$ABIDE_II/ABIDE_II_all_FS_all_CMC_all_phenotypic_reduced_TARGET__CLS__autism.parquet"
    "$ABIDE_II/ABIDE_II_all_FS_all_CMC_all_phenotypic_reduced_TARGET__CLS__dsm5_spectrum.parquet"
    "$ABIDE_II/ABIDE_II_all_FS_all_CMC_all_phenotypic_reduced_TARGET__MULTI__dsm_iv.parquet"
    "$ABIDE_II/ABIDE_II_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__fiq.parquet"
    "$ABIDE_II/ABIDE_II_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__int_g_like.parquet"
    "$ABIDE_II/ABIDE_II_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__piq.parquet"
    "$ABIDE_II/ABIDE_II_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__viq.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__CLS__adhd_spectrum.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__MULTI__diagnosis.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__CONSTR__adhd_z.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__CONSTR__adhd_z_hyper.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__CONSTR__adhd_z_inattentive.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__CONSTR__adhd_z_sum.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__REDUCED__adhd_factor.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__adhd_score.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__adhd_score_hyper.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__adhd_score_inattentive.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__fiq.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__int_g_like.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__piq.parquet"
    "$ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__viq.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__emotion_perf.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__emotion_rt.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__gambling_perf.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__gambling_rt.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__int_g_like.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__language_perf.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__language_rt.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__neg_emotionality.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__p_matrices.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__psqi_latent.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__relational_rt.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__social_random_perf.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__social_rt.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__wm_perf.parquet"
    "$HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__wm_rt.parquet"
)

df_analyze () {
    mkdir -p "$2"
    apptainer run --home $(readlink -f .) --app python "$ROOT/df_analyze.sif" \
        "$ROOT/df-analyze.py" \
        --df "$3" \
        --mode "$1" \
        --target "$4" \
        --classifiers knn lgbm rf lr dummy \
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
        --n-feat-filter 20 \
        --n-feat-wrapper 20 \
        --redundant-wrapper-selection \
        --redundant-threshold 0.01 \
        --htune-trials 100 \
        --htune-cls-metric acc \
        --test-val-size 0.5 \
        --outdir "$2" 2>&1 | tee "$5"
}

i="$SLURM_ARRAY_TASK_ID"
# for path in "${DATA[@]}"; do

datapath="${ALL_DATA[$i]}"
name="$(basename "$datapath")"
[[ $name =~ (TARGET__(:?REG|CLS)__.*)\.parquet$ ]]
target=${BASH_REMATCH[1]}
outdir="$RESULTS"/"$(basename "$(dirname "$datapath")")"/"$target"
[[ $name =~ ((:?REG|CLS)__.*)\.parquet$ ]]
txtout="$outdir"/${BASH_REMATCH[1]}_term_outputs.txt
if [[ "$datapath" == *"REG"* ]]; then
    mode=regress
else
    mode=classify
fi


echo '--mode' "$mode";
echo '--data' "$datapath";
echo '--outdir' "$outdir";
echo '--tee' "$txtout"
echo '--target' "$target"

df_analyze "$mode" "$outdir" "$datapath" "$target" "$txtout"

exit 0




# apptainer run --home $(readlink -f .) --app python df_analyze.sif \
#     df-analyze.py \
#     --df "$DATA1" \
#     --mode classify \
#     --target TARGET__CLS__autism \
#     --classifiers knn lgbm rf lr sgd dummy \
#     --norm robust \
#     --nan median \
#     --feat-select filter embed wrap \
#     --embed-select lgbm linear \
#     --wrapper-select step-up \
#     --wrapper-model linear \
#     --filter-method assoc pred \
#     --filter-assoc-cont-classify mut_info \
#     --filter-assoc-cat-classify mut_info \
#     --filter-pred-classify acc \
#     --n-feat-filter 20 \
#     --n-feat-wrapper 20 \
#     --redundant-wrapper-selection \
#     --redundant-threshold 0.01 \
#     --htune-trials 100 \
#     --htune-cls-metric acc \
#     --test-val-size 0.5 \
#     --outdir "$RUN1" 2>&1 | tee "$OUTS1"

apptainer run --home $(readlink -f .) --app python df_analyze.sif \
    df-analyze.py \
    --df "$DATA2" \
    --mode classify \
    --target TARGET__REG__int_g_like \
    --classifiers knn lgbm rf lr sgd dummy \
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
    --n-feat-filter 20 \
    --n-feat-wrapper 20 \
    --redundant-wrapper-selection \
    --redundant-threshold 0.01 \
    --htune-trials 100 \
    --htune-cls-metric acc \
    --test-val-size 0.5 \
    --outdir "$RUN1" 2>&1 | tee "$OUTS1"