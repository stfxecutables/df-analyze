#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=24:00:00
#SBATCH --job-name=embed_df_test
#SBATCH --output=embed_df_test__%j_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --profile=all
#SBATCH --array=0-7

PROJECT=$SCRATCH/df-analyze

cd "$PROJECT" || exit 1
bash embed_df_test_cc.sh

