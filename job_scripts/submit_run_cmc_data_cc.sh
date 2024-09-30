#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=08:00:00
#SBATCH --job-name=cmc_all
#SBATCH --output=cmc_all__%j_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --profile=all
#SBATCH --array=0-42


PROJECT=$SCRATCH/df-analyze
cd "$PROJECT" || exit 1
bash run_cmc_data_cc.sh
