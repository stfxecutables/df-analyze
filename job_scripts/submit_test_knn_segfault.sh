#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=24:00:00
#SBATCH --job-name=knn_segfault
#SBATCH --output=knn_segfault__%j_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --profile=all
#SBATCH --array=9,22,47,50,59,64,65,66,69

PROJECT=$SCRATCH/df-analyze

cd "$PROJECT" || exit 1
bash run_python_with_home.sh test/test_main.py

