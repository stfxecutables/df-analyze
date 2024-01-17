#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=24:00:00
#SBATCH --job-name=test_main
#SBATCH --output=test_main__%j_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --profile=all
#SBATCH --array=0-70

PROJECT=$HOME/projects/def-jlevman/dberger/df-analyze

cd "$PROJECT" || exit 1
bash run_python_with_home.sh test/test_main.py

