#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=24:00:00
#SBATCH --job-name=mlp
#SBATCH --output=mlp%A_array%a__%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
PROJECT=$HOME/projects/def-jlevman/dberger/df-analyze

module load python/3.8.2
cd $SLURM_TMPDIR
tar -xzf $PROJECT/venv.tar.gz .venv
source .venv/bin/activate

PYTHON=$(which python)

# virtualenv --no-download .venv
# source .venv/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r $PROJECT/requirements.txt

$PYTHON $PROJECT/analysis/main.py --classifier=mlp
