import os
from pathlib import Path

from df_analyze.hypertune import Classifier

CLASSIFIER_CHOICES = ["svm", "dtree", "bag", "rf", "mlp"]
RUNTIMES = {
    "svm": "6:00:00",
    "dtree": "6:00:00",
    "bag": "12:00:00",
    "rf": "24:00:00",
    "mlp": "24:00:00",
}
SCRIPT_OUTDIR = Path(__file__).resolve().parent / "job_scripts"
if not SCRIPT_OUTDIR.exists():
    os.makedirs(SCRIPT_OUTDIR, exist_ok=True)

HEADER = """#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time={time}
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}__%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL"""

SCRIPT = """
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

{}
"""


def generate_script(
    classifier: Classifier = "svm",
    step_up: bool = False,
    script_outdir: Path = SCRIPT_OUTDIR,
) -> str:
    pythonfile = "run.py"
    stepup = " --step-up" if step_up else ""
    job_name = classifier if not step_up else f"{classifier}_step-up"
    command = f"$PYTHON $PROJECT/{pythonfile} --classifier={classifier}{stepup}"
    header = HEADER.format(time=RUNTIMES[classifier], job_name=job_name)

    script = f"{header}{SCRIPT.format(command)}"
    out = script_outdir / f"submit_{job_name}.sh"
    with open(out, mode="w") as file:
        file.write(script)
    print(f"Saved downsampling script to {out}")

    return script


if __name__ == "__main__":
    print(f"Will save scripts in {SCRIPT_OUTDIR}")
    for classifier in CLASSIFIER_CHOICES:
        script = generate_script(classifier, step_up=True)
        script = generate_script(classifier, step_up=False)
