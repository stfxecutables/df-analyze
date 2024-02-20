from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import re
import unicodedata
from argparse import ArgumentParser
from pathlib import Path

SCRIPT_OUTDIR = Path(__file__).resolve().parent / "student_scripts"
if not SCRIPT_OUTDIR.exists():
    SCRIPT_OUTDIR.mkdir(exist_ok=True, parents=True)

SCRATCH = os.environ.get("SCRATCH")
ACCOUNT = os.environ.get("SALLOC_ACCOUNT")
USER = os.environ.get("USER")
if ACCOUNT is None or USER is None or SCRATCH is None:
    raise EnvironmentError(
        "This script must be run on a SLURM Compute Canada / DRAC Cluster"
    )

HELP = """Path to directory containing subdirectories named after students,
with each subdirectory having the spreadsheet file contained inside. E.g.

```python gen_student_scripts.py student_sheets```

Would expect the `student_sheets` directory to look something like:

└── student_sheets/
    ├── alfred_pennyworth/
    │   └── assignment_data.xlsx
    ├── john_smith/
    │   └── marketing_data.xlsx
    │   ...
    ├── татьяна_юмашевt/
    │   └── cpu_cycles.xlsx
    └── 李娜/
        └── 北京_district_data.xlsx

This is because, internally, the script will recursively search for all files
with extensions .xlsx and .csv, and then use assume the parent folder has the
student's name. Then, df-analyze outputs will go to a sub-subdirectory of the
same name, e.g.:

└── student_sheets/
    └── alfred_pennyworth/
        └── alfred_pennyworth/     #  outputs will go here
        └── assignment_data.xlsx
"""

HEADER = f"""#!/bin/bash
#SBATCH --account={ACCOUNT}
#SBATCH --time=24:00:00
#SBATCH --job-name={{jobname}}
#SBATCH --output={{jobname}}_outputs__%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mail-user={USER}@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --profile=all
"""


SCRIPT = """
PROJECT=$SCRATCH/df-analyze
cd "$PROJECT" || exit 1

bash run_python_with_home.sh df-analyze.py --spreadsheet "{sheet}" --outdir "{outdir}"
"""


def to_filename(s: str, allow_unicode: bool = False):
    """
    See https://stackoverflow.com/a/295466.

    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    s = str(s)
    if allow_unicode:
        s = unicodedata.normalize("NFKC", s)
    else:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s-]", "", s.lower())
    return re.sub(r"[-\s]+", "-", s).strip("-_")


def generate_script(jobname: str, sheet: Path, outdir: Path) -> str:
    out = str(outdir.resolve())
    header = HEADER.format(jobname=jobname)
    script = f"{header}{SCRIPT.format(sheet=sheet, outdir=out)}"
    return script


def get_options() -> Path:
    parser = ArgumentParser()
    parser.add_argument("sheets", type=Path, help=HELP)
    args = parser.parse_args()
    sheets_dir = Path(args.sheets)
    return sheets_dir


def find_sheets() -> list[Path]:
    sheets_dir = get_options()
    xlsxs = sorted(sheets_dir.rglob("*.xlsx"))
    csvs = sorted(sheets_dir.rglob("*.csv"))
    all_sheets = xlsxs + csvs
    return all_sheets


if __name__ == "__main__":
    sheets = find_sheets()
    for sheet in sheets:
        name = to_filename(sheet.parent.stem)
        outdir = sheet.parent / name
        sheetname = to_filename(sheet.stem)
        jobname = f"{name}_{sheetname}"
        outdir.mkdir(exist_ok=True, parents=True)
        script = generate_script(jobname=jobname, sheet=sheet, outdir=outdir)
        script_out = SCRIPT_OUTDIR / f"submit_{jobname}.sh"
        script_out.write_text(script)
        print(f"Wrote job script to {script_out}")
