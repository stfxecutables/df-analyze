# fmt: off
import os
import traceback
from pathlib import Path
from warnings import warn

DF_ENV = "DF_ANALYZE_CACHE"
DF_CACHE_NAME = "__DF_ANALYZE_CACHE__"
TESTFILE = "permission_test"

def is_writeable(dir: Path) -> bool:
    testfile = dir / TESTFILE
    if testfile.exists():
        testfile.unlink(missing_ok=True)
    try:
        with open(testfile, "x") as file:
            pass
        return True
    except OSError:
        traceback.print_exc()
        return False



def get_cache_dir() -> Path:
    existing = os.environ.get(DF_ENV)
    if existing is None:
        try:
            new = Path(os.environ.get("HOME")).resolve()
        except:
            traceback.print_exc()
            raise EnvironmentError("Could not resolve $HOME environment variable. Details above.")
        existing = new / DF_CACHE_NAME
        warn(f"""
No existing environment variable `DF_ANALYZE_CACHE` currently defined. A
default directory of {existing} will be used.
""")
        os.environ[DF_ENV] = str(existing)

    if not is_writeable(existing):
        raise EnvironmentError(f"""
Unable to write to {existing}. Specify a different cache directory either by
defining a permanent environment variable {DF_ENV} or, if in a Unix-based
system, by specifying the location prior to running `df-analyze`, e.g.:

    {DF_ENV}=<your/desired/cache/path/here> python df-analyze.py <options>
""")


from joblib import Memory

# fmt: on

CACHE_DIR
