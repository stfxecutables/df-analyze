#!/bin/bash
# shellcheck disable=SC2016
THIS_SCRIPT_PARENT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cd "$THIS_SCRIPT_PARENT" || exit 1

if [ "$PWD" = "$HOME" ]; then
    echo "You are running the script from the correct directory"
fi

case $(realpath .)/ in "$(realpath /home)"/*)
    echo "=================================================================="
    echo "ERROR:"
    echo ""
    echo "Containers can generate a large amount of files and should not "
    echo 'be built in the /home (i.e. $HOME) directory. Make sure to clone '
    echo 'df-analyze into $SCRATCH and build there.'
    echo "=================================================================="
esac

if [ -d "$SCRATCH/.singularity" ]; then
    echo "Found existing Singularity cache. This seems to sometimes cause "
    echo "inordinately long build times if present, so this will be removed "
    rm -rf "$SCRATCH/.singularity"
fi

if [ -d "$SCRATCH/.apptainer" ]; then
    echo "Found existing Apptainer cache. This seems to sometimes cause "
    echo "inordinately long build times if present, so this will be removed "
    rm -rf "$SCRATCH/.apptainer"
fi

apptainer build --fakeroot --force debian_app.sif build_debian.def
echo "Copying container to project root..."
cp debian_app.sif ../df_analyze.sif