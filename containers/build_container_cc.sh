#!/bin/bash
# shellcheck disable=SC2016
# shellcheck disable=SC2005
THIS_SCRIPT_PARENT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cd "$THIS_SCRIPT_PARENT" || exit 1

case $(realpath .)/ in "$(realpath /home)"/*)
    echo "=================================================================="
    echo "ERROR:"
    echo ""
    echo "Containers can generate a large amount of files and should not "
    echo 'be built in the /home (i.e. $HOME) directory. Make sure to clone '
    echo 'df-analyze into $SCRATCH and build there.'
    echo "=================================================================="
    exit 1
esac

if [ -d "$SCRATCH/.singularity" ]; then
    echo "=================================================================="
    echo "Found existing Singularity cache. This seems to sometimes cause "
    echo "inordinately long build times if present, so this will be removed "
    echo "=================================================================="
    rm -rf "$SCRATCH/.singularity"
fi

if [ -d "$SCRATCH/.apptainer" ]; then
    echo "=================================================================="
    echo "Found existing Apptainer cache. This seems to sometimes cause "
    echo "inordinately long build times if present, so this will be removed "
    echo "=================================================================="
    rm -rf "$SCRATCH/.apptainer"
fi

apptainer build --fakeroot --force debian_app.sif build_debian.def
echo "Copying container to project root..."
cp debian_app.sif ../df_analyze.sif

echo "=================================================================="
echo "Container built successfully. Built container located at:"
echo "$(realpath ../df_analyze.sif)"
echo "=================================================================="