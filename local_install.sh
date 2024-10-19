#!/bin/bash

PROJECT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
VENV="$PROJECT/.venv"
ACTIVATE="$VENV/bin/activate"
PYTHON="$VENV/bin/python"
VERSION="3.12.5"

if ! command -v pyenv 2>&1 >/dev/null
then
    echo "================================================================================="
    echo "\`pyenv\` command not found. Install pyenv:"
    echo ""
    echo "    https://github.com/pyenv/pyenv"
    echo ""
    echo "or pyenv-win if on Windows and not using the Windows Subsytem for Linux (WSL):"
    echo ""
    echo "    https://github.com/pyenv-win/pyenv-win"
    echo ""
    echo "and reopen a new / fresh shell, and run this script again."
    echo ""
    echo "If you are sure you already have pyenv installed, then make sure it is on your"
    # shellcheck disable=SC2016
    echo '$PATH (https://github.com/pyenv/pyenv?tab=readme-ov-file#understanding-path)'
    echo "================================================================================="
    exit 1
else
    echo "\`pyenv\` command found. Compiling and installing python $VERSION"
fi


pyenv install "$VERSION" --skip-existing
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv shell $VERSION

echo "Python $VERSION installed. Preparing virtual environment using $(python --version)..."
echo "virtual environment will be installed in:"
echo "$VENV"

python -m venv .venv --clear
echo "Virtual environment created ($PYTHON). Installing core core dependencies"

echo "================================================================================="
echo "On Microsoft Windows, it may be required to enable the Activate.ps1 script by "
echo "setting the execution policy for the user. You can do this by issuing the "
echo "following PowerShell command:"
echo ""
echo "PS C:> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
echo ""
echo "See About Execution Policies for more information."
echo ""
echo "However, trying to get df-analyze working on Windows outside of the Windows "
echo "Linux Subsystem is likely to be an exercise in futility. You should only be "
echo "running this script from inside the WSL. If you *really* insist on doing this "
echo "in plain Windows, then you should try to figure out how to make a virtual "
echo "environment through pyenv-win and then run the pip install command at the "
echo "bottom of this script, and proceed from there."
echo "================================================================================="

source "$ACTIVATE"

python -m pip install --upgrade pip setuptools wheel --no-cache-dir
python -m pip install \
    cli-test-helpers \
    joblib \
    jsonpickle \
    lightgbm \
    llvmlite \
    matplotlib \
    numba \
    numpy \
    openpyxl \
    optuna \
    pandas \
    pyarrow \
    pytest \
    "pytest-xdist[psutil]" \
    python-dateutil \
    scikit-image \
    scikit-learn \
    scipy \
    seaborn \
    statsmodels \
    tabulate \
    torch \
    torchaudio \
    torchvision \
    tqdm \
    typing_extensions \
    skorch \
    "transformers[torch]" \
    accelerate \
    "datasets[vision]" \
    protobuf \
    sentencepiece \
    'pytorch_tabular' \
    || echo "Failed to install some python libs"

# python -m pytest test/test_embedding.py::test_vision_padding || echo "Basic embedding test failed!"
