# Windows Installation

For those who can't use the [shell
script](https://github.com/stfxecutables/df-analyze?tab=readme-ov-file#local-install-by-shell-script)
to install the necessary `df-analyze` dependencies for a local install, this
will be the attempted installation procedure.

You must use PowerShell for all commands, not cmd.exe.

# `pyenv-win` Setup

1. Install [`pyenv-win`](https://github.com/pyenv-win/pyenv-win?tab=readme-ov-file#quick-start)
   by doing the following steps, in order:
   1. In PowerShell, run: `Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"`
   2. Reopen PowerShell
   3. Run `pyenv --version` to check if the installation was successful.
   4. Run `pyenv install 3.12.5`
   5. `cd` to the directory where you cloned `df-analyze`
   6. Run `pyenv local 3.12.5`
      - this creates a permanent file `.python-version` in the `df-analyze`
        directory, and whenever you subseqeuntly open a new PowerShell and `cd`
        to this location, ensures that the correct python version is automatically
        used
   7. Restart PowerShell
   8. `cd` to the directory where you cloned `df-analyze`
   9. Run `python --version` and confirm that the output is `Python 3.12.5`

For all future uses of PowerShell, it should now be that case that if you are in the
`df-analyze` directory, then the correct python version is used. You should always
check this before doing anything else by running:

```powershell
python --version
```

first before doing anything. If for some reason the above does not return `Python 3.12.5`,
you can manually activate the correct python version at any location by running

```powershell
pyenv shell 3.12.5
```

# Virtual Environment Creation

Make sure to `cd` to the `df-analye` directory, and that `python --version` returns `Python 3.12.5`.
Then, run the following command:

```powershell
python -m venv .venv
```

This creates a virtual environment directory `.venv` in the `df-analyze`
directory. Now, we need to activate this virtual environment (i.e. tell
PowerShell to use the Python binary and libraries contained in the local
`.venv` folder rather than the global python 3.12.5 installation files).
In order to this, it is best to first set some permissions by running:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

It might be a good idea to restart PowerShell after running this command.

## Activating the Virtual Environment

The environment can then be activated by running one of:

```powershell
.\.venv\Scripts\Activate
```

or

```powershell
.\.venv\Scripts\activate.ps1
```

or

```powershell
.\.venv\Scripts\Activate.ps1
```

To confirm this worked, you should be able to run one of:

```powershell
where python
```

or

```powershell
(get-command python).Path
```

You should see something like `df-analyze/.venv/bin/python.exe` somewhere in the output.
If you do, virtual environment creation was a success.

# Installing Libraries to the Virtual Environment

After you have confirmed that you have installed and activated the virtual
environment, first run the following command to update `pip`:

```powershell
python -m pip install --upgrade pip setuptools wheel --no-cache-dir
```

Then run the following command:

```powershell
python -m pip install cli-test-helpers joblib jsonpickle lightgbm llvmlite matplotlib numba numpy openpyxl optuna pandas pyarrow pytest "pytest-xdist[psutil]" python-dateutil scikit-image scikit-learn scipy seaborn statsmodels tabulate torch torchaudio torchvision tqdm typing_extensions skorch "transformers[torch]" accelerate "datasets[vision]" protobuf sentencepiece
```

All necessary dependencies should now be installed. You can verify that the installation
was a success by running:

```
python df-analyze.py --help
```

This should produce the documentation for the `df-analyze` tabular prediction interface.

# After Installation

Anytime you open PowerShell and navigate to the `df-analyze` directory, you will have
to [activate the virtual environment](#activating-the-virtual-environment) prior to
running `df-analyze`. But you won't have to do any of the other install procedures again.









