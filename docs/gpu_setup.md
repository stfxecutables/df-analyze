# GPU Setup

This project keeps GPU-only dependencies optional. A normal df-analyze install
should work on CPU, while CUDA runs should use an environment prepared for the
optional GPU backends.

## Recommended Environments

Use two practical environments:

- **Base/CPU environment**: the normal `uv sync` or Windows legacy install. This
  includes CatBoost, XGBoost, LightGBM, PyTorch-backed models, and TabPFN, but it does
  not require RAPIDS/cuML.
- **CUDA environment**: Linux or WSL2 with a CUDA-compatible driver,
  CUDA-enabled PyTorch, RAPIDS/cuML, CatBoost, XGBoost, TabPFN, and a CUDA-enabled
  LightGBM build if you are benchmarking LightGBM outside df-analyze's default
  model execution path.
Do not add cuML to the base dependencies. RAPIDS/cuML is platform and CUDA-stack
specific, and generic Windows Python installs are not a reliable target for it.

For WSL2/Linux machines with a CUDA 13-capable NVIDIA driver, the project
provides a locked RAPIDS/cuML dependency group:

```bash
uv run df-analyze-gpu-setup
uv run --group cuda-cu13 python df-analyze.py --device cuda ...
```

Use this instead of `--no-sync` for first-time GPU setup. `df-analyze-gpu-setup`
detects native Windows, macOS, unsupported Linux architectures, missing NVIDIA
drivers, and the CUDA major version before choosing a setup path. After syncing
the CUDA dependency group, it runs a preflight over the CUDA-capable df-analyze
model families:

```text
torch, cuML KNN, cuML LogisticRegression, cuML ElasticNet, cuML MBSGD,
XGBoost CUDA, CatBoost GPU, MLP/skorch CUDA, GANDALF/Lightning CUDA, TabPFN API
```

LightGBM `lgbm`/`rf` and `dummy` are reported as CPU paths because df-analyze
does not use CUDA LightGBM for normal model execution. The locked `cuda-cu13`
group is limited to x86_64 Linux so normal Windows and CPU installs continue to
use the base dependency set. CUDA 12 can be attempted with
`df-analyze-gpu-setup --allow-inexact-cu12`, but that path deliberately uses
`uv sync --inexact` because the current Torch CUDA 12.8 stack and RAPIDS 26.4
cu12 packages cannot be represented together in one exact `uv.lock`.

For a smaller first-time check that matches a limited run, pass the models
directly:

```bash
uv run df-analyze-gpu-setup --models knn lr xgboost catboost mlp gandalf tabpfn
```

For only the historical core checks:

```bash
uv run df-analyze-gpu-setup --models core
```

TabPFN's default preflight verifies the installed API without calling `fit()` so
setup does not unexpectedly download model weights. To intentionally warm the
local TabPFN model cache and surface license/token problems before a long run:

```bash
uv run df-analyze-gpu-setup --models tabpfn --tabpfn-fit-smoke
```

## Backend Notes

- `--device auto` chooses CUDA first, then Apple MPS, then CPU.
- CUDA is considered available when PyTorch CUDA is available or a visible
  NVIDIA driver is detected. If the driver is visible but PyTorch is CPU-only,
  PyTorch/skorch/Lightning components remain on CPU while traditional CUDA
  backends are still checked.
- MPS is useful for PyTorch/skorch/Lightning and embeddings, but CUDA-only
  libraries such as cuML, CatBoost GPU, and XGBoost CUDA do not use MPS.
  LightGBM `lgbm`/`rf` stays on CPU in df-analyze.
- When CUDA resolves and selected classifier/regressor models can use RAPIDS
  cuML (`knn`, classifier `lr`, regressor `elastic`, or regressor `sgd`),
  df-analyze checks cuML at startup. If it is missing or its smoke check fails,
  df-analyze attempts to install the matching `cuml-cu12` or `cuml-cu13` package
  from NVIDIA's Python package index, then restarts the same command so cuML is
  loaded. Set `DF_ANALYZE_AUTO_INSTALL_CUML=0` to disable automatic cuML
  installation, or set `DF_ANALYZE_CUML_CUDA_SUFFIX=cu12` / `cu13` to override
  CUDA wheel selection.
- CatBoost currently uses one visible CUDA GPU. On shared machines, set
  `CUDA_VISIBLE_DEVICES` before launching df-analyze, or set
  `DF_ANALYZE_CATBOOST_DEVICES` when CatBoost needs a non-default device id
  within the visible CUDA set.
- XGBoost uses `device="cuda"` with `tree_method="hist"` when its CUDA smoke
  check passes. On shared machines, set `CUDA_VISIBLE_DEVICES` before launching
  df-analyze, or set `DF_ANALYZE_XGBOOST_DEVICE` when XGBoost needs a specific
  CUDA device string such as `cuda:1`.
- LightGBM `lgbm` and `rf` are pinned to CPU execution with `n_jobs=-1`, even
  when `--device cuda` is selected or `--device auto` resolves to CUDA. The
  LightGBM CUDA RF backend can be pathologically slow on some workloads, so
  df-analyze does not validate or auto-install a CUDA LightGBM build for model
  execution.
  Users who still want to benchmark CUDA LightGBM must install that build
  manually and opt in outside df-analyze's automatic setup. On Linux/WSL with
  CUDA build tools available, one convenient project-local install command is:

  ```bash
  uv pip install --python .venv/bin/python --reinstall-package lightgbm --no-binary lightgbm -C cmake.define.USE_CUDA=ON lightgbm
  ```

- cuML-backed KNN and linear/SGD paths are optional and may use smaller or
  different solver/search spaces than sklearn.
- The main `SGDClassifier` intentionally stays on sklearn because df-analyze
  classifier scoring can require probability outputs.

## TabPFN Checklist

Before non-interactive or cluster runs:

1. Install a `tabpfn` release that supports TabPFN-3 as the package default.
2. Use `--tabpfn-version auto` or `v3` for new runs, and
   `--tabpfn-version v2_5` only for older TabPFN 2.5 / DeepTune comparisons.
3. Use `--tabpfn-tune-folds 1` for practical runs. Use `3` or `5` only when
   tuning-time comparability is more important than runtime.
4. Confirm that the TabPFN model-weight license is acceptable for the final
   work.
5. Run one tiny local fit once so model weights are downloaded and any license
   or cache flow is handled before the long experiment job.
6. Use CUDA for realistic TabPFN runtime. This implementation falls back to CPU
   when the resolved device is MPS.

TabPFN receives raw categorical features rather than one-hot encoded dummy
columns. If feature selection chooses a one-hot column such as `city_Halifax`,
df-analyze maps it back to the base categorical feature `city` for the TabPFN
input matrix.

When TabPFN is selected, `device_report.json` checks whether the TabPFN API is
importable, but it does not call `fit()` or instantiate TabPFN models, so
writing the report should not download model weights. `df-analyze-gpu-setup`
uses the same default behavior unless `--tabpfn-fit-smoke` is provided. For CUDA
runs, the report also runs smoke checks for the selected CUDA-capable backend
families only.

Useful smoke tests:

```powershell
$env:DF_ANALYZE_REAL_TABPFN_TESTS='1'
uv run pytest test/test_tabpfn.py::test_real_tabpfn_tiny_fit_smoke -q

$env:DF_ANALYZE_REAL_TABPFN_DEVICE='cuda'
uv run pytest test/test_tabpfn.py::test_real_tabpfn_tiny_fit_smoke -q
```

## Validation

Run at least one small df-analyze command with the intended device and explicit
model list before launching long jobs. Include opt-in models directly when they
are part of the analysis:

```text
--classifiers dummy knn lgbm rf sgd lr catboost xgboost mlp gandalf tabpfn
--regressors dummy knn lgbm rf sgd elastic catboost xgboost mlp gandalf tabpfn
```

Review `device_report.json` plus the result-table `device`, `backend`,
`requested_device`, `resolved_device`, and `torch_device` columns after the run
to confirm which models actually used accelerated backends. Use `backend` as
the authoritative implementation label; `device` is only a compact grouping.
`device_report.json` is intentionally scoped to the current command: it has
`device`, `checks`, `selected_models`, and `models` sections and omits unrelated
unselected backend fields.

## Dependency Files

Use `pyproject.toml`/`uv.lock` as the current dependency source for the GPU and
TabPFN implementation. The legacy `requirements*.lock` and
`local_requirements*.txt` files are retained for older install workflows and may
not include the current optional GPU stack. Keep RAPIDS/cuML environment setup
separate from the base dependencies because it depends on the CUDA, driver, OS,
and Python combination.
