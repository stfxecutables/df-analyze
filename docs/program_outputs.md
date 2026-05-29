# Program Outputs

## Output Directories

- **inspection**: inferred cardinality (categorical, ordinal, continuous) and
  types for features
- **prepared**: summary of data preparation (compute times, final data shape
  after encoding and drops)
- **features**: information about the features
- **features/associations**: univariate (statistical) associations of each
  feature with the target variable
- **features/predictions**: predictive performance on the target for each
  feature in isolation
- **selection**: information about features selected and methods used during
  feature selection
- **selection/embed**: embedded feature selection (via LASSO or LightGBM)
  details
- **selection/filter**: feature selection based on univariate associations or
  predictive performance
- **selection/wrapper**: feature selection based on wrapper methods
  (currently only step-up and backward)
- **results**: final tables of predictive performance after feature selection
  and tuning
- **tuning**: (currently unused) params for each tuned model and each
  selection method

## Device, Backend, And Timing Metadata

Model tuning and final performance outputs include:

- **device**: a compact execution-device label for this model result, such as
  `cpu`, `cuda`, `mps`, or `accelerated`
- **requested_device**: the user request from `--device auto/cpu/cuda/mps`
- **resolved_device**: the global device resolved before model-specific
  fallback, such as `cpu`, `cuda`, or `mps`
- **torch_device**: the device used by PyTorch/skorch/Lightning components
- **backend**: the authoritative model implementation label, such as
  `lightgbm-cpu`, `catboost-gpu`, `xgboost-cuda`, `tabpfn-v3-cuda`,
  `cuml-knn`, or a CPU/sklearn fallback
- **model_version**: version metadata for models that expose it, such as
  `v3` or `v2_5` for TabPFN
- **tune_seconds**: time spent in hyperparameter tuning
- **eval_seconds**: time spent in final holdout and k-fold evaluation
- **total_seconds**: tuning plus evaluation time

A run with `--device cuda` can still report a CPU backend for a specific model
if that model's CUDA backend is unavailable or a particular configuration is not
supported. In that case `resolved_device` can be `cuda` while `device` is `cpu`.
When `device` and `backend` seem ambiguous, use `backend` to decide which model
implementation path actually ran.
The hashed run directory also includes `device_report.json`. It is intentionally
small and user-facing: `device` records the requested and resolved device,
`checks` records whether smoke checks ran, `selected_models` records the models
from this run, and `models` contains one entry per selected model/backend that
df-analyze actually needs to explain. Unselected models are omitted instead of
being represented by `null` fields. LightGBM `lgbm` and `rf` appear as CPU
entries because df-analyze pins those execution paths to CPU.

## Output Files

## Makrkdown Reports

- **`associations_report.md`**
  - summarizes the (statistical) associations between each feature and the
    target variable
  - columns (classification):
    - **mut_info**: `sklearn.feature_selection` mutual information, either
      classification or regression variant as appropriate
    - **t**: independent samples t-test
    - **t_p**: p-value for t-test
    - **U**: Mann-Whitney U statistic
    - **U_p**: p-value for Mann-Whitney U statistic
    - **W**: Brunner Munzel W
      (https://en.wikipedia.org/wiki/Brunner_Munzel_Test). Same idea as
      Mann-Whitney U, but less assumptions.
    - **W_p**: p-value for Brunner Munzel test
    - **cohen_d**: Cohen's d
    - **AUROC**: area under the ROC curve (defined as
      [here](https://en.wikipedia.org/w/index.php?title=Mann%E2%80%93Whitney_U_test&oldid=1188631305#Area-under-curve_(AUC)_statistic_for_ROC_curves))
    - **corr**: Pearson's correlation coefficient
    - **corr_p**: p-value for Pearson's correlation coefficient
  - columns (regression):
    - **mut_info**: `sklearn.feature_selection` mutual information, either
      classification or regression variant as appropriate
    - **pearson_r**: Pearson's correlation coefficient
    - **pearson_p**: p-value for Pearson's correlation coefficient
    - **spearman_r**: Spearman's correlation coefficient
    - **spearman_p**: p-value for Spearman's correlation coefficient
    - **F**: `sklearn.feature_selection.f_regression`, i.e. F-statistic
    - **F_p**: p-value for F-statistic above
    - **H**: Kruskal-Wallace H, i.e. one-way ANOVA on ranks
      (https://en.wikipedia.org/w/index.php?title=Kruskal%E2%80%93Wallis_one-way_analysis_of_variance&oldid=1193273201)
    - **H_p**: p-value for Kruskal-Wallace H

- **`predictions_report.md`**
  - summarizes the univariate predictive performance of each feature on the
    target variable
- **`short_inspection_report.md`**
  - summarizes the inferred cardinalities (categorical, ordinal, continuous)
    of each feature
  - also summarizes if any features have been identified as unusable (e.g.
    timeseries features, identifiers, or constant features)
- **`preparation_report.md`**
  - summarizes changes to data shape after encoding of NaNs and categoricals
- **`results_report.md`**
  - summarizes predictive performance of the tuned models based on each
    feature selection procedure
- **`embedded_selection_report.md`**
  - summarizes which features were selected by the embedded method, as well
    as the selecting model and overall feature importances generated by this
    model
- **`association_selection_report.md`**
  - summarizes which features were selected based on their statistical
    associations, and which measure of association was used
- **`prediction_selection_report.md`**
  - summarizes which features were selected based on their univariate
    predictive performance, and which performance metric was used
- **`wrapper_selection_report.md`**
  - summarizes which features were selected based on the wrapper selection
    method, and the wrapper model and method used


## Tables

Raw data (unrounded) used to generate the tables in each report are present
in `.csv` files in the same folder as the Markdown (`.md`) reports.

Compressed forms of each table are also often saved in `.parquet` [file
format](https://parquet.apache.org/docs/file-format/) in each directory.
