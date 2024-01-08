```sh
usage:

    The df-analyze program can be used in one of two modes: CLI mode, and
    spreadsheet mode. In spreadsheet mode, df-analyze options are specified in
    a special format at the top of a spreadsheet or .csv file, and spreadsheet
    columns are given specific names to identify targets, continuous features,
    and categorical features. In spreadsheet mode, only a single argument needs
    to be passed, which is the path to the df-analyze formatted spreadsheet:

        python df-analyze.py --spreadsheet my_formatted_sheet.xlsx

optional arguments:
  -h, --help            show this help message and exit
  --spreadsheet SPREADSHEET

                        The path to the formatted spreadsheet to analyze.

                        Currently only spreadsheets saved as either `.xlsx` or `.csv` are supported.

                        If your data is saved as a Pandas `DataFrame`, it must have shape
                        `(n_samples, n_features)` or `(n_samples, n_features + 1)`. The name of the
                        column holding the target variable (or feature) can be specified by the
                        `--target` / `-y` argument, but is "target" by default if such a column name
                        exists, or the last column if it does not.

                        If your data is in a NumPy array, the array must have the shape
                        `(n_samples, n_features + 1)` where the last column is the target for either
                        classification or prediction.

  --df DF
                        The dataframe to analyze.

                        Currently only Pandas `DataFrame` objects saved as either `.json` or `.csv`, or
                        NumPy `ndarray`s saved as "<filename>.npy" are supported, but a Pandas
                        `DataFrame` is recommended.

                        If your data is saved as a Pandas `DataFrame`, it must have shape
                        `(n_samples, n_features)` or `(n_samples, n_features + 1)`. The name of the
                        column holding the target variable (or feature) can be specified by the
                        `--target` / `-y` argument, but is "target" by default if such a column name
                        exists, or the last column if it does not.

                        If your data is in a NumPy array, the array must have the shape
                        `(n_samples, n_features + 1)` where the last column is the target for either
                        classification or prediction.

  --separator SEPARATOR

                        Separator used in .csv files. Default ",".
  --target TARGET
                        The location of the target variable for either regression or classification.

                        If a string, then `--df` must be a Pandas `DataFrame` and the string passed in
                        here specifies the name of the column holding the targer variable.

                        If an integer, and `--df` is a NumPy array only, specifies the column index.

  --categoricals CATEGORICALS [CATEGORICALS ...]

                        A string or list of strings, e.g.

                            --categoricals sex gender ancestry education

                        that specifies which features will be treated as categorical regardless of the
                        number of levels or format of the data. If during data cleaning categorical
                        variables are detected that are NOT specified by the user, a warning will be
                        raised.
  --ordinals ORDINALS [ORDINALS ...]

                        A string or list of strings, e.g.

                            --ordinals star_rating number_of_purchases times_signed_in

                        that specifies which features will be treated as ordinal regardless of the
                        number of levels or format of the data. If during data cleaning categorical
                        variables are detected that are NOT specified by the user, a warning will be
                        raised. If the values of the specified variables cannot be interpreted as
                        integers, then df-analyze will exit with an error.
  --drops DROPS [DROPS ...]

                        A string or list of strings, e.g.

                            --drops subject_id location_id useless_feature1 useless_feat2

                        that specifies which features will be removed from the data and not considered
                        for any inspection, description or univariate analysis, and which will not be
                        included in any feature selection, model tuning, or final predictive models.
  --mode {classify,regress}

                        If "classify", do classification. If "regress", do regression.

  --classifiers {rf,svm,dtree,mlp,bag,dummy,lgb} [{rf,svm,dtree,mlp,bag,dummy,lgb} ...]

                        The list of classifiers to use when comparing classification performance.
                        Can be a list of elements from: bag dtree dummy lgb mlp rf svm.

  --regressors {linear,rf,svm,adaboost,gboost,mlp,knn,lgb} [{linear,rf,svm,adaboost,gboost,mlp,knn,lgb} ...]

                        The list of regressors to use when comparing regression model performance.
                        Can be a list of elements from: adaboost gboost knn lgb linear mlp rf svm.

  --feat-select {filter,embed,wrap} [{filter,embed,wrap} ...]

                        The feature selection methods to use. Available options are:

                          auc:        Select features with largest AUC values relative to the two
                                      classes (classification only).

                          d:          Select features with largest Cohen's d values relative to the two
                                      classes (classification only).

                          kpca:       Generate features by using largest components of kernel PCA.

                          pca:        Generate features by using largest components from a PCA.

                          pearson:    Select features with largest Pearson correlations with target.

                          step-up:    Use step-up (forward) feature selection. Costly.

                          step-up:    Use step-down (backward) feature selection. Also costly.

                        NOTE: Feature selection currently uses the full data provided in the `--df`
                        argument to `df-analyze.py`. Thus, if you take the final reported test results
                        following feature selection and hyperparamter tuning as truly cross-validated
                        or heldout test results, you are in fact double-dipping and reporting biased
                        performance. To properly test the discovered features and optimal estimators,
                        you should have held-out test data that never gets passed to `df-analyze`.

  --model-select {embed,wrap,none} [{embed,wrap,none} ...]

                        Methods of model-based feature selection methods to use. Available options are:

                          embed:      Select using an embedded method, i.e. a method where the model
                                      produces values for each feature that can be interpreted as
                                      feature importances. Which model is used is determined by
                                      `--embed-model`.

                          wrap:       Select using a wrapper method, i.e. a method which uses ("wraps")
                                      a specific model, and then optimizes the feature set via some
                                      alternation of model evaluations and feature-space search /
                                      navigation strategy.

                          none:       Do not select features using any model.
  --embed-select {lgbm,linear,none} [{lgbm,linear,none} ...]

                        Methods of model-based feature selection methods to use. Available options are:

                          embed:      Select using an embedded method, i.e. a method where the model
                                      produces values for each feature that can be interpreted as
                                      feature importances. Which model is used is determined by
                                      `--embed-model`.

                          wrap:       Select using a wrapper method, i.e. a method which uses ("wraps")
                                      a specific model, and then optimizes the feature set via some
                                      alternation of model evaluations and feature-space search /
                                      navigation strategy.

                          none:       Do not select features using any model.
  --wrapper-select {step-up,step-down}

                        Wrapper-based feature selection method, i.e. method/optimizer to use to
                        search feature-set space during wrapper-based feature selection. Currently
                        only (recursive) step-down and step-up methods are supported, but future
                        versions may support random subset search, LIPO
                        (http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html,
                        https://arxiv.org/abs/1703.02628) and evolutionary algorithms such as
                        particle-swarm optimization and genetic algorithms (since feature selection
                        is just a black-box optimization problem where the search space is a set of
                        feature sets).

                        Model to use in wrapper-based feature selection. Available options are:

                          stepup:     Start with the empty feature set, and greedily add the feature
                                      that most improves prediction of the target variable. Also called
                                      forward feature selection.

                          stepdown:   Start with the full feature set, and remove the feature
                                      that most improves (or least decreases) prediction of the target
                                      variable. Also called backward / recursive feature slection or
                                      elimination.
  --wrapper-model {linear,lgbm}

                        Model to use during wrapper-based feature selection. Available options are:

                          linear:     For classification tasks, logistic regression, and for regression
                                      tasks, linear regression.

                          lgbm:       Use a LightGBM gradient-boosted decision tree model.
  --embed-model {lgbm,linear}

                        Model to use for embedded feature selection. Supported models are:

                          linear      Tuned SGDRegressor or SGDClassifier, in both cases with L1
                                      regularization.

                          lgbm:       LightGBM regressor or classifier, depending on task.
  --n-selection-tune-rounds N_SELECTION_TUNE_ROUNDS

                        If not the default of zero, the number of tuning rounds to do before evaluating
                        each feature set during wrapper-based feature selection.
  --nan {drop,mean,median,impute}

                        How to handle NaN values in non-categorical features. Categorical features
                        are handled by representing the NaN value as another category level (class),
                        i.e. one extra one-hot column is created for each categorical feature with a
                        NaN value.

                          drop:      Attempt to remove all non-categorical NaN values. Note this could
                                     remove all data if a lot of values are missing, which will cause
                                     errors.

                          mean:      Replace all NaN values with the feature mean value.

                          median:    Replace all NaN values with the feature median value.

                          impute:    Use scikit-learn experimental IterativeImputer to attempt to
                                     predictively fill NaN values based on other feature values. May
                                     be computationally demanding on larger datasets.

  --n-feat-filter N_FEAT_FILTER

                        Number or percentage (as a value in [0, 1]) of total features of any kind
                        (categorical or continuous) to select via filter-based feature selection.
                        Note only two of two of the three options:`--n-filter-total`, `--n-filter-cont`, and  `--n-filter-cat` may be specified at once, otherwise the `--n-filter-total` argument will be ignored.
  --n-feat-wrapper N_FEAT_WRAPPER

                        Number of features to select during wrapper-based feature selection. Note
                        that specifying values greater than e.g. 10-20 with slower algorithms (e.g.
                        LightGBM) and for data with a large number of features (e.g. over 50) can
                        easily result in compute times of many hours.
  --n-filter-cont N_FILTER_CONT

                        Number or percentage (as a value in [0, 1]) of continuous features to select
                        via filter-based feature selection. Note only two of two of the three options:`--n-filter-total`, `--n-filter-cont`, and  `--n-filter-cat` may be specified at once, otherwise the `--n-filter-total` argument will be ignored.
  --n-filter-cat N_FILTER_CAT

                        Number or percentage (as a value in [0, 1]) of categorical features to select
                        via filter-based feature selection. Note only two of two of the three options:`--n-filter-total`, `--n-filter-cont`, and  `--n-filter-cat` may be specified at once, otherwise the `--n-filter-total` argument will be ignored.
  --filter-method FILTER_METHOD
                        Method(s) to use for filter selection.

                        Method 'relief' is the most sophisticated and can detect interactions among pairs of features without dramatic compute costs (see https://www.sciencedirect.com/science/article/pii/S1532046418301400 or https://doi.org/10.1016/j.jbi.2018.07.014 for details and overview). This is in contrast to the 'assoc' and 'pred' methods (below) which do not detect any feature interactions.

                        Method 'assoc' is the fastest and is based on a measure of association between the feature and the target variable, where the measure of association is appropriate based on the cardinality (e.g. categorical vs. continuous) of the feature and target. However, because association need not imply generalizable predictive utility (and because the absence of an association does not imply an absence of predictive utility), it is possible that this method selects features that generalize poorly for prediction tasks.

                        Method 'pred' is based on the k-fold univariate predictive performance of each feature on the target variable, where the estimator is a lightly tuned sklearn.linear_model.SGDClassifier or sklearn.linear_model.SGDRegressor, depending on the task. Computing these univariate predictive performances is quite expensive, but because of the internal k-fold validation used, these predictive performance metrics directly asses the potential predictive utility of each feature.
  --filter-assoc-cont-classify {t,U,W,corr,cohen_d,AUROC,mut_info}

                        Type of association to use for selecting continuous features when the task or
                        target is classification / categorical.
  --filter-assoc-cat-classify {mut_info,H,cramer_v}

                        Type of association to use for selecting categorical features when the task or
                        target is classification / categorical.
  --filter-assoc-cont-regress {pearson_r,spearman_r,mut_info,F}

                        Type of association to use for selecting continuous features when the task or
                        target is regression / continuous.
  --filter-assoc-cat-regress {mut_info,H}

                        Type of association to use for selecting categorical features when the task or
                        target is regression / continuous.
  --filter-pred-regress {mae,msqe,mdae,r2,var-exp}

                        Regression score to use for filter-based selection of features. Options:

                          mae:       Mean Absolute Error

                          msqe:      Mean Squared Error

                          mdae:      Median Absolute Error

                          r2:        R-Squared

                          var-exp:   Percent Variance Explained


  --filter-pred-classify {acc,auroc,sens,spec}

                        Classification score to use for filter-based selection of features. Options:

                          acc:       Accuracy

                          auroc:     Area Under the ROC Curve

                          sens:      Sensitivity

                          spec:      Specificity


  --htune
                        If provided, use Optuna TPESampler to attempt to optimize classifier performance
                        prior to fitting and evaluating.

  --htune-val {holdout,kfold,k-fold,loocv,mc,none}

                        If hyperparamater tuning using `--htune` option, specifies the validation style
                        to use internally for each Optuna trial. Number of trials is specified by
                        `--htune-trials`, so the number of estimator fits interacts with that values
                        and `--htune-val-size`, which has a different meaning depending on validation
                        type chosen. Available options:

                          holdout:    Create a single holdout set and use to validate all Optuna trials.
                                      The float value in (0, 1) specified in `--htune-val-size` sets the
                                      percentage of samples to use for this holdout / test set.

                          kfold:      Use k-fold cross validation to compute performance for each Optuna
                                      trial. The value for `k` is specified by `--htune-val-size`.

                          loocv:      Use Leave-One-Out cross validation. `--htune-val-size` is ignored
                                      in this case.

                          mc:         Use "Monte-Carlo Cross Validation", e.g. generate multiple random
                                      train / test splits, i.e. equivalent to using all the splits
                                      generated by `sklearn.model_selection.StratifiedShuffleSplit`
                                      Currently generates 20 splits at 80%/20% train/test.
                                      `--htune-val-size` specifies the size of the test split in this
                                      case.

                          none:       Just fit the full data (e.g. validate on fitting / training data).
                                      Fast but highly biased. `--htune-val-size` is ignored in this case.

  --htune-val-size HTUNE_VAL_SIZE

                        See documentation for `--htune-val` (directly above if using `--help`). The
                        meaning of this argument depends on the choice of `--htune-val`:

                          holdout:    A float in (0, 1) specifies the proportion of samples from the
                                      input spreadsheet or table to set aside for evaluating during
                                      hyperparameter tuning.

                                      An integer specifies the number of samples to set aside for
                                      testing.

                          kfold:      An integer being one of 3, 5, 10, or 20 specifies the number of
                                      folds.

                          loocv:      Ignored.

                          mc:         A float in (0, 1) specifies the proportion of samples from the
                                      input spreadsheet or table to set aside for evaluating during
                                      each Monte-Carlo repeat evaluation.

                                      An integer specifies the number of samples to set aside for
                                      each repeat.

  --htune-trials HTUNE_TRIALS

                        Specifies number of trials in Optuna study, and for each estimator and feature
                        selection method. E.g. fitting two estimators using three feature selection
                        methods with `--htune-trials=100` will results in 2 x 3 x 100 = 600 trials. If
                        also using e.g. the default 3-fold validation for `--htune-val-sizes`, then the
                        total number of estimator fits from tuning will be 600 x 3.

                        NOTE: if you can afford it, it is strongly recommended to set this value to a
                        minimum of 100 (default), or 50 if your budget is constrained. Lower values
                        often will fail to find good fits, given the wide range on hyperparameters
                        needed to make this tool generally useful.

  --mc-repeats MC_REPEATS

                        Ignored unless using Monte-Carlo style cross validation via `--htune-val mc`.
                        Otherwise, specifies the number of random subsets of proportion
                        `--htune-val-size` on which to validate the data. Default 10.
  --test-val {holdout,kfold,k-fold,loocv,mc,none}

                        Specify which validation method to use for testing. Same behavour as for
                        `--htune-val` argument (see above).

  --test-val-sizes TEST_VAL_SIZES [TEST_VAL_SIZES ...]

                        Specify sizes of test validation sets. Same behavour as for `--htune-val-sizes`
                        argument (see above), except that multiple sizes can be specified. E.g.

                          python df-analyze.py <omitted> --test-val=kfold --test-val-sizes 5 10 20

                        will efficiently re-use the same trained model and evaluate 5-, 10-, and
                        20-fold k-fold estimates of performance, and include these all in the final
                        results.

                        The meaning of this argument depends on the choice of `--test-val`:

                          holdout:    A float in (0, 1) specifies the proportion of samples from the
                                      input spreadsheet or table to set aside for evaluating during
                                      hyperparameter tuning.

                                      An integer specifies the number of samples to set aside for
                                      testing.

                          kfold:      An integer being one of 3, 5, 10, or 20 specifies the number of
                                      folds.

                          loocv:      Ignored.

                          mc:         A float in (0, 1) specifies the proportion of samples from the
                                      input spreadsheet or table to set aside for evaluating during
                                      each Monte-Carlo repeat evaluation.

                                      An integer specifies the number of samples to set aside for
                                      each repeat.
  --outdir OUTDIR
                        Specifies location of all results, as well as cache files for slow
                        computations (e.g. stepwise feature selection). If unspecified, will attempt
                        to default to a number of common locations (/Users/derekberger, the
                        current working directory /Users/derekberger/Documents/Antigonish/df-analyze, or a temporary directory).

  --verbosity VERBOSITY

                        Controls amount of output to stdout and stderr. Options:

                          0:         ERROR: Minimal output and errors only
                          1:         INFO: Logging for each Optuna trial and various interim results.
                          2:         DEBUG: Currently unimplemented.

  --no-warn-explosion
                        If this flag is present, silence the warnings about large increases in the
                        number of features due to one-hot encoding of categoricals.

USAGE EXAMPLE (assumes you have run `poetry shell`):

    python df-analyze.py \
        --df="weather_data.json" \
        --target='temperature' \
        --mode=regress \
        --regressors=svm linear \
        --drop-nan=rows \
        --feat-clean=constant \
        --feat-select=pca pearson \
        --n-feat=5 \
        --htune \
        --test-val=kfold \
        --test-val-size=5 \
        --outdir='./results'

```