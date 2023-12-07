FLOAT_INFO = (
    "Found string-valued features that seem to mostly contain continuous values. "
    "This likely means these columns are formatted oddly (e.g. all values might "
    "be quoted) or that there is a typo or strange value in the column. "
    "df-analyze will automatically treat these columns as continuous to prevent "
    "wasted compute resources that would be incurred with encoding them as "
    "categorical, however, this might be an error.\n\n"
    "Columns that may be "
    "continuous:\n{info} "
)

ORD_INFO = (
    "Found string-valued features that could contain ordinal values (i.e. "
    "non-categorical integer values) NOT specified with the `--ordinals` option. "
    "Check if these features are ordinal or categorical, and then explicitly "
    "pass them to either the `--categoricals` or `--ordinals` options when "
    "configuring df-analyze.\n\n "
    "Columns that may be ordinal:\n{info}"
)

ID_INFO = (
    "Found string-valued features likely containing identifiers (i.e. unique "
    "string or integer values that are assigned arbitrarily), or which have more "
    "levels (unique values) than one half of the number of (non-NaN) samples in "
    "the data. This is most likely an identifier or junk feature which has no "
    "predictive value, and thus should be removed from the data. Even if the "
    "feature is not an identifer, with such a large number of levels, then a "
    "test set (either in k-fold, or holdout) will, on average, mostly contain "
    "values that were never seen during training. Thus, these features are "
    "essentially undersampled, and too sparse to be useful given the amount of "
    "data. Encoding this many values also massively increases compute costs for "
    "little gain. We thus REMOVE these features. However, but you should inspect "
    "these features yourself and ensure these features are not better described "
    "as either ordinal or continuous. If so, specify them using the `--ordinals` "
    "or `--continous` options to df-analyze.\n\n "
    "Columns that likely are identifiers:\n{info}"
)

TIME_INFO = (
    "Found string-valued features that appear to be datetime data or time "
    "differences. Datetime data cannot currently be handled by `df-analyze` (or "
    "most AutoML or or most automated predictive approaches) due to special data "
    "preprocessing needs (e.g. Fourier features), splitting (e.g. time-based "
    "cross-validation, forecasting, hindcasting) and in the models used (ARIMA, "
    "VAR, etc.).\n\n "
    "Columns that are likely timestamps:\n{info}\n\n"
    ""
    "To remove this warning, DELETE these columns from your data, or manually "
    "edit or convert the column values so they are interpretable as a "
    "categorical or continuous variable reflecting some kind of cyclicality that "
    "may be relevant to the predictive task. E.g. a variable that stores the "
    "time a sample was recorded might be converted to categorical variable "
    "like:\n\n "
    ""
    "  - morning, afternoon, evening, night (for daily data)\n"
    "  - day of the week (for monthly data)\n"
    "  - month (for yearly data)\n"
    "  - season (for multi-year data)\n\n"
    ""
    "Or to continuous / ordinal cyclical versions of these, like:\n\n"
    ""
    "  - values from 0 to 23 for ordinal representation of day hour\n"
    "  - values from 0.0 to 23.99 for continuous version of above\n"
    "  - values from 0 to 7 for day of the week, 0 to 365 for year\n\n"
    ""
    "It is possible to convert a single time feature into all of the above, i.e. "
    "to expand the feature into multiple cyclical features. This would be a "
    "variant of Fourier features (see e.g.\n\n"
    ""
    "\tTian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong "
    'Jin:\n\t"FEDformer: Frequency Enhanced Decomposed Transformer\n\tfor '
    'Long-term Series Forecasting", 2022;\n\t'
    "[http://arxiv.org/abs/2201.12740].\n\n"
    ""
    "`df-analyze` may in the future attempt to automate this via `sktime` "
    "(https://www.sktime.net)."
)

CAT_INFO = (
    "Found string-valued features not specified with `--categoricals` argument, "
    "and that look categorical. These will be one-hot encoded to allow use in "
    "subsequent analyses. To silence this warning, specify them as categoricals "
    "or ordinals manually either via the CLI or in the spreadsheet file header, "
    "using the `--categoricals` and/or `--ordinals` option.\n\n"
    "Features that look categorical not specified by `--categoricals`:\n{info}"
)

CONST_INFO = (
    "Found features that are constant (i.e. all values) are NaN or all values "
    "are the same non-NaN value). These contain no information and will be "
    "removed automatically.\n\n "
    "Features that are constant:\n{info}"
)

BIG_INFO = (
    "Found string-valued features with more than 50 unique levels. Unless you "
    "have a large number of samples, or if these features have a highly "
    "imbalanced / skewed distribution, then they will cause sparseness after "
    "one-hot encoding. This is generally not beneficial to most algorithms. You "
    "should inspect these features and think if it makes sense if they would be "
    "predictively useful for the given target. If they are unlikely to be "
    "useful, consider removing them from the data. This will also likely "
    "considerably improve `df-analyze` predictive performance and reduce compute "
    "times. However, we do NOT remove these features automatically.\n\n"
    "String-valued features with over 50 levels: {info}"
)
NYAN_INFO = (
    "Found features that are constant if dropping NaN values. Whether or not "
    "these features have predictive value will depend on if data is missing "
    "completely at random (MCAR - no information), are correlated with other "
    "data features that have predictive value, or if missingness is related to "
    "the prediction target. If you have a lot of features, these would be good "
    "candidates to consider for removal."
    "\n\n"
    "Features that are constant when dropping NaNs: {info}"
)
