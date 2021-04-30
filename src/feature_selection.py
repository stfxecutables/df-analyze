# TODO
# - implement step-up feature selection
# - implement step-down feature selection
# - extract N largest PCA components as features
# - choose features with largest univariate separation in classes (d, AUC)
# - smarter methods
#   - remove correlated features
#   - remove constant features (no variance)

# NOTE: feature selection is PRIOR to hypertuning, but "what features are best" is of course
# contengent on the choice of regressor / classifier
# correct way to frame this is as an overall derivative-free optimization problem where the
# classifier choice is *just another hyperparameter*

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectPercentile,
    GenericUnivariateSelect,
    RFECV,
    SelectFromModel,
    SequentialFeatureSelector,
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LassoCV

# see https://scikit-learn.org/stable/modules/feature_selection.html
# for SVM-based feature selection, LASSO based feature selction, and RF-based feature-selection
# using SelectFromModel