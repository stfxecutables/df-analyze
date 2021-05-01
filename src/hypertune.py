import numpy as np
import optuna

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union, Optional
from typing_extensions import Literal

from numpy import ndarray
from optuna import Trial
from optuna.study import Study
from optuna.trial import FrozenTrial
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier as DTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (
    train_test_split,
    cross_val_score as cvs,
    cross_validate as cv,
    LeaveOneOut,
    StratifiedShuffleSplit,
    BaseCrossValidator,
)

from src.cleaning import get_clean_data
from src.constants import VAL_SIZE, SEED

Estimator = Union[RF, SVC, DTreeClassifier, MLP, BaggingClassifier]
Classifier = Literal["rf", "svm", "dtree", "mlp", "bag"]
Kernel = Literal["rbf", "linear", "sigmoid"]
CVMethod = Union[int, float, Literal["loocv", "mc"]]
Splits = Iterable[Tuple[ndarray, ndarray]]

LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_LAYER_SIZES = [4, 8, 16, 32]
N_SPLITS = 5
TEST_SCORES = ["accuracy", "roc_auc"]


@dataclass(init=True, repr=True, eq=True, frozen=True)
class HtuneResult:
    classifier: Classifier
    n_trials: int
    cv_method: CVMethod
    val_acc: float = np.nan
    best_params: Dict = field(default_factory=dict)


def train_val_splits(
    df: DataFrame, val_size: float = VAL_SIZE
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Split data.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    train, val = train_test_split(
        df, test_size=val_size, random_state=SEED, shuffle=True, stratify=df.target
    )
    X_train = train.drop(columns="target")
    X_val = val.drop(columns="target")
    y_train = train.target.astype(int)
    y_val = val.target.astype(int)
    return X_train, X_val, y_train, y_val


def get_cv(y_train: DataFrame, cv_method: CVMethod) -> Union[int, Splits, BaseCrossValidator]:
    if isinstance(cv_method, int):
        return int(cv_method)
    if isinstance(cv_method, float):  # stratified holdout
        if cv_method <= 0 or cv_method >= 1:
            raise ValueError("Holdout test_size must be in (0, 1)")
        test_size = cv_method
        y = np.array(y_train).ravel()
        idx = np.arange(y.shape[0])
        return [
            train_test_split(idx, test_size=test_size, random_state=SEED, shuffle=True, stratify=y)
        ]
    cv_method = str(cv_method).lower()  # type: ignore
    if cv_method == "loocv":
        return LeaveOneOut()
    if cv_method == "mc":
        return StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=SEED)
    raise ValueError("Invalid `cv_method`")


def svm_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            kernel=trial.suggest_categorical("kernel", choices=["rbf"]),
            C=trial.suggest_loguniform("C", 1e-10, 1e10),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = SVC(**args)
        scores = cv(estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv)
        return float(np.mean(scores["test_score"]))

    return objective


def rf_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 5, 500),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            max_depth=trial.suggest_int("max_depth", 2, 50),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = RF(**args)
        scores = cv(estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv)
        return float(np.mean(scores["test_score"]))

    return objective


def dtree_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            splitter=trial.suggest_categorical("splitter", ["best", "random"]),
            max_depth=trial.suggest_int("max_depth", 2, 50),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = DTreeClassifier(**args)
        scores = cv(estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv)
        return float(np.mean(scores["test_score"]))

    return objective


def logistic_bagging_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 5, 23, 2),
            max_features=trial.suggest_uniform("max_features", 0, 1),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
        _cv = get_cv(y_train, cv_method)
        estimator = BaggingClassifier(
            base_estimator=LR(solver=LR_SOLVER), random_state=SEED, **args
        )
        scores = cv(estimator, X=X_train, y=y_train, scoring="accuracy", cv=_cv)
        return float(np.mean(scores["test_score"]))

    return objective


# needed for uniform interface only
def bagger(**kwargs: Any) -> Callable:
    return BaggingClassifier(base_estimator=LR(solver=LR_SOLVER), **kwargs)  # type: ignore


def mlp_layers(l1: int, l2: int, l3: int, l4: int, l5: int) -> Tuple[int, ...]:
    layers = [l1, l2, l3, l4, l5]
    return tuple([layer for layer in layers if layer > 0])


def mlp_args_from_params(params: Dict) -> Dict:
    d = {**params}
    l1 = d.pop("l1")
    l2 = d.pop("l2")
    l3 = d.pop("l3")
    l4 = d.pop("l4")
    l5 = d.pop("l5")
    d["hidden_layer_sizes"] = mlp_layers(l1, l2, l3, l4, l5)
    return d


def mlp_objective(
    X_train: DataFrame, y_train: DataFrame, cv_method: CVMethod = 5, val_size: float = VAL_SIZE
) -> Callable[[Trial], float]:
    if val_size > 0:
        df = X_train.copy()
        df["target"] = y_train
        X_train, X_test, y_train, y_test = train_val_splits(df, val_size)
    else:
        X_test, y_test = X_train, y_train

    def objective(trial: Trial) -> float:
        l1 = trial.suggest_categorical("l1", choices=MLP_LAYER_SIZES)
        l2 = trial.suggest_categorical("l2", choices=MLP_LAYER_SIZES)
        l3 = trial.suggest_categorical("l3", choices=MLP_LAYER_SIZES)
        l4 = trial.suggest_categorical("l4", choices=MLP_LAYER_SIZES)
        l5 = trial.suggest_categorical("l5", choices=MLP_LAYER_SIZES)
        args: Dict = dict(
            hidden_layer_sizes=mlp_layers(l1, l2, l3, l4, l5),
            activation=trial.suggest_categorical("activation", ["relu"]),
            solver=trial.suggest_categorical("solver", ["adam"]),
            # alpha=trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            alpha=trial.suggest_loguniform("alpha", 1e-6, 1e-1),
            batch_size=trial.suggest_categorical("batch_size", choices=[8, 16, 32]),
            learning_rate=trial.suggest_categorical(
                "learning_rate", choices=["constant", "adaptive"]
            ),
            learning_rate_init=trial.suggest_loguniform("learning_rate_init", 5e-5, 5e-1),
            max_iter=trial.suggest_categorical("max_iter", [100]),
            early_stopping=trial.suggest_categorical("early_stopping", [False]),
            validation_fraction=trial.suggest_categorical("validation_fraction", [0.1]),
        )
        mlp = MLP(**args)
        mlp.fit(X_train, y_train.astype(float))
        # y_pred = mlp.predict(X_test)
        # acc = accuracy_score(y_test, y_pred)
        # y_score = mlp.predict_proba(X_test)
        # auc = roc_auc_score(y_test, y_score)
        return float(mlp.score(X_test, y_test.astype(float)))

    return objective


def get_classifier_constructor(name: Classifier) -> Callable:
    CLASSIFIERS: Dict[str, Callable] = {
        "rf": RF,
        "svm": SVC,
        "dtree": DTreeClassifier,
        "mlp": MLP,
        "bag": bagger,
    }
    constructor = CLASSIFIERS[name]
    return constructor


def hypertune_classifier(
    classifier: Classifier,
    X_train: DataFrame,
    y_train: DataFrame,
    n_trials: int = 200,
    cv_method: CVMethod = 5,
    mlp_args: Dict = {},
) -> HtuneResult:
    OBJECTIVES: Dict[str, Callable] = {
        "rf": rf_objective(X_train, y_train, cv_method),
        "svm": svm_objective(X_train, y_train, cv_method),
        "dtree": dtree_objective(X_train, y_train, cv_method),
        "mlp": mlp_objective(X_train, y_train, cv_method, **mlp_args),
        "bag": logistic_bagging_objective(X_train, y_train, cv_method),
    }
    # HYPERTUNING
    objective = OBJECTIVES[classifier]
    constructor = get_classifier_constructor(name=classifier)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)

    # RESULTS
    print("Best params:", study.best_params)
    acc = np.round(study.best_value, 3)
    if classifier != "mlp":
        print(f"Best {cv_method}-Fold Accuracy on Training Set: {acc:0.3f}")
    else:
        print(f"Best Accuracy on MLP Validation Set: {acc:0.3f}")

    return HtuneResult(
        classifier=classifier,
        n_trials=n_trials,
        cv_method=cv_method,
        val_acc=study.best_value,
        best_params=study.best_params,
    )

    # TESTING
    args = mlp_args_from_params(study.best_params) if classifier == "mlp" else study.best_params
    estimator: Any = constructor(**args)
    X_test = X_train if X_test is None else X_test
    y_test = y_train if y_test is None else y_test
    if classifier == "mlp":
        print("Running final MLP test...")
    test_acc = estimator.fit(X_train, y_train).score(X_test, y_test)
    print("Test Accuracy:", test_acc)


def evaluate_hypertuned(
    htuned: HtuneResult,
    cv_method: CVMethod,
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: Optional[DataFrame] = None,
    y_test: Optional[DataFrame] = None,
) -> Any:
    classifier = htuned.classifier
    estimator = get_classifier_constructor(classifier)(**htuned.best_params)
    if (X_test is None) and (y_test is None):
        _cv = get_cv(y_train, cv_method)
        scores = cv(estimator, X=X_train, y=y_train, scoring=TEST_SCORES, cv=_cv)
        acc_mean = float(np.mean(scores["test_accuracy"]))
        auc_mean = float(np.mean(scores["test_roc_auc"]))
        acc_sd = float(np.std(scores["test_accuracy"], ddof=1))
        auc_sd = float(np.std(scores["test_roc_auc"], ddof=1))
        print("Hypertuned model performance:")
        print(f"  Accuracy: μ={np.round(acc_mean, 3):0.3f} ({np.round(acc_sd, 4):0.4f})")
        print(f"  AUC:      μ={np.round(auc_mean, 3):0.3f} ({np.round(auc_sd, 4):0.4f})")
    elif (X_test is not None) and (y_test is not None):
        y_pred = estimator.fit(X_train, y_train).predict(X_test)
        y_score = (
            estimator.decision_function(X_test)
            if classifier != "mlp"
            else estimator.predict_proba(X_test)
        )
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)
        percent = int(100 * float(cv_method))
        print(f"Hypertuned model performance on final {percent}% holdout set:")
        print(f"  Accuracy: μ={np.round(acc, 3):0.3f} ({np.round(acc, 4):0.4f})")
        print(f"  AUC:      μ={np.round(auc, 3):0.3f} ({np.round(auc, 4):0.4f})")

