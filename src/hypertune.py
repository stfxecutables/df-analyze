import numpy as np
import optuna

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from typing_extensions import Literal

from numpy import ndarray
from optuna import Trial
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier as DTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score as cv

from src.cleaning import get_clean_data
from src.constants import VAL_SIZE, SEED

Estimator = Union[RF, SVC, DTreeClassifier, MLP, BaggingClassifier]
Classifier = Literal["rf", "svm", "dtree", "mlp", "bag"]
Kernel = Literal["rbf", "linear", "sigmoid"]
CVMethod = Literal["cv", "loocv", "mc"]

LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_LAYER_SIZES = [4, 8, 16, 32]
N_SPLITS = 5


@dataclass(init=True, repr=True, eq=True, frozen=True)
class HtuneResult:
    classifier: str
    n_trials: int
    cv_method: CVMethod
    k: int
    mean_acc: ndarray = np.nan
    mean_auc: ndarray = np.nan
    test_acc: ndarray = np.nan
    test_auc: ndarray = np.nan


def train_val_splits(
    df: DataFrame, val_size: float = VAL_SIZE
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    train, val = train_test_split(
        df, test_size=val_size, random_state=SEED, shuffle=True, stratify=df.target
    )
    X_train = train.drop(columns="target")
    X_val = val.drop(columns="target")
    y_train = train.target.astype(int)
    y_val = val.target.astype(int)
    return X_train, X_val, y_train, y_val


def svm_objective(
    X_train: DataFrame, y_train: DataFrame, k: int = N_SPLITS
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            kernel=trial.suggest_categorical("kernel", choices=["rbf"]),
            C=trial.suggest_loguniform("C", 1e-10, 1e10),
        )
        svc = SVC(**args)
        return float(np.mean(cv(svc, X=X_train, y=y_train, scoring="accuracy", cv=k)))

    return objective


def rf_objective(
    X_train: DataFrame, y_train: DataFrame, k: int = N_SPLITS
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 5, 500),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            max_depth=trial.suggest_int("max_depth", 2, 50),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
        rf = RF(**args, n_jobs=2)
        return float(np.mean(cv(rf, X=X_train, y=y_train, scoring="accuracy", cv=k)))

    return objective


def dtree_objective(
    X_train: DataFrame, y_train: DataFrame, k: int = N_SPLITS
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            splitter=trial.suggest_categorical("splitter", ["best", "random"]),
            max_depth=trial.suggest_int("max_depth", 2, 50),
        )
        dt = DTreeClassifier(**args)
        return float(np.mean(cv(dt, X=X_train, y=y_train, scoring="accuracy", cv=k)))

    return objective


def logistic_bagging_objective(
    X_train: DataFrame, y_train: DataFrame, k: int = N_SPLITS
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 5, 23, 2),
            max_features=trial.suggest_uniform("max_features", 0, 1),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
        bc = BaggingClassifier(base_estimator=LR(solver=LR_SOLVER), random_state=SEED, **args)
        return float(np.mean(cv(bc, X=X_train, y=y_train, scoring="accuracy", cv=k)))

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
    X_train: DataFrame, y_train: DataFrame, val_size: float = VAL_SIZE, k: int = N_SPLITS
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
    X_test: Optional[DataFrame],
    y_test: Optional[DataFrame],
    n_trials: int = 200,
    cv_method: CVMethod = "cv",
    k: int = N_SPLITS,
    mlp_args: Dict = {},
) -> DataFrame:
    OBJECTIVES: Dict[str, Callable] = {
        "rf": rf_objective(X_train, y_train, k),
        "svm": svm_objective(X_train, y_train, k),
        "dtree": dtree_objective(X_train, y_train, k),
        "mlp": mlp_objective(X_train, y_train, k, **mlp_args),
        "bag": logistic_bagging_objective(X_train, y_train, k),
    }
    objective = OBJECTIVES[classifier]
    constructor = get_classifier_constructor(name=classifier)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    print("Best params:", study.best_params)
    if classifier != "mlp":
        print(f"Best {k}-Fold Accuracy on Training Set:", study.best_value)
    else:
        print("Best Accuracy on MLP Validation Set:", study.best_value)
    args = mlp_args_from_params(study.best_params) if classifier == "mlp" else study.best_params
    estimator: Any = constructor(**args)
    X_test = X_train if X_test is None else X_test
    y_test = y_train if y_test is None else y_test
    if classifier == "mlp":
        print("Running final MLP test...")
    test_acc = estimator.fit(X_train, y_train).score(X_test, y_test)
    print("Test Accuracy:", test_acc)
