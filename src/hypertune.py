import numpy as np
import optuna

from typing import Any, Callable, Dict, List, Tuple, Union
from typing_extensions import Literal

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

LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_LAYER_SIZES = [4, 8, 16, 32]

def svm_objective(
    X_train: DataFrame, y_train: DataFrame, kernels: List[Kernel]=["rbf", "linear"]
) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            kernel=trial.suggest_categorical("kernel", choices=kernels),
            C=trial.suggest_loguniform("C", 1e-10, 1e10),
        )
        svc = SVC(**args)
        return float(np.mean(cv(svc, X=X_train, y=y_train, scoring="accuracy", cv=3)))

    return objective


def rf_objective(X_train: DataFrame, y_train: DataFrame) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 5, 500),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            max_depth=trial.suggest_int("max_depth", 2, 50),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
        rf = RF(**args, n_jobs=2)
        return float(np.mean(cv(rf, X=X_train, y=y_train, scoring="accuracy", cv=3)))

    return objective


def dtree_objective(X_train: DataFrame, y_train: DataFrame) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            splitter=trial.suggest_categorical("splitter", ["best", "random"]),
            max_depth=trial.suggest_int("max_depth", 2, 50),
        )
        dt = DTreeClassifier(**args)
        return float(np.mean(cv(dt, X=X_train, y=y_train, scoring="accuracy", cv=3)))

    return objective


def logistic_bagging_objective(X_train: DataFrame, y_train: DataFrame) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        args: Dict = dict(
            n_estimators=trial.suggest_int("n_estimators", 5, 23, 2),
            max_features=trial.suggest_uniform("max_features", 0, 1),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
        bc = BaggingClassifier(base_estimator=LR(solver=LR_SOLVER), random_state=SEED, **args)
        return float(np.mean(cv(bc, X=X_train, y=y_train, scoring="accuracy", cv=3)))

    return objective


# needed for uniform intergace only
def bagger(**kwargs) -> Any:
    return BaggingClassifier(base_estimator=LR(solver=LR_SOLVER), **kwargs)

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


def mlp_objective(X_train: DataFrame, y_train: DataFrame) -> Callable[[Trial], float]:
    # df = X_train.copy()
    # df["target"] = y_train
    # X_train, X_test, y_train, y_test = train_val_splits(df)
    def objective(trial: Trial) -> float:
        l1 = trial.suggest_categorical("l1", choices=MLP_LAYER_SIZES)
        l2 = trial.suggest_categorical("l2", choices=MLP_LAYER_SIZES)
        l3 = trial.suggest_categorical("l3", choices=MLP_LAYER_SIZES)
        l4 = trial.suggest_categorical("l4", choices=MLP_LAYER_SIZES)
        l5 = trial.suggest_categorical("l5", choices=MLP_LAYER_SIZES)
        args: Dict = dict(
            hidden_layer_sizes=mlp_layers(l1, l2, l3, l4, l5),
            activation="relu",
            solver="adam",
            # alpha=trial.suggest_loguniform("alpha", 1e-8, 1e-1),
            alpha=trial.suggest_loguniform("alpha", 1e-6, 1e-1),
            batch_size=trial.suggest_categorical("batch_size", choices=[8, 16, 32, 64]),
            learning_rate=trial.suggest_categorical("learning_rate", choices=["constant", "adaptive"]),
            learning_rate_init=trial.suggest_loguniform("learning_rate_init", 5e-5, 5e-1),
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.3,
        )
        mlp = MLP(**args)
        mlp.fit(X_train, y_train.astype(float))
        # return float(mlp.score(X_test, y_test.astype(float)))
        return float(mlp.score(X_train, y_train.astype(float)))

    return objective


def hypertune_classifier(
    classifier: Classifier,
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: DataFrame,
    y_test: DataFrame,
    n_trials: int = 100,
    objective_args: Dict = {},
) -> DataFrame:
    OBJECTIVES: Dict[str, Callable] = {
        "rf": rf_objective(X_train, y_train),
        "svm": svm_objective(X_train, y_train, **objective_args),
        "dtree": dtree_objective(X_train, y_train),
        "mlp": mlp_objective(X_train, y_train),
        "bag": logistic_bagging_objective(X_train, y_train),
    }
    CLASSIFIERS: Dict[str, Callable[[Any], Any]] = {
        "rf": RF,
        "svm": SVC,
        "dtree": DTreeClassifier,
        "mlp": MLP,
        "bag": bagger,
    }
    objective = OBJECTIVES[classifier]
    constructor = CLASSIFIERS[classifier]
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(objective, n_trials=n_trials)
    print("Best params:", study.best_params)
    if classifier != "mlp":
        print("Best 3-Fold Accuracy on Training Set:", study.best_value)
    else:
        print("Best Accuracy on MLP Validation Set:", study.best_value)
    args = mlp_args_from_params(study.best_params) if classifier == "mlp" else study.best_params
    estimator: Any = constructor(**args)
    if classifier == "mlp":
        print("Running final MLP test...")
    test_acc = estimator.fit(X_train, y_train).score(X_test, y_test)
    print("Test Accuracy:", test_acc)


def train_val_splits(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    train, val = train_test_split(
        df, test_size=VAL_SIZE, random_state=SEED, shuffle=True, stratify=df.target
    )
    X_train = train.drop(columns="target")
    X_val = val.drop(columns="target")
    y_train = train.target.astype(int)
    y_val = val.target.astype(int)
    return X_train, X_val, y_train, y_val


# need a validation set for hypertuning and also a training set...
# but given number of samples this is not very viable
# Do subject-level k-fold for hparam tuning? Or keep a final test set to evaluate?
if __name__ == "__main__":

    df = get_clean_data()
    X_train, X_val, y_train, y_val = train_val_splits(df)
