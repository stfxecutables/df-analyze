from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Mapping, Optional

from optuna import Trial

from pandas import DataFrame, Series  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)
from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.datasets import make_classification
from sklearn.svm import SVC, SVR
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.callbacks.lr_scheduler import CosineAnnealingLR
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    CrossEntropyLoss,
    Dropout,
    LazyLinear,
    LeakyReLU,
    Linear,
    Module,
    ModuleList,
    Sequential,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing_extensions import Literal

from src.models.base import DfAnalyzeModel

"""
See:

Arlind Kadra, Marius Lindauer, Frank Hutter, Josif Grabocka: “Well-tuned Simple
Nets Excel on Tabular Datasets”, 2021; [http://arxiv.org/abs/2106.11189
arXiv:2106.11189]. https://arxiv.org/pdf/2106.11189.pdf

Fixed Architecture and Optimization Hyperparameters

> In order to focus exclusively on investigating the effect of regularization we
> fix the neural architecture to a simple multilayer perceptron (MLP) and also
> fix some hyperparameters of the general training procedure. These fixed
> hyperparameter values, as specified in Table 4 of Appendix B.1, have been tuned
> for maximizing the performance of an unregularized neural network on our
> dataset collection (see Table 9 in Appendix D). We use a 9-layer feed-forward
> neural network with 512 units for each layer, a choice motivated by previous
> work [42].
>
> Moreover, we set a low learning rate of 10−3 after performing a grid search for
> finding the best value across datasets. We use AdamW [36], which implements
> decoupled weight decay, and cosine annealing with restarts [35] as a learning
> rate scheduler. Using a learning rate scheduler with restarts helps in our case
> because we keep a fixed initial learning rate. For the restarts, we use an
> initial budget of 15 epochs, with a budget multiplier of 2, following published
> practices [62]. Additionally, since our benchmark includes imbalanced datasets,
> we use a weighted version of categorical cross-entropy and balanced accuracy
> [4] as the evaluation metric.

Relevant Details from Table 4 of Fixed hparams:

Table 4: [Fixed Params]

Category                                 Value
......................................   .....
Cosine Annealing Iterations multiplier   2.0
Max. iterations                          15
Activation                               ReLU
Bias initialization                      Yes
Embeddings                               One-Hot encoding
Units in a layer                         512
Preprocessing Preprocessor               None
Batch size                               128
Imputation                               Median
Initialization method                    Default
Learning rate                            1e-3
Loss module                              Weighted Cross-Entropy
Normalization strategy                   Standardize
Optimizer                                AdamW
Scheduler                                Cosine-Annealing w Warm Restarts
Seed                                     11

Table 1 of The paper describes the regularization cocktails: we replicate just
a few, but also tune the LR (they did not find this to be meaningful, likely
given the large amount of regularization options used).

Batch-Norm                  {True, False}
Use Weight-Decay            {True, False}
Weight-Decay                [1e-5, 1e-1]
Use dropout                 {True, False}
Dropout rate                [0.0, 0.8]


"""

NETWORK_DEPTH = 8
BATCH_SIZE = 128


class SkorchMLP(Module):
    def __init__(
        self,
        width: int = 512,
        use_bn: bool = True,
        use_wd: bool = True,
        use_drop: bool = True,
        lr: float = 1e-3,
        wd: float = 1e-4,
        dropout: float = 0.4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hparams = dict(
            width=width,
            use_bn=use_bn,
            use_wd=use_wd,
            use_drop=use_drop,
            lr=lr,
            wd=wd,
            dropout=dropout,
        )
        self.num_classes = num_classes
        self.is_classification = num_classes >= 2
        self.out_channels = num_classes

        W = width
        D = dropout
        self.input = LazyLinear(out_features=width)
        self.layers = ModuleList()
        for i in range(NETWORK_DEPTH):
            self.layers.append(Linear(W, W, bias=not use_bn))
            if use_bn:
                self.layers.append(BatchNorm1d(W))
            if use_drop and i != NETWORK_DEPTH - 1:
                self.layers.append(Dropout(D))
            self.layers.append(LeakyReLU())

        self.backbone = Sequential(*self.layers)
        self.output = Linear(W, self.out_channels, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        x = self.backbone(x)
        x = self.output(x)
        return x


class CosineAnnealingLRWarm(LRScheduler):
    pass


def get_T0(
    X: ndarray,
    n_epochs: int,
    batch_size: int = BATCH_SIZE,
    val_split: Optional[Union[int, float]] = None,
) -> tuple[int, int]:
    N = len(X)
    if val_split is None:
        n_batches = N // batch_size
    elif isinstance(val_split, int):
        n_batches = (N - val_split) // batch_size
    elif isinstance(val_split, float) and (0.0 < val_split < 1.0):
        n_batches = int(N * val_split) // batch_size
    else:
        raise ValueError("Invalid `val_split` argument.")

    return n_epochs * n_batches, n_batches


if __name__ == "__main__":
    X, y = make_classification(2000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X_train = X[:1000]
    y_train = y[:1000]
    X_test = X[1000:]
    y_test = y[1000:]
    wd = 1e-4
    lr = 1e-3
    T0, n_batches = get_T0(X, n_epochs=8, val_split=0.2)
    sched = LRScheduler(
        policy=CosineAnnealingWarmRestarts,  # type: ignore
        T_0=T0,
        T_mult=2,
        eta_min=1e-7,
        verbose=True,
        step_every="epoch",
    )
    lrs = sched.simulate(steps=50 * n_batches, initial_lr=lr)
    epochs = [int(i / n_batches) for i, lr in enumerate(lrs)]
    plt.plot(epochs, lrs, color="black")
    plt.show()
    sys.exit()

    with catch_warnings():
        simplefilter("ignore", UserWarning)
        net = NeuralNetClassifier(
            module=SkorchMLP,
            module__width=512,
            module__use_bn=True,
            module__use_wd=True,
            module__use_drop=True,
            module__lr=lr,
            module__wd=wd,
            module__dropout=0.4,
            module__num_classes=2,
            criterion=CrossEntropyLoss,  # type: ignore
            criterion__weight=None,
            optimizer=AdamW,
            optimizer__weight_decay=wd,
            optimizer__lr=lr,
            callbacks=[
                sched,
                # Effectively min_epochs=20
                # EarlyStopping(patience=20, load_best=False),
            ],
            max_epochs=50,
            batch_size=BATCH_SIZE,
            # train_split=None,  # we do all tuning with 5-fold anyway...
            device="cpu",
        )
        net.fit(X_train, y_train)
        probs = net.predict_proba(X_test)
        print(probs[:10])
        preds = net.predict(X_test)
        print(preds[:10])
        print("Acc:", net.score(X_test, y_test))
        net.history
