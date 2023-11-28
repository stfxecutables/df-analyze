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

from sklearn.svm import SVC, SVR
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module

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


class SkorchMLP(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
