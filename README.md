# Running the Script

To run the script do ...

# Analysis Plan

For each algorithm, there are n ways to hypertune:

1. some kind of holdout
  1. Create a final holdout test set (size=p in [0, 0.4]), optimize using params that give **best
     k-fold mean accuracy k-fold on the training set** (means each train set is really size (1 - p) x
     (1 - 1/k)) and **report final test accuracy** of classifier trained on full training set and
  2. Create train, val, test sets, optimize using params that give **best validation accuracy**
     training on training set, **report final test accuracy**
  3. maybe LOOCV here?, e.g. fit using LOOCV, evaluate using k-fold?
  4. Maybe a 20% Monte-Carlo about 20 times would be better here than either k-fold or LOOCV given
     the tiny dataset (or at least for e.g. MLP which is so sensitive to train set)
1. just overfit (overestimate accuracy)
  1. E.g. just run optimizer with k-fold on the whole data (no held-out split) and mean accs and
     mean AUCs are reported as is
  2. do above but with LOOCV instead

For each way to hypertune, there are f model-agnostic ways to select features:

1. univariate
  1. d
  2. AUC
  3. pearson
2. PCA
3. KernelPCA

But then there are the more annoying model-specific versions:

1. step-up
2. step-down

# Data Notes

- the dataframe is divided into one for schizophrenia and one for normals in variables titled
  HealthyMeasurements and SchizophreniaMeasurements: you probably want to concatenate the two and
  create a corresponding label array
- Minimal hypertuning is needed so that I can say to the reviewer that we've 'optimized' the
  performance
- 5-fold validation is fine
- Basic feature analysis is fine.
- No deep learning is fine and expected. Off-the-shelf algorithms are great here.
- Saving the comparative performance and limited feature analysis in a spreadsheet is great.
- Simple python script is great.

The first version should implement:

  - SVM
  - RF
  - decision tree
  - "least squares bagging"
  - basic ANN
  - no deep learning, but it would be great if you could run a basic MLP with hypertuning

I also need some off-the-shelf feature selection including stepwise regression. The
first round of effort included some basic feature selection:
  - running PCA and selecting the leading components to inform prediction
    - this didn't work very well on this dataset
  - ranking all the features by a basic univariate statistic (like cohen's d or AUC) and selecting the leading
    features
    - these were very basic feature selection baselines to compare against and didn't work that
      well and weren't very interesting
  - stepwise regression got the best results so I would need that included

We do need to provide a few (let's say 3 minimum) off-the-shelf feature selection techniques applied
to each of the 5 classifiers. If there is a problem with any of that, please let me know and I'm
open to suggestions for modification (I don't mind explaining to reviewers that we redid the
experiment, though it is generally best for us to have the next version be similar to previous). We
can probably strengthen the paper with some well selected off-the-shelf feature selection
algorithms.

Please report OA (??? - Overall Accuracy?) and AUC mean and std deviation. The manuscript is due in ~9 days.