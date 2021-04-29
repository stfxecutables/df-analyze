# Running the Script

To run do ...

# Data Notes


Let's start with a very basic script:

- I know the form of the data
- we do zero/minimal hypertuning and feature analysis with only 3-fold or 5-fold validation
- the focus is on a basic assessment of the data with only standard ML / scikit-learn algorithms (no deep learning)
- only key results are saved as a DataFrame (e.g. in .csv, .json, or spreadsheet)
- the desired usage is a simple python script, and not a nice/clean API or library

Please find attached the form of the data, which can be modified to your liking if you want to
simplify things (extract then you've got a .mat file). Note the dataframe is divided into one for
schizophrenia and one for normals in variables titled HealthyMeasurements and
SchizophreniaMeasurements, so you probably want to concatenate the two and create a corresponding
label array. Minimal hypertuning is needed so that I can say to the reviewer that we've 'optimized'
the performance (they are whining about this). 5-fold validation is fine. Basic feature analysis is
fine. No deep learning is fine and expected. Off-the-shelf algorithms are great here. Saving the
comparative performance and limited feature analysis in a spreadsheet is great. Simple python script
is great.

I would like this first default version to: implement the SVM, decision tree, RF, least squares
bagging and a basic ANN (no deep learning, but it would be great if you could run a basic MLP with
hypertuning). I also need some off-the-shelf feature selection including stepwise regression. The
first round of effort included some basic feature selection (running PCA and selecting the leading
components to inform prediction—this didn't work very well on this dataset, and ranking all the
features by a basic univariate statistic (like cohen's d or AUC) and selecting the leading
features—these were very basic feature selection baselines to compare against and didn't work that
well and weren't very interesting (but the stepwise regression got the best results so I would need
that included). That said we do need to provide a few (let's say 3 minimum) off-the-shelf feature
selection techniques applied to each of the 5 classifiers. If there is a problem with any of that,
please let me know and I'm open to suggestions for modification (I don't mind explaining to
reviewers that we redid the experiment, though it is generally best for us to have the next version
be similar to previous). We can probably strengthen the paper with some well selected off-the-shelf
feature selection algorithms.

Please report OA and AUC mean and std deviation.

Thanks for your help on this - the manuscript is due in like 9 days, but I can probably get another
extension if this turns out to take longer than 2 days.

Cheers,

Jacob