from src.options import get_options


if __name__ == "__main__":
    # TODO:
    # get arguments
    # run analyses based on args
    options = get_options()
    classifiers = options.classifiers

    n_features = [10, 50, 100]
    if stepup and classifier == "mlp":
        n_features = [10, 50]  # 100 features will probably take about 30 hours

    ARG_OPTIONS = dict(
        classifier=[classifier],
        feature_selection=["step-up"] if stepup else SELECTIONS,
        n_features=n_features,
        htune_validation=[5],
    )
    ARGS = list(ParameterGrid(ARG_OPTIONS))
    run_analysis(ARGS, classifier, stepup)
