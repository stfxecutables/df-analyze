from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path

import numpy as np
from numpy.random import Generator
from pandas import Series
from sklearn.metrics import recall_score

from src.df_analyze.scoring import npv, ppv, sensitivity, specificity
from src.df_analyze.testing.datasets import fake_data


def random_preds(rng: Generator, num_classes: int) -> tuple[Series, Series]:
    n_samples = rng.integers(100, 500)
    n_cat_features = rng.integers(1, 7)
    n_wrong_min = int(0.075 * n_samples)
    n_wrong_max = int(0.25 * n_samples)
    n_wrong = rng.integers(n_wrong_min, n_wrong_max)

    try:
        y_true = fake_data(
            mode="classify",
            N=n_samples,
            C=n_cat_features,
            num_classes=num_classes,
            rng=rng,
        )[2]
    except ValueError as e:
        raise RuntimeError("Problem generating data:") from e

    y_pred = y_true.copy()
    idx_wrong = rng.permutation(len(y_pred))[:n_wrong]
    y_pred[idx_wrong] = rng.choice(np.unique(y_true), size=n_wrong, replace=True)
    return y_true, y_pred


def test_sensitivity() -> None:
    """
    We are just doing this on the off chance our manual use of the confusion
    matrix somehow deviates from scikit-learn implementation...
    """

    # test binary case
    seeds = np.random.SeedSequence().spawn(100)
    n_success = 0
    for seed in seeds:
        rng = np.random.default_rng(seed=seed)
        y_true, y_pred = random_preds(rng, num_classes=2)

        df_sens = sensitivity(y_true=y_true, y_pred=y_pred)
        sk_sens = float(
            recall_score(y_true=y_true, y_pred=y_pred, zero_division=np.nan)  # type: ignore
        )

        try:
            np.testing.assert_approx_equal(actual=df_sens, desired=sk_sens, significant=5)
        except AssertionError as e:
            raise ValueError(
                f"Sensitivity does not match for seed: {seed}. Values matched for {n_success} random tests prior."
            ) from e
        n_success += 1


def test_sens_binary_sanity() -> None:
    y_true = Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    y0 = Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # tps=5 fps=0 tns=5 fns=0 ps=5
    y1 = Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # tps=5 fps=1 tns=4 fns=0 ps=6
    y2 = Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # tps=4 fps=0 tns=5 fns=1 ps=4
    y3 = Series([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])  # tps=4 fps=1 tns=5 fns=1 ps=5

    y_preds = [y0, y1, y2, y3]
    tps = [((y_true == 1) & (y == y_true)).sum() for y in y_preds]
    ps = [y_true.sum() for y in y_preds]
    expecteds = [tp / p for tp, p in zip(tps, ps)]

    # mats = [
    #     multilabel_confusion_matrix(y_true=y_true, y_pred=y, labels=[1])[0]
    #     for y in y_preds
    # ]
    # mat_tps = [mat[1, 1] for mat in mats]
    # mat_fns = [mat[1, 0] for mat in mats]
    # mat_ps = [tp + fn for tp, fn in zip(mat_tps, mat_fns)]

    for y_pred, expected in zip(y_preds, expecteds):
        sens = sensitivity(y_true=y_true, y_pred=y_pred)
        assert sens == expected, f"\n{y_pred}"


def test_spec_binary_sanity() -> None:
    y_true = Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    y0 = Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # tps=5 fps=0 tns=5 fns=0 ps=5
    y1 = Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # tps=5 fps=1 tns=4 fns=0 ps=6
    y2 = Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # tps=4 fps=0 tns=5 fns=1 ps=4
    y3 = Series([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])  # tps=4 fps=1 tns=5 fns=1 ps=5

    y_preds = [y0, y1, y2, y3]
    tns = [((y_true == 0) & (y == 0)).sum() for y in y_preds]
    fps = [((y_true == 0) & (y == 1)).sum() for y in y_preds]
    ns = [tn + fp for tn, fp in zip(tns, fps)]
    expecteds = [tn / n for tn, n in zip(tns, ns)]

    for y_pred, expected in zip(y_preds, expecteds):
        spec = specificity(y_true=y_true, y_pred=y_pred)
        assert spec == expected, f"\n{y_pred}"


def test_ppv_binary_sanity() -> None:
    y_true = Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    y0 = Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # tps=5 fps=0 tns=5 fns=0 ps=5
    y1 = Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # tps=5 fps=1 tns=4 fns=0 ps=6
    y2 = Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # tps=4 fps=0 tns=5 fns=1 ps=4
    y3 = Series([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])  # tps=4 fps=1 tns=5 fns=1 ps=5

    y_preds = [y0, y1, y2, y3]
    tps = [((y_true == 1) & (y == 1)).sum() for y in y_preds]
    fps = [((y_true == 0) & (y == 1)).sum() for y in y_preds]
    denoms = [tp + fp for tp, fp in zip(tps, fps)]
    expecteds = [tp / denom for tp, denom in zip(tps, denoms)]

    for y_pred, expected in zip(y_preds, expecteds):
        value = ppv(y_true=y_true, y_pred=y_pred)
        assert value == expected, f"\n{y_pred}"


def test_npv_binary_sanity() -> None:
    y_true = Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    y0 = Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # tps=5 fps=0 tns=5 fns=0 ps=5
    y1 = Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # tps=5 fps=1 tns=4 fns=0 ps=6
    y2 = Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # tps=4 fps=0 tns=5 fns=1 ps=4
    y3 = Series([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])  # tps=4 fps=1 tns=5 fns=1 ps=5

    y_preds = [y0, y1, y2, y3]
    tns = [((y_true == 0) & (y == 0)).sum() for y in y_preds]
    fns = [((y_true == 1) & (y == 0)).sum() for y in y_preds]
    denoms = [tn + fn for tn, fn in zip(tns, fns)]
    expecteds = [tp / denom for tp, denom in zip(tns, denoms)]

    for y_pred, expected in zip(y_preds, expecteds):
        value = npv(y_true=y_true, y_pred=y_pred)
        assert value == expected, f"\n{y_pred}"
