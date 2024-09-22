import os
from itertools import islice
from typing import Optional


def batched(iterable, n):
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := list(tuple(islice(iterator, n))):
        yield batch


def get_n_test_samples(n_samples: Optional[int] = None) -> int:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    if n_samples is None:
        N = 256 if ON_CLUSTER else 32
    else:
        N = n_samples
    return N
