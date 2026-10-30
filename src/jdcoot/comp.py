"""Pairwise label costs for optimal transport."""

import numpy as np


def comp_(v=1e6):
    """Return a mismatch cost that ignores unlabelled target observations."""
    return lambda x, y: 0 if x == y or y == -1 else v


def comp_regression():
    """Return squared error, ignoring missing target values."""
    def comp(x, y):
        if x == y or np.isnan(y):
            return 0
        return (x - y) ** 2

    return comp
