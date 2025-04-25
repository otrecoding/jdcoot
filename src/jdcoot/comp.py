import numpy as np

def comp_(v=1e6):
    return lambda x, y: 0 if x == y or y == -1 else v


def comp_regression():
    def comp(x, y):
        if x == y or np.isnan(y):
            return 0
        else:
            return (x - y) ** 2  # MSE ou np.abs()

    return comp
