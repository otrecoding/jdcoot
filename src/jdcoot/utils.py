import math
import numpy as np

def continuous_labels(source, prop_source, target, prop_target):

    size_s = source.Y.size
    size_t = target.Y.size
    
    l_source = np.full(size_s, True)
    l_target = np.full(size_t, True)
    
    l_source[np.random.choice(np.arange(size_s), math.ceil(prop_source * size_s), replace=False)] = False
    l_target[np.random.choice(np.arange(size_t), math.ceil(prop_target * size_t), replace=False)] = False

    return l_source, l_target


def rmse(ypred, ytest):
    n = ytest.size
    assert ypred.size == ytest.size
    return math.sqrt(sum((ypred - ytest) ** 2) / n)


def xcolumns(df):
    return [a for a in df.columns if 'X' in a]
