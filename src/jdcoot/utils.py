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

def discrete_labels(source, prop_source, target, prop_target):

    source_levels = np.unique(source.Z)
    target_levels = np.unique(target.Z)

    if len(source_levels) > 2:
    
        del_idx = []
        for k in source_levels:
            if sum(source.Z == k) < 0.01 * len(source.Z):
                del_idx.append()
        source = source.loc[~np.in1d(source.Z, del_idx), :].reset_index(drop=True)
    
    if len(target_levels) > 2:
        del_idx = []
        for k in target_levels:
            if sum(target.Z == k) < 0.01 * len(target.Z):
                del_idx.append(k)
        target = target.loc[~np.in1d(target.Z, del_idx), :].reset_index(drop=True)
    
    # number of observations kept referenced by the min number of available observation per class
    S_nPerClass = min(np.unique(source.Z, return_counts=True)[1])
    T_nPerClass = min(np.unique(target.Z, return_counts=True)[1])
    
    z_kept_source = np.array([]).astype(int)
    for lab in source_levels:
        z_kept_source = np.append(z_kept_source,
                                  np.random.choice(np.where(source.Z == lab)[0], S_nPerClass, replace=False))
    
    source = source.loc[z_kept_source, :].reset_index(drop=True)
    
    z_kept_target = np.array([]).astype(int)
    for lab in target_levels:
        z_kept_target = np.append(z_kept_target,
                                  np.random.choice(np.where(target.Z == lab)[0], T_nPerClass, replace=False))
    target = target.loc[z_kept_target, :].reset_index(drop=True)
    
    source_labels = np.array([]).astype(int)
    for lab in source_levels:
        a = np.random.choice(np.where(source.Z == lab)[0], 
                             math.ceil(prop_source * sum(source.Z == lab)), 
                             replace=False)

        source_labels = np.append(source_labels, a)
    
    target_labels = np.array([]).astype(int)
    for lab in target_levels:
        b = np.random.choice(np.where(target.Z == lab)[0], 
                             math.ceil(prop_target * sum(target.Z == lab)), 
                             replace=False)

        target_labels = np.append(target_labels, b)
    
    return source_labels, target_labels


def rmse(ypred, ytest):
    n = ytest.size
    assert ypred.size == ytest.size
    return math.sqrt(sum((ypred - ytest) ** 2) / n)


def xcolumns(df):
    return [a for a in df.columns if 'X' in a]
