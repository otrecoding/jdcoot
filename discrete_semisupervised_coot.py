import os
import sys
sys.path.append(os.path.abspath('src'))

import numpy as np
import ot
from sklearn.preprocessing import OneHotEncoder as onehot

from jdcoot.comp import comp_
from jdcoot.coot import cot_numpy
from jdcoot.utils import xcolumns, discrete_classifier
from sklearn.model_selection import train_test_split


def discrete_semisupervised_coot( source, target, source_test, target_test):

    prop_target = 0.1
    
    source_levels = np.sort(np.unique(source.Z))
    target_levels = np.sort(np.unique(target.Z))

    nClass = len(np.union1d(source_levels, target_levels))
    categories=[np.arange(nClass)]

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    z_source = source.Z.values
    z_target = target.Z.values

    n_target = len(z_target)

    l_target_train, l_target_test = train_test_split(np.arange(n_target), 
                                                     test_size = prop_target, 
                                                     stratify = z_target)

    z_target_train = z_target.copy()
    z_target_train[l_target_test] = -1

    z_target_test = target.loc[l_target_test, 'Z'].values

    # cost matrix with ot dist
    def compute_cost_matrix(ys, yt, v=10000):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_(v))
        return M
    
    M_lin = compute_cost_matrix(yt=z_target_train, ys=z_source)
    
    Ts, Tv, cost = cot_numpy(X1=x_source,
                             X2=x_target,
                             niter=100, C_lin=M_lin,
                             algo='sinkhorn', reg=1,
                             algo2='emd', verbose=False)
    
    # target estimation
    enc = onehot(handle_unknown='ignore', sparse_output=False, categories=categories)

    zs_onehot = enc.fit_transform(z_source[:, np.newaxis])
    zt_onehot_estimated = n_target * np.dot(Ts.T, zs_onehot)
    zt_estimated = enc.inverse_transform(zt_onehot_estimated).ravel()

    perf_pure = np.mean(z_target_test == zt_estimated[l_target_test])
    
    clf = discrete_classifier(target, 'sigmoid', 'sigmoid', nClass)
    
    clf.fit(x_target, enc.fit_transform(zt_estimated[:, np.newaxis]), 
            batch_size=10, epochs=20, verbose=0) 

    z_test = clf.predict(target_test.loc[:, xcolumns(target)])
    z_test = enc.inverse_transform(z_test).ravel()
    perf_test = np.mean(z_test == target_test.Z) 

    return perf_pure, perf_test


if __name__ == "__main__":

    from scenario import generate_data

    data = generate_data()

    perf_pure, perf_test = discrete_semisupervised_coot(*data)
    
    print(f"Pure Performance : {perf_pure}")
    print(f"Test Performance : {perf_test}")
