import os
import sys
sys.path.append(os.path.abspath('src'))

import numpy as np
import ot

from jdcoot.comp import comp_regression
from jdcoot.coot import cot_numpy
from jdcoot.utils import xcolumns, continuous_classifiers, continuous_labels, continuous_accuracy
from scenario import generate_data

def continuous_partial_coot( source, target, source_test, target_test) :

    prop_source = 0.1 
    prop_target = 0.1
    
    l_source, l_target = continuous_labels( source, prop_source, target, prop_target)

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    xtrain_source = x_source[~l_source, :]
    xtrain_target = x_target[~l_target, :]

    ytrain_source = source.Y.values.copy()
    ytrain_source[l_source] = np.nan
    ytrain_target = target.Y.values.copy()
    ytrain_target[l_target] = np.nan
    
    # cost matrix with ot dist
    def compute_cost_matrix(ys, yt):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1),
                    metric=comp_regression())
        return M
    
    M_lin = compute_cost_matrix(yt=ytrain_target, ys=ytrain_source[~l_source])

    Ts, Tv, cost = cot_numpy( X1=xtrain_source, X2=x_target,
        niter=100, C_lin=M_lin,
        algo='sinkhorn', reg=1,
        algo2='emd', verbose=False)
    
    zt_estimated = len(target.Y) * np.dot(Ts.T, source.Y.loc[~l_source])
    
    M_lin = compute_cost_matrix(yt=ytrain_source, ys=ytrain_target[~l_target])

    Ts, Tv, cost = cot_numpy( X1=xtrain_target, X2=x_source,
        niter=100, C_lin=M_lin,
        algo='sinkhorn', reg=1,
        algo2='emd', verbose=False)
    
    zs_estimated = len(source.Y) * np.dot(Ts.T, target.Y[~l_target])
    
    perf_pure = continuous_accuracy(target.Y.loc[l_target], zt_estimated[l_target],
               source.Y.loc[l_source], zs_estimated[l_source])
    
    clf_source, clf_target = continuous_classifiers(source, target)

    clf_target.fit(x_target, zt_estimated, batch_size=10, epochs=20, verbose=0)

    clf_source.fit(x_source, zs_estimated, batch_size=10, epochs=20, verbose=0)
    
    zt_test = clf_target.predict(target_test.loc[:, xcolumns(target_test)]).ravel()
    zs_test = clf_source.predict(source_test.loc[:, xcolumns(source_test)]).ravel()
    
    perf_test = continuous_accuracy(zt_test, target_test.Y, zs_test, source_test.Y)

    return perf_pure, perf_test
    


if __name__ == "__main__":

    data = generate_data()
    pure, test = continuous_partial_coot(*data)

    print(f"Pure Performance COOT : {pure} ")
    print(f"Test Performance COOT : {test} ")
