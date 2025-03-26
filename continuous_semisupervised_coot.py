import os
import sys
sys.path.append(os.path.abspath('src'))

import numpy as np
import ot

from jdcoot.coot import cot_numpy
from jdcoot.comp import comp_regression
from jdcoot.utils import xcolumns, continuous_classifier, continuous_accuracy
from sklearn.model_selection import train_test_split


def continuous_semisupervised_coot(source, target, source_test, target_test):

    prop_target = 0.1 # Labelled_Proportion_target

    y_source = source.Y.values
    y_target = target.Y.values

    n_target = len(y_target)
    
    l_target_train, l_target_test = train_test_split(np.arange(n_target), test_size = prop_target)

    ytrain_source = source.Y.values.copy()
    ytrain_target = target.Y.values.copy()
    ytrain_target[l_target_test] = np.nan
    
    def compute_cost_matrix(ys, yt):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1),
                    metric=comp_regression())  # comp_reg ? ou comp_
        return M
    
    M_lin = compute_cost_matrix(yt=ytrain_target, ys=ytrain_source)
    
    xtrain_source = source.loc[:, xcolumns(source)].values
    xtrain_target = target.loc[:, xcolumns(target)].values

    Ts, Tv, cost = cot_numpy(X1=xtrain_source,
                             X2=xtrain_target,
                             niter=100, C_lin=M_lin,
                             algo='sinkhorn', reg=1,
                             algo2='emd', verbose=False)
    
    y_target_pred = n_target * np.dot(Ts.T, y_source)
    
    perf_pure = continuous_accuracy(y_target_pred[l_target_test], y_target[l_target_test])
    
    clf_target = continuous_classifier(target)

    clf_target.fit(xtrain_target, y_target_pred, batch_size=10, epochs=20, verbose=0)  

    y_target_test_pred = clf_target.predict(target_test.loc[:, xcolumns(target_test)]).ravel()

    perf_test = continuous_accuracy(y_target_test_pred, target_test.Y)

    return perf_pure, perf_test

if __name__ == '__main__':

    from scenario import generate_data

    data = generate_data()
    
    perf_pure, perf_test = continuous_semisupervised_coot( *data )

    print(f"Pure Performance COOT : {perf_pure}")
    print(f"Test Performance COOT : {perf_test}")
