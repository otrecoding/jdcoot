import os
import sys
sys.path.append(os.path.abspath('src'))

import math
import numpy as np
import pandas as pd
import ot
import tf_keras
from tf_keras.layers import Dense

from jdcoot.comp import comp_regression
from jdcoot.coot import cot_numpy
from jdcoot.data_scenario import DataScenario, DataScenarioTest
import jdcoot
from jdcoot.utils import labelled_indexes, rmse, xcolumns

    
def continuous_unsupervised_coot( source, target, source_test, target_test) :

    
    prop_S = 1
    prop_T = 0
    alpha = 0.3
    
    l_source, l_target = labelled_indexes( source, prop_S, target, prop_T)

    x_source = target.loc[:, xcolumns(target)]
    x_target = target.loc[:, xcolumns(target)]
    
    ytrain_source = source.Y.values.copy()
    ytrain_source[l_source] = np.nan
    ytrain_target = target.Y.values.copy()
    ytrain_target[l_target] = np.nan
    
    def compute_cost_matrix(ys, yt):
        return ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_regression())  
    
    M_lin = compute_cost_matrix(yt=ytrain_target, ys=ytrain_source)

    Ts, Tv, cost = cot_numpy(X1=x_source, X2=x_target, niter=100, 
                             C_lin=M_lin,
                             algo='sinkhorn', reg=1,
                             algo2='emd', verbose=False)
    
    ytrue = target.Y
    ypred = target.Y.size * np.dot(Ts.T, source.Y)
    
    perf_coot = rmse(ypred[l_target], ytrue[l_target])
    
    def clf_seq(shape, nClass):
        model = tf_keras.Sequential([
            Dense(units=128, input_shape=shape, activation='linear'),
            Dense(units=nClass, activation='linear')])
        return model
    
    fe_size = len(xcolumns(target))
    shape = (fe_size,)
    loss = 'MeanSquaredError'
    clf = clf_seq(shape, nClass=1)
    clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    clf.fit(x_target, ypred, batch_size=10, epochs=20, verbose=0) 
    ytest = clf.predict(target_test.loc[:, xcolumns(target)])[:, 0]
    perf_coot_test = rmse(ytest, target_test.Y)

    return perf_coot, perf_coot_test


if __name__ == "__main__":

    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    reference_scenario = DataScenario()
    test_scenario = DataScenarioTest()
    
    source, target = reference_scenario.generate(INDEX_GENERATION)
    source_test, target_test = test_scenario.generate(INDEX_GENERATION)
    perf_coot, perf_coot_test = continuous_unsupervised_coot( source, target, source_test, target_test)
    
    print(f"Pure Performance COOT : {perf_coot}")
    print(f"Test Performance COOT : {perf_coot_test}")
