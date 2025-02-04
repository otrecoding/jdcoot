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
from jdcoot.comp import comp_
from jdcoot.data_scenario import DataScenario, DataScenarioTest
import jdcoot
from jdcoot.utils import *

def continuous_partial_coot( source, target, source_test, target_test) :

    S = source
    T = target
    S_test = source_test
    T_test = target_test
    
    S_test = S_test.loc[:, S.columns]
    T_test = T_test.loc[:, T.columns]
    
    prop_S = 0.1 # Labelled_Proportion_Source
    prop_T = 0.1 # Labelled_Proportion_Target
    alpha = 2.425
    
    l_source, l_target = labelled_indexes( S, prop_S, T, prop_T)

    x_source = S.loc[:, xcolumns(S)].values
    x_target = T.loc[:, xcolumns(T)].values

    xtrain_source = x_source[~l_source, :]
    xtrain_target = x_target[~l_target, :]

    ytrain_source = S.Y.values.copy()
    ytrain_source[l_source] = np.nan
    ytrain_target = T.Y.values.copy()
    ytrain_target[l_target] = np.nan
    
    # cost matrix with ot dist
    def compute_cost_matrix(ys, yt):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1),
                    metric=comp_regression())  # comp_reg ? ou comp_
        return M
    
    M_lin = compute_cost_matrix(yt=ytrain_target, ys=ytrain_source[~l_source])

    Ts, Tv, cost = cot_numpy( X1=xtrain_source, X2=x_target,
        niter=100, C_lin=M_lin,
        algo='sinkhorn', reg=1,
        algo2='emd', verbose=False)
    
    zt_estimated = len(T.Y) * np.dot(Ts.T, S.Y.loc[~l_source])
    
    M_lin = compute_cost_matrix(yt=ytrain_source, ys=ytrain_target[~l_target])

    Ts, Tv, cost = cot_numpy( X1=xtrain_target, X2=x_source,
        niter=100, C_lin=M_lin,
        algo='sinkhorn', reg=1,
        algo2='emd', verbose=False)
    
    zs_estimated = len(S.Y) * np.dot(Ts.T, T.Y[~l_target])
    
    perf_coot = (sum((T.Y.loc[l_target] - zt_estimated[l_target]) ** 2) + sum((S.Y.loc[l_source] - zs_estimated[l_source]) ** 2)) / ( sum(l_target)+ sum(l_source) )
    
    #############train classifier and evaluate the performance on test

    def clf_seq(shape, nClass):
        model = tf_keras.Sequential([
            Dense(units=128, input_shape=shape, activation='linear'),
            Dense(units=nClass, activation='linear')])
        return model
    
    fe_sizeT = len(xcolumns(T))  # Nombre de variables de Target
    shapeT = (fe_sizeT,)
    loss = 'MeanSquaredError'
    clfT = clf_seq(shapeT, nClass=1)
    clfT.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    fe_sizeS = len(xcolumns(S))  # Nombre de variables de Target
    shapeS = (fe_sizeS,)
    loss = 'MeanSquaredError'
    clfS = clf_seq(shapeS, nClass=1)
    clfS.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    clfT.fit(x_target, zt_estimated, batch_size=10, epochs=20,
             verbose=0)  # we train the classifier with target data estimated

    clfS.fit(x_source, zs_estimated, batch_size=10, epochs=20,
             verbose=0)  # we train the classifier with target data estimated
    
    zt_test = clfT.predict(T_test.loc[:, xcolumns(T_test)])[:, 0]
    zs_test = clfS.predict(S_test.loc[:, xcolumns(S_test)])[:, 0]
    
    perf_coot_test = (sum((zt_test - T_test.Y) ** 2) + sum((zs_test - S_test.Y) ** 2)) / (len(zs_test) + len(zt_test))

    return perf_coot, perf_coot_test
    


if __name__ == "__main__":

    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    reference_scenario = DataScenario()
    test_scenario = DataScenarioTest()
    
    source, target = reference_scenario.generate(INDEX_GENERATION)
    source.to_csv("source.csv", index = False)
    target.to_csv("target.csv", index = False)
    
    source_test, target_test = test_scenario.generate(INDEX_GENERATION)
    source_test.to_csv("source_test.csv", index = False)
    target_test.to_csv("target_test.csv", index = False)

    perf_coot, perf_coot_test = continuous_partial_coot( source, target, source_test, target_test)

    print("Pure Performance COOT : {} ".format(perf_coot))
    print("Test Performance COOT : {} ".format(perf_coot_test))
