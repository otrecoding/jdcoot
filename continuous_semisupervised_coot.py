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
from jdcoot.utils import *
from sklearn.model_selection import train_test_split


def continuous_semisupervised_coot(source, target, source_test, target_test):

    prop_target = 0.1 # Labelled_Proportion_target

    y_source = source.Y.values
    y_target = target.Y.values

    n_source = len(y_source)
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
    
    xtrain_source = source.loc[:, xcolumns(source)]
    xtrain_target = target.loc[:, xcolumns(target)]

    Ts, Tv, cost = cot_numpy(X1=xtrain_source,
                             X2=xtrain_target,
                             niter=100, C_lin=M_lin,
                             algo='sinkhorn', reg=1,
                             algo2='emd', verbose=False)
    
    zt_estimated = n_target * np.dot(Ts.T, y_source)
    
    perf_coot = sum((zt_estimated[l_target_test] - y_target[l_target_test])**2) / len(l_target_test)
    
    def clf_seq(shape, nClass):
        model = tf_keras.Sequential([
            Dense(units=128, input_shape=shape, activation='linear'),
            Dense(units=nClass, activation='linear')])
        return model
    
    fe_size = len(xcolumns(target))  # Nombre de variables de Target
    shape = (fe_size,)
    loss = 'MeanSquaredError'
    clf = clf_seq(shape, nClass=1)
    clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    clf.fit(xtrain_target, zt_estimated, batch_size=10, epochs=20, verbose=0)  

    z_test = clf.predict(target_test.loc[:, xcolumns(target_test)])[:, 0]
    perf_coot_test = sum((z_test - target_test.Y) ** 2) / len(z_test)

    return perf_coot, perf_coot_test

if __name__ == '__main__':

    np.random.seed(2025)
    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    try:
        data_source = pd.read_csv("source.csv")
        data_target = pd.read_csv("target.csv")
    
        data_source_test = pd.read_csv("source_test.csv")
        data_source_test = data_source_test.loc[:, data_source.columns]
        data_target_test = pd.read_csv("target_test.csv")
        data_target_test = data_target_test.loc[:, data_target.columns]
    
    except FileNotFoundError:
    
        reference_scenario = DataScenario()
        test_scenario = DataScenarioTest()
    
        data_source, data_target = reference_scenario.generate(INDEX_GENERATION)
        data_source.to_csv("source.csv", index = False)
        data_target.to_csv("target.csv", index = False)
    
        data_source_test, data_target_test = test_scenario.generate(INDEX_GENERATION)
        data_source_test.to_csv("source_test.csv", index = False)
        data_target_test.to_csv("target_test.csv", index = False)


    perf_coot, perf_coot_test = continuous_semisupervised_coot( data_source, 
                                data_target, data_source_test, data_target_test)

    print("Pure Performance COOT : {} ".format(perf_coot))
    print("Test Performance COOT : {} ".format(perf_coot_test))
