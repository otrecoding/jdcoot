import os
import sys
sys.path.append(os.path.abspath('src'))

import math
import numpy as np
import pandas as pd
import ot
import tf_keras
from tf_keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder as onehot

from jdcoot.comp import comp_
from jdcoot.comp import comp_regression
from jdcoot.coot import cot_numpy
from jdcoot.data_scenario import DataScenario, DataScenarioTest
import jdcoot
from jdcoot.utils import *
from sklearn.model_selection import train_test_split


def discrete_semisupervised_coot( source, target, source_test, target_test):

    prop_target = 0.1
    
    source_levels = np.unique(source.Z)
    target_levels = np.unique(target.Z)

    nClass = len(np.union1d(source_levels, target_levels))
    categories=[np.arange(nClass)]

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    z_source = source.Z.values
    z_target = target.Z.values

    n_source = len(z_source)
    n_target = len(z_target)

    l_target_train, l_target_test = train_test_split(np.arange(n_target), 
                                                     test_size = prop_target, 
                                                     stratify = z_target)

    z_target_train = z_target.copy()
    z_target_train[l_target_test] = -1

    x_target_test = target.loc[l_target_test, xcolumns(target)].values
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

    zs_onehot = enc.fit_transform(z_source.reshape(-1, 1))
    zt_onehot_estimated = n_target * np.dot(Ts.T, zs_onehot)
    zt_estimated = enc.inverse_transform(zt_onehot_estimated).reshape(-1)

    perf_coot = np.mean(z_target_test == zt_estimated[l_target_test])
    
    #############train classifier and evaluate performance on test
    def clf_seq(shape, nClass):
        model = tf_keras.Sequential([
            Dense(units=128, input_shape=shape, activation='sigmoid'),
            Dense(units=nClass, activation='sigmoid')])
        return model
    
    fe_size = len(xcolumns(target))  # Nombre de variables de target
    shape = (fe_size,)
    loss = 'categorical_crossentropy'
    clf = clf_seq(shape, nClass=nClass)
    clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    clf.fit(x_target, enc.fit_transform(zt_estimated.reshape(-1, 1)), batch_size=10,
            epochs=20, verbose=0) 

    z_test = clf.predict(target_test.loc[:, xcolumns(target)])
    z_test = enc.inverse_transform(z_test).reshape(-1)
    perf_coot_test = np.mean(z_test == target_test.Z) 

    return perf_coot, perf_coot_test


if __name__ == "__main__":

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
    
    
    S = data_source
    T = data_target
    S_test = data_source_test.loc[:, S.columns]
    T_test = data_target_test.loc[:, T.columns]

    perf_coot, perf_coot_test = discrete_semisupervised_coot(S, T, S_test, T_test)
    
    print("Pure Performance COOT : {} ".format(perf_coot))
    print("Test Performance COOT : {} ".format(perf_coot_test))
