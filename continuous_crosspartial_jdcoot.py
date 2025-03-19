import math
import os
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot
import tf_keras
from tf_keras.layers import Dense

sys.path.append(os.path.abspath('src'))

import jdcoot
from jdcoot.jdcot.multitask_classif import jdcot_multitask_classif
from jdcoot.jdcot.multitask_reg import jdcot_multitask_reg
from jdcoot.utils import *

def continuous_croos_partial_jdcoot( source, target, source_test, test):


    prop_S = 0.1 # Labelled_Proportion_Source
    prop_T = 0.1 # Labelled_Proportion_Target
    alpha = 2.425

    l_source, l_target = labelled_indexes( S, prop_S, T, prop_T)

    xtrain_source = S.loc[:, xcolumns(S)].values
    xtrain_target = T.loc[:, xcolumns(T)].values
    
    ytrain_source = S.Y.values.copy()
    ytrain_source[l_source] = np.nan
    ytrain_target = T.Y.values.copy()
    ytrain_target[l_target] = np.nan
    
    def clf_seq(shape):
        model = tf_keras.Sequential([Dense(units=128, input_shape=shape, activation='linear'),
                                     Dense(units=1, activation='linear')])
        return model
    
    fe_sizeB = len(xcolumns(T))  # Nombre de variables de Target
    shape = (fe_sizeB,)
    loss = 'MeanSquaredError'
    clfB = clf_seq(shape)
    clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    fe_sizeA = len(xcolumns(S))  # Nombre de variables de Source
    shape = (fe_sizeA,)
    loss = 'MeanSquaredError'
    clfA = clf_seq(shape)
    clfA.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    # Source model
    
    mod, model1, results = jdcot_multitask_reg(modelB=clfA, modelA=clfB,
                                               XB=xtrain_source,
                                               YB=ytrain_source.reshape(-1, 1),
                                               XA=xtrain_target[~l_target,:],
                                               YA=ytrain_target[~l_target].reshape(-1, 1),
                                               yBtruth=S.Y,
                                               yAtruth=T.loc[~l_target, 'Y'],
                                               reshape_data=False, algo='sinkhorn', reg=100, alpha=alpha)
    # Target Model
    mod, model2, results = jdcot_multitask_reg(modelA=clfA, modelB=clfB,
                                               XA=xtrain_source[~l_source,:],
                                               YA=ytrain_source[~l_source].reshape(-1, 1),
                                               XB=xtrain_target,
                                               YB=ytrain_target.reshape((-1, 1)),
                                               yAtruth=S.Y[~l_source],
                                               yBtruth=T.Y, reshape_data=False, algo='sinkhorn', reg=100,
                                               alpha=alpha)
    
    zpred_target = model2.predict(xtrain_target[l_target, :]).ravel()
    zpred_source = model1.predict(xtrain_source[l_source, :]).ravel()

    perf_jdcoot = (sum((zpred_source - S.Y[l_source]) ** 2) 
                 + sum((zpred_target - T.Y[l_target]) ** 2)) / (len(zpred_source) + len(zpred_target))
    
    zt_test = model2.predict(T_test.loc[:, xcolumns(T_test)])[:, 0]
    zs_test = model1.predict(S_test.loc[:, xcolumns(S_test)])[:, 0]

    perf_jdcoot_test = ( sum((zt_test - T_test.Y) ** 2) 
                       + sum((zs_test - S_test.Y) ** 2)) / (len(zt_test) + len(zs_test))
    
    

    return perf_jdcoot, perf_jdcoot_test



if __name__ == '__main__':

    np.random.seed(2025)
    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    S, T = jdcoot.Sref(INDEX_GENERATION)
    S_test, T_test = jdcoot.Sref_test(INDEX_GENERATION)
    
    S_test = S_test.loc[:, S.columns]
    T_test = T_test.loc[:, T.columns]
    
    perf, perf_test = continuous_croos_partial_jdcoot( S, T, S_test, T_test)

    print(f"Pure Performance JDCOOT : {perf} ")
    print(f"Test Performance JDCOOT : {perf_test} ")
