import os, sys
sys.path.append(os.path.abspath('src'))
import math
import numpy as np
import pandas as pd
import ot
import tf_keras
from sklearn.preprocessing import OneHotEncoder as onehot
from tf_keras.layers import Dense

import jdcoot
from jdcoot.jdcot.multitask_reg import jdcot_multitask_reg
from jdcoot.utils import *


def continuous_unsupervised_jdcoot( source, target, test_source, test_target):

    alpha = 0.3
    
    s_size = source.Y.size
    t_size = target.Y.size
    
    xtrain_source = source.loc[:, xcolumns(source)].values
    xtrain_target = target.loc[:, xcolumns(target)].values
    
    ytrain_source = source.Y.values.reshape(-1, 1)
    ytrain_target = np.full_like(target.Y.values, np.nan).reshape(-1, 1)
    
    def clf_seq(shape):
        model = tf_keras.Sequential([Dense(units=128, input_shape=shape, activation='linear'),
                                     Dense(units=1, activation='linear')])
        return model
    
    
    fe_sizeB = len(xcolumns(target))  # Nombre de variables de Target
    shape = (fe_sizeB,)
    loss = 'MeanSquaredError'
    clfB = clf_seq(shape)
    clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    fe_sizeA = len(xcolumns(source))  # Nombre de variables de Source
    shape = (fe_sizeA,)
    
    loss = 'MeanSquaredError'
    clfA = clf_seq(shape)
    clfA.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    model1, model2, results = jdcot_multitask_reg(modelA=clfA, modelB=clfB,
                                                  XA = xtrain_source,
                                                  YA = ytrain_source,
                                                  XB = xtrain_target,
                                                  YB = ytrain_target,
                                                  yAtruth=source.Y,
                                                  yBtruth=target.Y, reshape_data=False, algo='sinkhorn',
                                                  reg=100, alpha=alpha)
    
    zpred_target = model2.predict(xtrain_target).ravel()
    
    perf_jdcoot = sum((zpred_target - target.Y) ** 2) / len( zpred_target)
    
    zt_test = model2.predict(test_target.loc[:, xcolumns(test_target)]).ravel()
    zs_test = model1.predict(test_source.loc[:, xcolumns(test_source)]).ravel()
    
    perf_jdcoot_test = sum((zt_test - test_target.Y) ** 2) / len(zt_test)
    
    
    zt_test = model2.predict(test_target.loc[:, xcolumns(test_target)]).ravel()
    zs_test = model1.predict(test_source.loc[:, xcolumns(test_source)]).ravel()
    
    perf_jdcoot_test = (sum((zt_test - test_target.Y) ** 2) + sum((zs_test - test_source.Y) ** 2)) / (len(zt_test) + len(zs_test))

    return perf_jdcoot, perf_jdcoot_test


if __name__ == "__main__":

    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    source, target = jdcoot.Sref(INDEX_GENERATION)
    test_source, test_target = jdcoot.Sref_test(INDEX_GENERATION)

    test_source = test_source.loc[:, source.columns]
    test_target = test_target.loc[:, target.columns]

    perf, perf_test = continuous_unsupervised_jdcoot( source, target, test_source, test_target)
    print("Pure Performance JDCOOT : {} ".format(perf))
    print("Test Performance JDCOOT : {} ".format(perf_test))
