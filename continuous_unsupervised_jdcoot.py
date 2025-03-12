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
        model = tf_keras.Sequential([Dense(units=128, 
                                     input_shape=shape, activation='linear'),
                                     Dense(units=1, activation='linear')])
        return model
    
    
    fe_size = len(xcolumns(target))  # Nombre de variables de Target
    shape = (fe_size,)
    loss = 'MeanSquaredError'
    clf_target = clf_seq(shape)
    clf_target.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    fe_size = len(xcolumns(source))  # Nombre de variables de Source
    shape = (fe_size,)
    loss = 'MeanSquaredError'
    clf_source = clf_seq(shape)
    clf_source.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    model_source, model_target, results = jdcot_multitask_reg(
                                                  modelA=clf_source, 
                                                  modelB=clf_target,
                                                  XA = xtrain_source,
                                                  YA = ytrain_source,
                                                  XB = xtrain_target,
                                                  YB = ytrain_target,
                                                  yAtruth=source.Y,
                                                  yBtruth=target.Y, 
                                                  reshape_data=False, 
                                                  algo='sinkhorn',
                                                  reg=100, alpha=alpha)
    
    zpred_target = model_target.predict(xtrain_target).ravel()
    
    perf_jdcoot = sum((zpred_target - target.Y) ** 2) / len( zpred_target)
    

    xsource = test_source.loc[:, xcolumns(test_source)].values
    xtarget = test_target.loc[:, xcolumns(test_target)].values

    ytarget = model_target.predict(xtarget).ravel()
    ysource = model_source.predict(xsource).ravel()
    
    perf_jdcoot_test = (sum((ytarget - test_target.Y) ** 2) 
                      + sum((ysource - test_source.Y) ** 2)) / (len(ytarget) + len(ysource))

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
