import os
import sys
sys.path.append(os.path.abspath('src'))

import math
import numpy as np
import pandas as pd
import ot
import tf_keras
from tf_keras.layers import Dense

from jdcoot.jdcot.multitask_reg import jdcot_multitask_reg
from jdcoot.data_scenario import DataScenario, DataScenarioTest
import jdcoot
from jdcoot.utils import *
from sklearn.model_selection import train_test_split


def continuous_semisupervised_jdcoot( source, target, source_test, target_test ):

    """
    jdcot multi-task for multi regression problems
    npreds : number of parameters to predict (it has to be the same number for both datasets)
    yAtruth is (nA,npreds),yBtruth is (nB,npreds) : true value of the parameters to estimate
    YA is (nA,npreds), YB is (nB,npreds) : line of 0 if non observed labels and true values 
    if observed labels (semi supervision)
    """

    prop_target = 0.1 # 0.5 or 0.9
    alpha = 2.625

    n_source = len(source.Y)
    n_target = len(target.Y)
    
    l_target_train, l_target_test = train_test_split(np.arange(n_target), test_size = prop_target)

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values
    y_source = source.Y.values
    y_target = target.Y.values

    ytrain_target = np.copy(y_target)
    ytrain_target[l_target_test] = np.nan
    
    def clf_seq(shape):
        model = tf_keras.Sequential([Dense(units=128, input_shape=shape, 
                                           activation='linear'),
                                     Dense(units=1, activation='linear')])
        return model
    
    fe_size = len(xcolumns(target))  # Nombre de variables de target
    shape = (fe_size,)
    loss = 'MeanSquaredError'
    clf_target = clf_seq(shape)
    clf_target.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    fe_size = len(xcolumns(source))  # Nombre de variables de Source
    shape = (fe_size,)
    loss = 'MeanSquaredError'
    clf_source = clf_seq(shape)
    clf_source.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    model1, model2, results = jdcot_multitask_reg(modelA=clf_source, 
                                                  modelB=clf_target,
                                                  XA=x_source,
                                                  YA=y_source.reshape((-1, 1)),
                                                  XB=x_target,
                                                  YB=ytrain_target.reshape((-1, 1)),
                                                  yAtruth=y_source,
                                                  yBtruth=y_target, 
                                                  reshape_data=False, 
                                                  algo='sinkhorn',
                                                  reg=100, 
                                                  alpha=alpha)
    
    # predict test data in target
    
    xtest_target = x_target[l_target_test,:]
    ytest_target = y_target[l_target_test]

    ypred_target = model2.predict(xtest_target).ravel()
    perf_jdcoot = sum((ypred_target - ytest_target)**2) / len(ypred_target)
    
    zt_test = model2.predict(target_test.loc[:, xcolumns(target_test)])[:, 0]
    zs_test = model1.predict(source_test.loc[:, xcolumns(source_test)])[:, 0]

    perf_jdcoot_test = sum((zt_test - target_test.Y) ** 2) / len(zt_test)

    return perf_jdcoot, perf_jdcoot_test
    


if __name__ == '__main__':

    np.random.seed(2025)
    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    try:
        source = pd.read_csv("source.csv")
        target = pd.read_csv("target.csv")
    
        source_test = pd.read_csv("source_test.csv")
        source_test = source_test.loc[:, source.columns]
        target_test = pd.read_csv("target_test.csv")
        target_test = target_test.loc[:, target.columns]
   
    except FileNotFoundError:
    
        reference_scenario = DataScenario()
        test_scenario = DataScenarioTest()
    
        source, target = reference_scenario.generate(INDEX_GENERATION)
        source.to_csv("source.csv", index = False)
        target.to_csv("target.csv", index = False)
    
        source_test, target_test = test_scenario.generate(INDEX_GENERATION)
        source_test.to_csv("source_test.csv", index = False)
        target_test.to_csv("target_test.csv", index = False)

    perf, perf_test = continuous_semisupervised_jdcoot( source, target, source_test, target_test )

    print("Pure Performance JDCOOT : {} ".format(perf))
    print("Test Performance JDCOOT : {} ".format(perf_test))
