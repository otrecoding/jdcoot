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


def continuous_semisupervised_jdcoot( data_source, data_target, data_source_test, data_target_test ):

    Labelled_Proportion_target=0.1 # 0.5 or 0.9
    
    S = data_source
    T = data_target
    S_test = data_source_test
    T_test = data_target_test
    
    prop_S = 1
    prop_T = 0.1 # 0.5 or 0.9
    alpha = 2.625
    
    y_labelled_source = np.random.choice(np.arange(len(S['Y'])), math.ceil(prop_S * len(S['Y'])), replace=False)
    y_labelled_target = np.random.choice(np.arange(len(T['Y'])), math.ceil(prop_T * len(T['Y'])), replace=False)
    Y_training_data_source = S.loc[:, S.columns != 'Z']
    Y_training_data_source.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source), 'Y'] = np.NaN
    Y_training_data_target = T.loc[:, T.columns != 'Z']
    Y_training_data_target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target), 'Y'] = np.NaN
    
    def clf_seq(shape):
        model = tf_keras.Sequential([Dense(units=128, input_shape=shape, activation='linear'),
                                     Dense(units=1, activation='linear')])
        return model
    
    vfunc = np.vectorize(lambda arr: 'X' in arr)
    fe_sizeB = sum(vfunc(T.columns))  # Nombre de variables de target
    shape = (fe_sizeB,)
    loss = 'MeanSquaredError'
    clfB = clf_seq(shape)
    clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    fe_sizeA = sum(vfunc(S.columns))  # Nombre de variables de Source
    shape = (fe_sizeA,)
    loss = 'MeanSquaredError'
    clfA = clf_seq(shape)
    clfA.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    """
    jdcot multi-task for multi regression problems
    npreds : number of parameters to predict (it has to be the same number for both datasets)
    yAtruth is (nA,npreds),yBtruth is (nB,npreds) : true value of the parameters to estimate
    YA is (nA,npreds), YB is (nB,npreds) : line of 0 if non observed labels and true values if observed labels (semi supervision)
    """
    model1, model2, results = jdcot_multitask_reg(modelA=clfA, modelB=clfB,
                                                  XA=np.array(Y_training_data_source.loc[:,
                                                              Y_training_data_source.columns != 'Y']),
                                                  YA=np.array(Y_training_data_source['Y']).reshape((-1, 1)),
                                                  XB=np.array(Y_training_data_target.loc[:,
                                                              Y_training_data_target.columns != 'Y']),
                                                  YB=np.array(Y_training_data_target['Y']).reshape((-1, 1)),
                                                  yAtruth=S['Y'],
                                                  yBtruth=T['Y'], reshape_data=False, algo='sinkhorn',
                                                  reg=100, alpha=alpha)
    
    if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target)) != 0:
        zpred_target = model2.predict(Y_training_data_target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                              y_labelled_target), Y_training_data_target.columns != 'Y'])
        zpred_target = zpred_target.ravel()
    
        perf_jdcoot = sum((zpred_target - T.loc[
            np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target), 'Y']) ** 2) / len(
            zpred_target)
    
    zt_test = model2.predict(T_test.loc[:, vfunc(T_test.columns)])[:, 0]
    zs_test = model1.predict(S_test.loc[:, vfunc(S_test.columns)])[:, 0]
    
    perf_jdcoot_test = sum((zt_test - T_test.loc[:, 'Y']) ** 2) / len(zt_test)

    return perf_jdcoot, perf_jdcoot_test
    


if __name__ == '__main__':

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

    perf, perf_test = continuous_semisupervised_jdcoot( data_source, data_target, data_source_test, data_target_test )

    print("Pure Performance JDCOOT : {} ".format(perf))
    print("Test Performance JDCOOT : {} ".format(perf_test))
