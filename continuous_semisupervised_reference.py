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


def continuous_semisupervised_reference( source, target, source_test, target_test):

    prop_target = 0.1 
    
    
    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    y_source = source.Y.values
    y_target = target.Y.values

    n_target = len(y_target)

    l_target_train, l_target_test = train_test_split(np.arange(n_target), test_size = prop_target)

    xtrain_target = x_target[l_target_train, :]
    ytrain_target = y_target[l_target_train]

    xtest_target = x_target[l_target_test, :]
    ytest_target = y_target[l_target_test]

    def clf_seq(shape, nClass):
        model = tf_keras.Sequential([
            Dense(units=128, input_shape=shape, activation='linear'),
            Dense(units=nClass, activation='linear')])
        return model
    
    fe_size = len(xcolumns(target))  # Nombre de variables de Targe
    shape = (fe_size,)
    loss = 'MeanSquaredError'
    clf = clf_seq(shape, nClass=1)
    clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    clf.fit(xtrain_target, ytrain_target, batch_size=10, epochs=20, verbose=0) 

    z_test = clf.predict(xtest_target).ravel()
    perf_ref1 = sum((z_test - ytest_target) ** 2) / len(z_test)
    
    x_target_test = target_test.loc[:, xcolumns(target)].values
    y_target_test = target_test.Y.values

    z_test = clf.predict(x_target_test).ravel()
    print(z_test.shape)
    perf_ref2 = sum((z_test - y_target_test) ** 2) / len(z_test)
    
    return perf_ref1, perf_ref2


if __name__ == "__main__":

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

    
    source_test = source_test.loc[:, source.columns]
    target_test = target_test.loc[:, target.columns]

    pure, test = continuous_semisupervised_reference(source, target, source_test, target_test)
    print("Pure Performance Reference : {} ".format(pure))
    print("Test Performance Reference : {} ".format(test))
