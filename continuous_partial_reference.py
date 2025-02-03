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

INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

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
S_test = data_source_test
T_test = data_target_test

S_test = S_test.loc[:, S.columns]
T_test = T_test.loc[:, T.columns]


prop_S = 0.1 # Labelled_Proportion_Source
prop_T = 0.1 # Labelled_Proportion_Target
alpha = 2.425

y_labelled_source = np.random.choice(np.arange(len(S['Y'])), math.ceil(prop_S * len(S['Y'])), replace=False)
y_labelled_target = np.random.choice(np.arange(len(T['Y'])), math.ceil(prop_T * len(T['Y'])), replace=False)
Y_training_data_source = S.drop(columns = 'Z')
Y_training_data_source.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source), 'Y'] = np.NaN
Y_training_data_Target = T.drop(columns = 'Z')
Y_training_data_Target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target), 'Y'] = np.NaN

def clf_seq(shape, nClass):
    model = tf_keras.Sequential([
        Dense(units=128, input_shape=shape, activation='linear'),
        Dense(units=nClass, activation='linear')])
    return model

vfunc = np.vectorize(lambda arr: 'X' in arr)
fe_size = sum(vfunc(T.columns))  # Nombre de variables de Targe
shape = (fe_size,)
# loss = 'categorical_crossentropy'
loss = 'MeanSquaredError'
clf = clf_seq(shape, nClass=1)
clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

clf.fit(Y_training_data_Target.loc[
            ~np.isnan(Y_training_data_Target['Y']), Y_training_data_Target.columns != 'Y'],
        Y_training_data_Target.loc[~np.isnan(Y_training_data_Target['Y']), 'Y'], batch_size=10, epochs=20,
        verbose=0)  # we train the classifier with target data estimated

z_test = clf.predict(T_test.loc[:, vfunc(T.columns)]).reshape(-1)

fe_size = sum(vfunc(S.columns))  # Nombre de variables de Targe
shape = (fe_size,)
# loss = 'categorical_crossentropy'
loss = 'MeanSquaredError'
clf2 = clf_seq(shape, nClass=1)
clf2.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

clf2.fit(Y_training_data_source.loc[
             ~np.isnan(Y_training_data_source['Y']), Y_training_data_source.columns != 'Y'],
         Y_training_data_source.loc[~np.isnan(Y_training_data_source['Y']), 'Y'], batch_size=10, epochs=20,
         verbose=0)  # we train the classifier with target data estimated

z_test2 = clf2.predict(S_test.loc[:, vfunc(S.columns)]).reshape(-1)

perf_ref = (sum((z_test - T_test.loc[:, 'Y']) ** 2) + sum((z_test2 - S_test.loc[:, 'Y']) ** 2)) / (
                                 len(z_test) + len(z_test2))

z_test = clf.predict(Y_training_data_Target.loc[np.isnan(
    Y_training_data_Target['Y']), Y_training_data_Target.columns != 'Y']).reshape(-1)
z_test2 = clf2.predict(Y_training_data_source.loc[np.isnan(
    Y_training_data_source['Y']), Y_training_data_source.columns != 'Y']).reshape(-1)

perf_ref2 = (sum((z_test - T.loc[ np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target), 'Z']) ** 2) + sum(
    (z_test2 - S.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source), 'Z']) ** 2)) / (
                                  len(z_test) + len(z_test2))

print("Pure Performance Reference : {} ".format(perf_ref2))
print("Test Performance Reference : {} ".format(perf_ref))
