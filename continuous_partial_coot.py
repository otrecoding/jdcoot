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

# cost matrix with ot dist
def compute_cost_matrix(ys, yt):
    M = ot.dist(ys.values.reshape(-1, 1), yt.values.reshape(-1, 1),
                metric=comp_regression())  # comp_reg ? ou comp_
    return M

Source_indexes_labelled = np.where(~np.isnan(Y_training_data_source['Y']))[0]
M_lin = compute_cost_matrix(yt=Y_training_data_Target['Y'],
                            ys=Y_training_data_source.loc[Source_indexes_labelled, 'Y'])
Ts, Tv, cost = cot_numpy(
    X1=Y_training_data_source.loc[Source_indexes_labelled, Y_training_data_source.columns != 'Y'],
    X2=Y_training_data_Target.loc[:, Y_training_data_Target.columns != 'Y'],
    niter=100, C_lin=M_lin,
    algo='sinkhorn', reg=1,
    algo2='emd', verbose=False)

zt_estimated = len(T.loc[:, 'Y']) * np.dot(Ts.T, S.loc[Source_indexes_labelled, 'Y'])

# Target labelled data learning
Target_indexes_labelled = np.where(~np.isnan(Y_training_data_Target['Y']))[0]
M_lin = compute_cost_matrix(yt=Y_training_data_source['Y'],
                            ys=Y_training_data_Target.loc[Target_indexes_labelled, 'Y'])
Ts, Tv, cost = cot_numpy(
    X1=Y_training_data_Target.loc[Target_indexes_labelled, Y_training_data_Target.columns != 'Y'],
    X2=Y_training_data_source.loc[:, Y_training_data_source.columns != 'Y'],
    niter=100, C_lin=M_lin,
    algo='sinkhorn', reg=1,
    algo2='emd', verbose=False)

zs_estimated = len(S.loc[:, 'Y']) * np.dot(Ts.T, T.loc[Target_indexes_labelled, 'Y'])

if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target)) != 0 and len(
        np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source)) != 0:
    perf_coot = (sum((T.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                              y_labelled_target), 'Y'] - zt_estimated[
                                               np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                            y_labelled_target)]) ** 2) + sum((S.loc[
                                                                                                  np.setdiff1d(
                                                                                                      np.arange(
                                                                                                          0,
                                                                                                          np.shape(
                                                                                                              S)[
                                                                                                              0]),
                                                                                                      y_labelled_source), 'Y'] -
                                                                                              zs_estimated[
                                                                                                  np.setdiff1d(
                                                                                                      np.arange(
                                                                                                          0,
                                                                                                          np.shape(
                                                                                                              S)[
                                                                                                              0]),
                                                                                                      y_labelled_source)]) ** 2)) / (
                                      len(np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                       y_labelled_target)) + len(
                                  np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source)))
else:
    perf_coot = (sum((T.loc[:, 'Y'] - zt_estimated) ** 2) + sum((S.loc[:, 'Y'] - zs_estimated) ** 2)) / (
                                      len(zt_estimated) + len(zs_estimated))

#############train classifier and evaluate the performance on test
def clf_seq(shape, nClass):
    model = tf_keras.Sequential([
        Dense(units=128, input_shape=shape, activation='linear'),
        Dense(units=nClass, activation='linear')])
    return model

vfunc = np.vectorize(lambda arr: 'X' in arr)
fe_sizeT = sum(vfunc(T.columns))  # Nombre de variables de Target
shapeT = (fe_sizeT,)
loss = 'MeanSquaredError'
clfT = clf_seq(shapeT, nClass=1)
clfT.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

fe_sizeS = sum(vfunc(S.columns))  # Nombre de variables de Target
shapeS = (fe_sizeS,)
loss = 'MeanSquaredError'
clfS = clf_seq(shapeS, nClass=1)
clfS.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

clfT.fit(T.loc[:, vfunc(T.columns)], zt_estimated, batch_size=10, epochs=20,
         verbose=0)  # we train the classifier with target data estimated
clfS.fit(S.loc[:, vfunc(S.columns)], zs_estimated, batch_size=10, epochs=20,
         verbose=0)  # we train the classifier with target data estimated

zt_test = clfT.predict(T_test.loc[:, vfunc(T_test.columns)])[:, 0]
zs_test = clfS.predict(S_test.loc[:, vfunc(S_test.columns)])[:, 0]

perf_coot_test = (sum((zt_test - T_test.loc[:, 'Y']) ** 2) + sum((zs_test - S_test.loc[:, 'Y']) ** 2)) / (
                                       len(zs_test) + len(zt_test))

print("Pure Performance COOT : {} ".format(perf_coot))
print("Test Performance COOT : {} ".format(perf_coot_test))
