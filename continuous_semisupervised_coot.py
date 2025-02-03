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

Objective_Variable = 'continuous'
Balance=True,
Labelled_Proportion_target=0.1 # 0.5 or 0.9

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
S_test = data_source_test
T_test = data_target_test

prop_S = 1
prop_T = 0.1 # Labelled_Proportion_target
alpha = 2.625


y_labelled_source = np.random.choice(np.arange(len(S['Y'])), math.ceil(prop_S * len(S['Y'])), replace=False)
y_labelled_target = np.random.choice(np.arange(len(T['Y'])), math.ceil(prop_T * len(T['Y'])), replace=False)
Y_training_data_source = S.loc[:, S.columns != 'Z']
Y_training_data_source.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source), 'Y'] = np.NaN
Y_training_data_Target = T.loc[:, T.columns != 'Z']
Y_training_data_Target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target), 'Y'] = np.NaN

def compute_cost_matrix(ys, yt):
    M = ot.dist(ys.values.reshape(-1, 1), yt.values.reshape(-1, 1),
                metric=comp_regression())  # comp_reg ? ou comp_
    return M

if prop_T == 0:
    M_lin = None
else:
    M_lin = compute_cost_matrix(yt=Y_training_data_Target['Y'], ys=Y_training_data_source['Y'])

    # plt.imshow(M_lin)

Ts, Tv, cost = cot_numpy(X1=Y_training_data_source.loc[:, Y_training_data_source.columns != 'Y'],
                         X2=Y_training_data_Target.loc[:, Y_training_data_Target.columns != 'Y'],
                         niter=100, C_lin=M_lin,
                         algo='sinkhorn', reg=1,
                         algo2='emd', verbose=False)

zt_estimated = len(T.loc[:, 'Y']) * np.dot(Ts.T, S['Y'])

if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target)) != 0:
    # perfs_tot=np.append(perf_coot,sum((zt_estimated-T.loc[:,'Y'])**2)/len(zt_estimated)
    perf_coot = sum((zt_estimated[np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                    y_labelled_target)] - T.loc[
                                              np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                           y_labelled_target), 'Y']) ** 2) / len(
        np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target))

else:
    perf_coot = sum((T.loc[:, 'Y'] - zt_estimated) ** 2) / len(zt_estimated)

def clf_seq(shape, nClass):
    model = tf_keras.Sequential([
        Dense(units=128, input_shape=shape, activation='linear'),
        Dense(units=nClass, activation='linear')])
    return model

vfunc = np.vectorize(lambda arr: 'X' in arr)
fe_size = sum(vfunc(T.columns))  # Nombre de variables de Target
shape = (fe_size,)
loss = 'MeanSquaredError'
clf = clf_seq(shape, nClass=1)
clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
clf.fit(T.loc[:, vfunc(T.columns)], zt_estimated, batch_size=10, epochs=20,
        verbose=0)  # we train the classifier with target data estimated
z_test = clf.predict(T_test.loc[:, vfunc(T_test.columns)])[:, 0]
perf_coot_test = sum((z_test - T_test.loc[:, 'Y']) ** 2) / len(z_test)

print("Pure Performance COOT : {} ".format(perf_coot))
print("Test Performance COOT : {} ".format(perf_coot_test))
