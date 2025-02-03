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
S_test = data_source_test.loc[:, S.columns]
T_test = data_target_test.loc[:, T.columns]

prop_S = 1

if Balance == True :

    if len(np.unique(S['Z'])) > 2:

        del_idx = np.array([])
        for k in np.unique(S['Z']):
            if sum(S['Z'] == k) < 0.01 * len(S['Z']):
                del_idx = np.append(del_idx, k)
        S = S.loc[~np.in1d(S['Z'], del_idx), :].reset_index(drop=True)

    if len(np.unique(T['Z'])) > 2:
        del_idx = np.array([])
        for k in np.unique(T['Z']):
            if sum(T['Z'] == k) < 0.01 * len(T['Z']):
                del_idx = np.append(del_idx, k)
        T = T.loc[~np.in1d(T['Z'], del_idx), :].reset_index(drop=True)

    S_nPerClass = math.ceil(min(np.unique(S['Z'], return_counts=True)[
                                    1]))  # number of observations kept referenced by the min number of available observation per class
    T_nPerClass = math.ceil(min(np.unique(T['Z'], return_counts=True)[1]))

    z_kept_source = np.array([]).astype(int)
    for lab in np.unique(S['Z']):
        z_kept_source = np.append(z_kept_source,
                                  np.random.choice(np.where(S['Z'] == lab)[0], S_nPerClass, replace=False))

    S = S.loc[z_kept_source, :].reset_index(drop=True)

    z_kept_target = np.array([]).astype(int)
    for lab in np.unique(T['Z']):
        z_kept_target = np.append(z_kept_target,
                                  np.random.choice(np.where(T['Z'] == lab)[0], T_nPerClass, replace=False))
    T = T.loc[z_kept_target, :].reset_index(drop=True)

prop_S = 1
prop_T = 0.1 # Labelled_Proportion_target
alpha = 3.335

z_labelled_source = np.array([]).astype(int)
for lab in np.unique(S['Z']):
    a = np.random.choice(np.where(S['Z'] == lab)[0], math.ceil(prop_S * sum(S['Z'] == lab)), replace=False)
    z_labelled_source = np.append(z_labelled_source, a)

z_labelled_target = np.array([]).astype(int)
for lab in np.unique(T['Z']):
    b = np.random.choice(np.where(T['Z'] == lab)[0], math.ceil(prop_T * sum(T['Z'] == lab)), replace=False)
    z_labelled_target = np.append(z_labelled_target, b)

Z_training_data_source = S.loc[:, S.columns != 'Y']
Z_training_data_source.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source), 'Z'] = -1

Z_training_data_target = T.loc[:, T.columns != 'Y']
Z_training_data_target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z'] = -1

assert prop_T != 0 and prop_T != 1

def clf_seq(shape, nClass):
    model = tf_keras.Sequential([
        Dense(units=128, input_shape=shape, activation='sigmoid'),
        Dense(units=nClass, activation='sigmoid')])
    return model

vfunc = np.vectorize(lambda arr: 'X' in arr)
fe_size = sum(vfunc(T.columns))  # Nombre de variables de Targe
shape = (fe_size,)
loss = 'categorical_crossentropy'
# loss = 'MeanSquaredError
clf = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
enc = onehot(handle_unknown='ignore', sparse_output = False,
             categories=[np.arange(len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))])

clf.fit(
    Z_training_data_target.loc[Z_training_data_target['Z'] != -1, Z_training_data_target.columns != 'Z'],
    enc.fit_transform(
        Z_training_data_target.loc[Z_training_data_target['Z'] != -1, 'Z'].values.reshape(-1, 1)),
    batch_size=10, epochs=20, verbose=0)  # we train the classifier with target data estimated

z_test = clf.predict(T_test.loc[:, vfunc(T.columns)])
z_test = enc.inverse_transform(z_test).reshape(-1)
perf_ref = sum(z_test == T_test.loc[:, 'Z']) / len(z_test)

z_test = clf.predict(
    Z_training_data_target.loc[Z_training_data_target['Z'] == -1, Z_training_data_target.columns != 'Z'])
z_test = enc.inverse_transform(z_test).reshape(-1)
perf_ref2 = sum(z_test == T.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z']) / len(z_test)


print("Pure Performance Reference : {} ".format(perf_ref2))
print("Test Performance Reference : {} ".format(perf_ref))
