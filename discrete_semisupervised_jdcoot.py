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
from jdcoot.jdcot.multitask_classif import jdcot_multitask_classif
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

if Balance == True:

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
prop_T = 0.1 #Labelled_Proportion_Target
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

Z_training_data_Target = T.loc[:, T.columns != 'Y']
Z_training_data_Target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z'] = -1

def clf_seq(shape, nClass):
    model = tf_keras.Sequential([
        Dense(units=128, input_shape=shape, activation='relu'),
        Dense(units=nClass, activation='softmax')])
    return model

vfunc = np.vectorize(lambda arr: 'X' in arr)
fe_sizeB = sum(vfunc(T.columns))  # Nombre de variables de Target
shape = (fe_sizeB,)
loss = 'categorical_crossentropy'
# loss = 'MeanSquaredError'
clfB = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

fe_sizeA = sum(vfunc(S.columns))  # Nombre de variables de Source
shape = (fe_sizeA,)
loss = 'categorical_crossentropy'
# loss = 'MeanSquaredError'
clfA = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
clfA.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

def one_hot(y, nClass):

    m = min(y)
    if m == -1:
        if len(np.unique(y)) != 1:
            m = np.sort(np.unique(y))[1]

    Y = np.zeros((len(y), nClass))
    for i in range(len(y)):
        if y[i] != -1:
            Y[i, (y[i] - m).astype(int)] = 1
    return Y

def one_hot_inv(z_encoded):
    return np.vectorize(lambda i: np.argmax(z_encoded[i, :]))(np.arange(z_encoded.shape[0]))

oh_source = one_hot(Z_training_data_source.loc[:, 'Z'],
                    len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
oh_target = one_hot(Z_training_data_Target.loc[:, 'Z'],
                    len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
# model1,model2,results=jdcot_multitask_classif(clfB,clfA,XA,YA,XB,YB,yAtruth,yBtruth)

model1, model2, results = jdcot_multitask_classif(modelA=clfA, modelB=clfB,
                                                  XA=np.array(Z_training_data_source.loc[:,
                                                              Z_training_data_source.columns != 'Z']),
                                                  YA=oh_source,
                                                  XB=np.array(Z_training_data_Target.loc[:,
                                                              Z_training_data_Target.columns != 'Z']),
                                                  YB=oh_target,
                                                  yAtruth=S['Z'],
                                                  yBtruth=T['Z'], algo='sinkhorn', reg=1, alpha=alpha)

if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target)) != 0:

    zpred_enc_target = model2.predict(Z_training_data_Target.loc[
                                          np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                       z_labelled_target), Z_training_data_Target.columns != 'Z'])
    zpred_target = one_hot_inv(zpred_enc_target) + min(np.unique(T['Z']))

    perf_jdcoot = sum(zpred_target == T.loc[
        np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z']) / len(zpred_target)
    zt_test = one_hot_inv(model2.predict(T_test.loc[:, vfunc(T_test.columns)])) + min(
        np.unique(T['Z']))
    perf_jdcoot_test = (sum(zt_test == T_test.loc[:, 'Z'])) / (len(zt_test))



print("Pure Performance JDCOOT : {} ".format(perf_jdcoot))
print("Test Performance JDCOOT : {} ".format(perf_jdcoot_test))
