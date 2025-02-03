import math
import os
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot
import tf_keras
from tf_keras.layers import Dense

sys.path.append(os.path.abspath('src'))

import jdcoot
from jdcoot.coot import cot_numpy
from jdcoot.jdcot.multitask_classif import jdcot_multitask_classif

alpha = 0.661

INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

S, T = jdcoot.Sref(INDEX_GENERATION)
S_test, T_test = jdcoot.Sref_test(INDEX_GENERATION)

S_test = S_test.loc[:, S.columns]
T_test = T_test.loc[:, T.columns]

S_levels = np.unique(S.Z)
T_levels = np.unique(S.T)

if len(S_levels) > 2:

    del_idx = np.array([])
    for k in S_levels:
        if sum(S['Z'] == k) < 0.01 * len(S['Z']):
            del_idx = np.append(del_idx, k)
    S = S.loc[~np.in1d(S['Z'], del_idx), :].reset_index(drop=True)

if len(T_levels) > 2:
    del_idx = np.array([])
    for k in T_levels:
        if sum(T.Z == k) < 0.01 * len(T.Z):
            del_idx = np.append(del_idx, k)
    T = T.loc[~np.in1d(T.Z, del_idx), :].reset_index(drop=True)

S_nPerClass = math.ceil(min(np.unique(S.Z, return_counts=True)[
                                1]))  # number of observations kept referenced by the min number of available observation per class
T_nPerClass = math.ceil(min(np.unique(T.Z, return_counts=True)[1]))

z_kept_source = np.array([]).astype(int)
for lab in np.unique(S.Z):
    z_kept_source = np.append(z_kept_source,
                              np.random.choice(np.where(S.Z == lab)[0], S_nPerClass, replace=False))

S = S.loc[z_kept_source, :].reset_index(drop=True)

z_kept_target = np.array([]).astype(int)
for lab in np.unique(T.Z):
    z_kept_target = np.append(z_kept_target,
                              np.random.choice(np.where(T.Z == lab)[0], T_nPerClass, replace=False))
T = T.loc[z_kept_target, :].reset_index(drop=True)

prop_S = 1
prop_T = 0
alpha = 0.661

z_labelled_source = np.array([]).astype(int)
for lab in np.unique(S.Z):
    a = np.random.choice(np.where(S.Z == lab)[0], math.ceil(prop_S * sum(S.Z == lab)), replace=False)
    z_labelled_source = np.append(z_labelled_source, a)

z_labelled_target = np.array([]).astype(int)
for lab in np.unique(T.Z):
    b = np.random.choice(np.where(T.Z == lab)[0], math.ceil(prop_T * sum(T.Z == lab)), replace=False)
    z_labelled_target = np.append(z_labelled_target, b)

source_size = S.Z.size
target_size = T.Z.size
l_source = np.full(source_size, True)
l_target = np.full(target_size, True)

l_source[z_labelled_source] = False
l_target[z_labelled_target] = False

Z_training_data_source = S.drop(columns = 'Y')
Z_training_data_source.loc[l_source, 'Z'] = -1

Z_training_data_Target = T.drop(columns = 'Y')
Z_training_data_Target.loc[l_target, 'Z'] = -1

y_labelled_source = z_labelled_source
y_labelled_target = z_labelled_target

Y_training_data_source = S.drop(columns = 'Z')
Y_training_data_source.loc[l_source, 'Y'] = np.NaN
Y_training_data_Target = T.drop(columns = 'Z')
Y_training_data_Target.loc[l_target, 'Y'] = np.NaN

def clf_seq(shape, nClass):
    model = tf_keras.Sequential([
        Dense(units=128, input_shape=shape, activation='relu'),
        Dense(units=nClass, activation='softmax')])
    return model

def xcolumns(df):
    return [a for a in df.columns if 'X' in a]

fe_sizeB = len(xcolumns(T))  
shape = (fe_sizeB,)
loss = 'categorical_crossentropy'
clfB = clf_seq(shape, nClass=len(np.union1d(S_levels, T_levels)))
clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

fe_sizeA = len(xcolumns(S))  # Nombre de variables de Source
shape = (fe_sizeA,)
loss = 'categorical_crossentropy'
# loss = 'MeanSquaredError'
clfA = clf_seq(shape, nClass=len(np.union1d(S_levels, T_levels)))
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

oh_source = one_hot(Z_training_data_source.Z, len(np.union1d(S_levels, T_levels)))
oh_target = one_hot(Z_training_data_Target.Z, len(np.union1d(np.unique(S.Z), np.unique(T.Z))))

model1, model2, results = jdcot_multitask_classif(modelA=clfA, modelB=clfB,
                                                  XA=np.array(Z_training_data_source.loc[:,
                                                              Z_training_data_source.columns != 'Z']),
                                                  YA=oh_source,
                                                  XB=np.array(Z_training_data_Target.loc[:,
                                                              Z_training_data_Target.columns != 'Z']),
                                                  YB=oh_target,
                                                  yAtruth=S.Z,
                                                  yBtruth=T.Z, algo='sinkhorn', reg=1, alpha=alpha)

zpred_enc_target = model2.predict(Z_training_data_Target.loc[l_target, Z_training_data_Target.columns != 'Z'])
zpred_target = one_hot_inv(zpred_enc_target) + min(np.unique(T.Z))

perf_jdcoot = sum(zpred_target == T.loc[l_target, 'Z']) / len(zpred_target)
zt_test = one_hot_inv(model2.predict(T_test.loc[:, xcolumns(T_test)])) + min(np.unique(T.Z))
perf_jdcoot_test = sum(zt_test == T_test.Z) / len(zt_test)

print("Pure Performance JDCOOT : {} ".format(perf_jdcoot))
print("Test Performance JDCOOT : {} ".format(perf_jdcoot_test))
