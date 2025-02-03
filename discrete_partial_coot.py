import math
import os
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot
import tf_keras
from tf_keras.layers import Dense
import ot

sys.path.append(os.path.abspath('src'))

import jdcoot
from jdcoot.comp import comp_
from jdcoot.coot import cot_numpy

INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

S, T = jdcoot.Sref(INDEX_GENERATION)
S_test, T_test = jdcoot.Sref_test(INDEX_GENERATION)

S_test = S_test.loc[:, S.columns]
T_test = T_test.loc[:, T.columns]

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

prop_S = 0.1
prop_T = 0.1
alpha = 2.875

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
        Dense(units=128, input_shape=shape, activation='sigmoid'),
        Dense(units=nClass, activation='sigmoid')])
    return model

vfunc = np.vectorize(lambda arr: 'X' in arr)
fe_size = sum(vfunc(T.columns))  # Nombre de variables de Target
shape = (fe_size,)
loss = 'categorical_crossentropy'
# loss = 'MeanSquaredError
clf = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
enc = onehot(handle_unknown='ignore', sparse_output=False,
             categories=[np.arange(len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))])
# print(enc.fit_transform(Z_training_data_Target.loc[Z_training_data_Target['Z']!=-1,'Z'].values.reshape(-1,1)))

clf.fit(
    Z_training_data_Target.loc[Z_training_data_Target['Z'] != -1, Z_training_data_Target.columns != 'Z'],
    enc.fit_transform(
        Z_training_data_Target.loc[Z_training_data_Target['Z'] != -1, 'Z'].values.reshape(-1, 1)),
    batch_size=10, epochs=20, verbose=0)  # we train the classifier with target data estimated

z_test = clf.predict(T_test.loc[:, vfunc(T.columns)])
z_test = enc.inverse_transform(z_test).reshape(-1)

fe_size = sum(vfunc(S.columns))  # Nombre de variables de Targe
shape = (fe_size,)
loss = 'categorical_crossentropy'
# loss = 'MeanSquaredError
clf2 = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))

clf2.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

clf2.fit(
    Z_training_data_source.loc[Z_training_data_source['Z'] != -1, Z_training_data_source.columns != 'Z'],
    enc.fit_transform(
        Z_training_data_source.loc[Z_training_data_source['Z'] != -1, 'Z'].values.reshape(-1, 1)),
    batch_size=10, epochs=20, verbose=0)  # we train the classifier with target data estimated
z_test2 = clf2.predict(S_test.loc[:, vfunc(S.columns)])
z_test2 = enc.inverse_transform(z_test2).reshape(-1)

perf_ref = (sum(z_test == T_test.loc[:, 'Z']) + sum(z_test2 == S_test.loc[:, 'Z'])) / (len(z_test) + len(z_test2))

z_test = clf.predict(
    Z_training_data_Target.loc[Z_training_data_Target['Z'] == -1, Z_training_data_Target.columns != 'Z'])
z_test = enc.inverse_transform(z_test).reshape(-1)

z_test2 = clf2.predict(
    Z_training_data_source.loc[Z_training_data_source['Z'] == -1, Z_training_data_source.columns != 'Z'])
z_test2 = enc.inverse_transform(z_test2).reshape(-1)

perf_ref2 = (sum(z_test == T.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z']) + sum(
        z_test2 == S.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source), 'Z'])) / (
                                          len(z_test) + len(z_test2))

def compute_cost_matrix(ys, yt, v=10000):
    M = ot.dist(ys.values.reshape(-1, 1), yt.values.reshape(-1, 1), metric=comp_(v))
    return M

# Source labelled data learning
Source_indexes_labelled = np.where(Z_training_data_source['Z'] != -1)[0]

M_lin = compute_cost_matrix(yt=Z_training_data_Target['Z'],
                            ys=Z_training_data_source.loc[Source_indexes_labelled, 'Z'])
Ts, Tv, cost = cot_numpy(
    X1=Z_training_data_source.loc[Source_indexes_labelled, Z_training_data_source.columns != 'Z'],
    X2=Z_training_data_Target.loc[:, Z_training_data_Target.columns != 'Z'],
    niter=100, C_lin=M_lin,
    algo='sinkhorn', reg=1,
    algo2='emd', verbose=False)
# Target estimation
enc = onehot(handle_unknown='ignore', sparse_output=False,
             categories=[np.arange(len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))])
zs_onehot = enc.fit_transform(S.loc[Source_indexes_labelled, 'Z'].values.reshape(-1, 1))
zt_onehot_estimated = len(T.loc[:, 'Z']) * np.dot(Ts.T, zs_onehot)
zt_estimated = enc.inverse_transform(zt_onehot_estimated).reshape(-1)

# Target labelled data learning
Target_indexes_labelled = np.where(Z_training_data_Target['Z'] != -1)[0]
M_lin = compute_cost_matrix(yt=Z_training_data_source['Z'],
                            ys=Z_training_data_Target.loc[Target_indexes_labelled, 'Z'])
Ts, Tv, cost = cot_numpy(
    X1=Z_training_data_Target.loc[Target_indexes_labelled, Z_training_data_Target.columns != 'Z'],
    X2=Z_training_data_source.loc[:, Z_training_data_source.columns != 'Z'],
    niter=100, C_lin=M_lin,
    algo='sinkhorn', reg=1,
    algo2='emd', verbose=False)
# Source estimation
enc = onehot(handle_unknown='ignore', sparse_output=False,
             categories=[np.arange(len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))])
zt_onehot = enc.fit_transform(T.loc[Target_indexes_labelled, 'Z'].values.reshape(-1, 1))
zs_onehot_estimated = len(S.loc[:, 'Z']) * np.dot(Ts.T, zt_onehot)
zs_estimated = enc.inverse_transform(zs_onehot_estimated).reshape(-1)

if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target)) != 0 and len(
        np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source)) != 0:
    perf_coot = (sum(
        T.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z'] == zt_estimated[
            np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target)]) + sum(
        S.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source), 'Z'] == zs_estimated[
            np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source)])) / (
                                      len(np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                       z_labelled_target)) + len(
                                  np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source)))
else:
    perf_coot = (sum(T.loc[:, 'Z'] == zt_estimated) + sum(S.loc[:, 'Z'] == zs_estimated)) / (len(zt_estimated) + len(zs_estimated))

#############train classifier and evaluate the performance on test
def clf_seq(shape, nClass):
    model = tf_keras.Sequential([
        Dense(units=128, input_shape=shape, activation='sigmoid'),
        Dense(units=nClass, activation='sigmoid')])
    return model

vfunc = np.vectorize(lambda arr: 'X' in arr)
fe_sizeT = sum(vfunc(T.columns))  # Nombre de variables de Target
shapeT = (fe_sizeT,)
loss = 'categorical_crossentropy'
clfT = clf_seq(shapeT, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
clfT.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

fe_sizeS = sum(vfunc(S.columns))  # Nombre de variables de Target
shapeS = (fe_sizeS,)
loss = 'categorical_crossentropy'
clfS = clf_seq(shapeS, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
clfS.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

clfT.fit(T.loc[:, vfunc(T.columns)], enc.fit_transform(zt_estimated.reshape(-1, 1)), batch_size=10,
         epochs=20, verbose=0)  # we train the classifier with target data estimated
clfS.fit(S.loc[:, vfunc(S.columns)], enc.fit_transform(zs_estimated.reshape(-1, 1)), batch_size=10,
         epochs=20, verbose=0)  # we train the classifier with target data estimated

zt_test = clfT.predict(T_test.loc[:, vfunc(T_test.columns)])
zs_test = clfS.predict(S_test.loc[:, vfunc(S_test.columns)])

zt_test = enc.inverse_transform(zt_test).reshape(-1)
zs_test = enc.inverse_transform(zs_test).reshape(-1)

perf_coot_test = (sum(zt_test == T_test.loc[:, 'Z']) + sum(zs_test == S_test.loc[:, 'Z'])) / (len(zs_test) + len(zt_test))

print("Pure Performance COOT : {} ".format(perf_coot))
print("Test Performance COOT : {} ".format(perf_coot_test))
print("\n")
print("Pure Performance Reference : {} ".format(perf_ref2))
print("Test Performance Reference : {} ".format(perf_ref))
