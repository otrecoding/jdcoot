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

alpha = 0.661

INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

S, T = jdcoot.Sref(INDEX_GENERATION)
S_test, T_test = jdcoot.Sref_test(INDEX_GENERATION)

S_test = S_test.loc[:, S.columns]
T_test = T_test.loc[:, T.columns]

levels_source = np.unique(S.Z)
levels_target = np.unique(S.Z)

size_s = S.Z.size
size_t = T.Z.size

if len(levels_source) > 2:

    del_idx = np.array([k for k in levels_source if sum(S.Z == k) < 0.01 * size_s])
    S = S.loc[~np.in1d(S.Z, del_idx), :].reset_index(drop=True)

if len(levels_target) > 2:

    del_idx = np.array([k for k in levels_target if sum(T.Z == k) < 0.01 * size_t])
    T = T.loc[~np.in1d(T.Z, del_idx), :].reset_index(drop=True)

S_nPerClass = min(np.unique(S.Z, return_counts=True)[1])
T_nPerClass = min(np.unique(T.Z, return_counts=True)[1])

z_kept_source = np.array([]).astype(int)
for lab in levels_source:
    z_kept_source = np.append(z_kept_source, np.random.choice(np.where(S.Z == lab)[0], S_nPerClass, replace=False))

S = S.loc[z_kept_source, :].reset_index(drop=True)

z_kept_target = np.array([]).astype(int)
for lab in levels_target:
    z_kept_target = np.append(z_kept_target, np.random.choice(np.where(T.Z == lab)[0], T_nPerClass, replace=False))

T = T.loc[z_kept_target, :].reset_index(drop=True)

def xcolumns(df):
    return [a for a in df.columns if 'X' in a]

prop_S = 1
prop_T = 0

z_labelled_source = np.array([]).astype(int)
for lab in levels_source:
    a = np.random.choice(np.where(S.Z == lab)[0], math.ceil(prop_S * sum(S.Z == lab)), replace=False)
    z_labelled_source = np.append(z_labelled_source, a)

z_labelled_target = np.array([]).astype(int)
for lab in levels_target:
    b = np.random.choice(np.where(T.Z == lab)[0], math.ceil(prop_T * sum(T.Z == lab)), replace=False)
    z_labelled_target = np.append(z_labelled_target, b)

size_s = S.Z.size
size_t = T.Z.size

l_source = np.full(size_s, True)
l_target = np.full(size_t, True)

l_source[z_labelled_source] = False
l_target[z_labelled_target] = False

train_source = S.drop(columns = 'Y')
train_source.loc[l_source, 'Z'] = -1

train_target = T.drop(columns = 'Y')
train_target.loc[l_target, 'Z'] = -1

xtrain_source = train_source.loc[:, train_source.columns != 'Z']
xtrain_target = train_target.loc[:, train_target.columns != 'Z']

ztrain_source = train_source.Z
ztrain_target = train_target.Z

nClass=len(np.union1d(levels_source, levels_target))
categories=[np.arange(nClass)]

# cost matrix with ot dist
def compute_cost_matrix(ys, yt, v=10000):
    M = ot.dist(ys.values.reshape(-1, 1), yt.values.reshape(-1, 1), metric=comp_(v))
    return M

if prop_T == 0:
    M_lin = None
else:
    M_lin = compute_cost_matrix(yt=ztrain_target, ys=ztrain_source)

Ts, Tv, cost = cot_numpy(X1=xtrain_source,
                         X2=xtrain_target,
                         niter=100, C_lin=M_lin,
                         algo='sinkhorn', reg=1,
                         algo2='emd', verbose=False)

# target estimation
enc = onehot(handle_unknown='ignore', sparse_output = False, categories=categories)
zs_onehot = enc.fit_transform(S.Z.values.reshape(-1, 1))
zt_onehot_estimated = len(T.Z) * np.dot(Ts.T, zs_onehot)
zt_estimated = enc.inverse_transform(zt_onehot_estimated).reshape(-1)

perf_coot = sum(T.Z[l_target] == zt_estimated[l_target]) / sum(l_target)

#############train classifier and evaluate performance on test
def clf_seq(shape, nClass):
    model = tf_keras.Sequential([
        Dense(units=128, input_shape=shape, activation='sigmoid'),
        Dense(units=nClass, activation='sigmoid')])
    return model


fe_size = len(xcolumns(T))  # Nombre de variables de target
shape = (fe_size,)
loss = 'categorical_crossentropy'
clf = clf_seq(shape, nClass=nClass)
clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

clf.fit(T.loc[:, xcolumns(T)], enc.fit_transform(zt_estimated.reshape(-1, 1)), batch_size=10,
        epochs=20, verbose=0)  
z_test = clf.predict(T_test.loc[:, xcolumns(T)])
z_test = enc.inverse_transform(z_test).reshape(-1)
perf_coot_test = sum(z_test == T_test.Z) / len(z_test)


print("Pure Performance COOT : {} ".format(perf_coot))
print("Test Performance COOT : {} ".format(perf_coot_test))
print("\n")
