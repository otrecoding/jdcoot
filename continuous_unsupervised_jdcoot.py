import os, sys
sys.path.append(os.path.abspath('src'))
import math
import numpy as np
import pandas as pd
import ot
import tf_keras
from sklearn.preprocessing import OneHotEncoder as onehot
from tf_keras.layers import Dense

import jdcoot
from jdcoot.jdcot.multitask_reg import jdcot_multitask_reg


type_supervision='unsupervised'
Objective_Variable='continuous'
Balance=True,
Labelled_Proportion_Target=None

INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

try:
    S = pd.read_csv("source.csv")
    T = pd.read_csv("target.csv")
    S_test = pd.read_csv("source_test.csv")
    T_test = pd.read_csv("target_test.csv")
except FileNotFoundError:
    S, T = jdcoot.Sref(INDEX_GENERATION)
    S_test, T_test = jdcoot.Sref_test(INDEX_GENERATION)

algo='JDCOOT'

S_test = S_test.loc[:, S.columns]
T_test = T_test.loc[:, T.columns]

def xcolumns(df) :
    return [ a for a in df.columns if 'X' in a]

prop_S = 1
prop_T = 0
alpha = 0.3

s_size = S.Y.size
t_size = T.Y.size

l_source = np.full(s_size, True)
l_target = np.full(t_size, True)

l_source[np.random.choice(np.arange(s_size), math.ceil(prop_S * len(S.Y)), replace=False)] = False
l_target[np.random.choice(np.arange(t_size), math.ceil(prop_T * len(T.Y)), replace=False)] = False

train_source = S.drop(columns = 'Z')
train_source.loc[l_source, 'Y'] = np.NaN

train_target = T.drop(columns = 'Z')
train_target.loc[l_target, 'Y'] = np.NaN

xtrain_source = train_source.loc[:, xcolumns(S)].values
xtrain_target = train_target.loc[:, xcolumns(T)].values

ytrain_source = train_source.Y.values.reshape(-1, 1)
ytrain_target = train_target.Y.values.reshape(-1, 1)

def clf_seq(shape):
    model = tf_keras.Sequential([Dense(units=128, input_shape=shape, activation='linear'),
                                 Dense(units=1, activation='linear')])
    return model


fe_sizeB = len(xcolumns(T))  # Nombre de variables de Target
shape = (fe_sizeB,)
loss = 'MeanSquaredError'
clfB = clf_seq(shape)
clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

fe_sizeA = len(xcolumns(S))  # Nombre de variables de Source
shape = (fe_sizeA,)

loss = 'MeanSquaredError'
clfA = clf_seq(shape)
clfA.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

model1, model2, results = jdcot_multitask_reg(modelA=clfA, modelB=clfB,
                                              XA = xtrain_source,
                                              YA = ytrain_source,
                                              XB = xtrain_target,
                                              YB = ytrain_target,
                                              yAtruth=S.Y,
                                              yBtruth=T.Y, reshape_data=False, algo='sinkhorn',
                                              reg=100, alpha=alpha)

zpred_target = model2.predict(xtrain_target[l_target, :]).ravel()

perf_jdcoot = sum((zpred_target[l_target] - T.Y[l_target]) ** 2) / len( zpred_target)

print("Pure Performance JDCOOT : {} ".format(perf_jdcoot))

zt_test = model2.predict(T_test.loc[:, xcolumns(T_test)]).ravel()
zs_test = model1.predict(S_test.loc[:, xcolumns(S_test)]).ravel()

perf_jdcoot_test = sum((zt_test - T_test.Y) ** 2) / len(zt_test)

print("Test Performance JDCOOT : {} ".format(perf_jdcoot_test))

zt_test = model2.predict(T_test.loc[:, xcolumns(T_test)]).ravel()
zs_test = model1.predict(S_test.loc[:, xcolumns(S_test)]).ravel()

perf_jdcoot_test = (sum((zt_test - T_test.Y) ** 2) + sum((zs_test - S_test.Y) ** 2)) / (len(zt_test) + len(zs_test))

############################################################

print("Test Performance JDCOOT : {} ".format(perf_jdcoot_test))
print("\n")

