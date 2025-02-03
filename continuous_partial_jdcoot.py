import math
import os
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot
import tf_keras
from tf_keras.layers import Dense

sys.path.append(os.path.abspath('src'))

import jdcoot
from jdcoot.jdcot.multitask_classif import jdcot_multitask_classif
from jdcoot.jdcot.multitask_reg import jdcot_multitask_reg

INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

S, T = jdcoot.Sref(INDEX_GENERATION)
S_test, T_test = jdcoot.Sref_test(INDEX_GENERATION)

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

def clf_seq(shape):
    model = tf_keras.Sequential([Dense(units=128, input_shape=shape, activation='linear'),
                                 Dense(units=1, activation='linear')])
    return model

vfunc = np.vectorize(lambda arr: 'X' in arr)
fe_sizeB = sum(vfunc(T.columns))  # Nombre de variables de Target
shape = (fe_sizeB,)
loss = 'MeanSquaredError'
clfB = clf_seq(shape)
clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

fe_sizeA = sum(vfunc(S.columns))  # Nombre de variables de Source
shape = (fe_sizeA,)
loss = 'MeanSquaredError'
clfA = clf_seq(shape)
clfA.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

"""
jdcot multi-task for multi regression problems
npreds : number of parameters to predict (it has to be the same number for both datasets)
yAtruth is (nA,npreds),yBtruth is (nB,npreds) : true value of the parameters to estimate
YA is (nA,npreds), YB is (nB,npreds) : line of 0 if non observed labels and true values if observed labels (semi supervision)
"""
# model1,model2,results=jdcot_multitask_reg(modelA,modelB,XA,YA,XB,YB,yAtruth,yBtruth)
model1, model2, results = jdcot_multitask_reg(modelA=clfA, modelB=clfB,
                                              XA=np.array(Y_training_data_source.loc[:,
                                                          Y_training_data_source.columns != 'Y']),
                                              YA=np.array(Y_training_data_source['Y']).reshape((-1, 1)),
                                              XB=np.array(Y_training_data_Target.loc[:,
                                                          Y_training_data_Target.columns != 'Y']),
                                              YB=np.array(Y_training_data_Target['Y']).reshape((-1, 1)),
                                              yAtruth=S['Y'],
                                              yBtruth=T['Y'], reshape_data=False, algo='sinkhorn',
                                              reg=100, alpha=alpha)

if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target)) != 0:
    zpred_target = model2.predict(Y_training_data_Target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                          y_labelled_target), Y_training_data_Target.columns != 'Y'])
    zpred_target = zpred_target.ravel()

    zpred_source = model1.predict(Y_training_data_source.loc[
                                      np.setdiff1d(np.arange(0, np.shape(S)[0]),
                                                   y_labelled_source), Y_training_data_source.columns != 'Y'])
    zpred_source = zpred_source.ravel()
    perf_jdcoot = (sum((zpred_source - S.loc[
        np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source), 'Y']) ** 2) + sum((
                                                                                                         zpred_target -
                                                                                                         T.loc[
                                                                                                             np.setdiff1d(
                                                                                                                 np.arange(
                                                                                                                     0,
                                                                                                                     np.shape(
                                                                                                                         T)[
                                                                                                                         0]),
                                                                                                                 y_labelled_target), 'Y']) ** 2)) / (
                                            len(zpred_source) + len(zpred_target))
###ON TEST
zt_test = model2.predict(T_test.loc[:, vfunc(T_test.columns)])[:, 0]
zs_test = model1.predict(S_test.loc[:, vfunc(S_test.columns)])[:, 0]
perf_jdcoot_test = (sum((zt_test - T_test.loc[:, 'Y']) ** 2) + sum((zs_test - S_test.loc[:, 'Y']) ** 2)) / (len(zt_test) + len(zs_test))


print("Pure Performance JDCOOT : {} ".format(perf_jdcoot))
print("Test Performance JDCOOT : {} ".format(perf_jdcoot_test))
