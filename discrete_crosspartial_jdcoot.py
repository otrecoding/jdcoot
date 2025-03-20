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

from jdcoot.utils import *
from discrete_partial_reference import *


def discrete_crosspartial_jdcoot( S, T, S_test, T_test) :

    if len(np.unique(S.Z)) > 2:
    
        del_idx = []
        for k in np.unique(S.Z):
            if sum(S.Z == k) < 0.01 * len(S.Z):
                del_idx.append(k)
        S = S.loc[~np.in1d(S.Z, del_idx), :].reset_index(drop=True)
    
    if len(np.unique(T.Z)) > 2:
        del_idx = []
        for k in np.unique(T.Z):
            if sum(T.Z == k) < 0.01 * len(T.Z):
                del_idx.append(k)
        T = T.loc[~np.in1d(T.Z, del_idx), :].reset_index(drop=True)
    
    # number of observations kept referenced by the min number of available observation per class
    S_nPerClass = math.ceil(min(np.unique(S.Z, return_counts=True)[1]))  
    T_nPerClass = math.ceil(min(np.unique(T.Z, return_counts=True)[1]))
    
    z_kept_source = np.array([]).astype(int)
    for lab in np.unique(S.Z):
        c = np.random.choice(np.where(S.Z == lab)[0], S_nPerClass, replace=False)
        z_kept_source = np.append(z_kept_source, c)
    
    S = S.loc[z_kept_source, :].reset_index(drop=True)
    
    z_kept_target = np.array([]).astype(int)
    for lab in np.unique(T.Z):
        z_kept_target = np.append(z_kept_target,
                                  np.random.choice(np.where(T.Z == lab)[0], T_nPerClass, replace=False))
    T = T.loc[z_kept_target, :].reset_index(drop=True)
    
    prop_S = 0.1 # Labelled_Proportion_Source
    prop_T = 0.1 # Labelled_Proportion_Target
    alpha = 2.875
    
    z_labelled_source = np.array([]).astype(int)
    for lab in np.unique(S.Z):
        a = np.random.choice(np.where(S.Z == lab)[0], math.ceil(prop_S * sum(S.Z == lab)), replace=False)
        z_labelled_source = np.append(z_labelled_source, a)
    
    z_labelled_target = np.array([]).astype(int)
    for lab in np.unique(T.Z):
        b = np.random.choice(np.where(T.Z == lab)[0], math.ceil(prop_T * sum(T.Z == lab)), replace=False)
        z_labelled_target = np.append(z_labelled_target, b)
    
    Z_training_data_source = S.drop(columns = 'Y')
    Z_training_data_source.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source), 'Z'] = -1
    
    Z_training_data_Target = T.drop(columns = 'Y')
    Z_training_data_Target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z'] = -1
    
    def clf_seq(shape, nClass):
        model = tf_keras.Sequential([
            Dense(units=128, input_shape=shape, activation='relu'),
            Dense(units=nClass, activation='softmax')])
        return model
    
    fe_sizeB = len(xcolumns(T))  # Nombre de variables de Target
    shape = (fe_sizeB,)
    loss = 'categorical_crossentropy'
    clfB = clf_seq(shape, nClass=len(np.union1d(np.unique(S.Z), np.unique(T.Z))))
    clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    fe_sizeA = len(xcolumns(S))  # Nombre de variables de Source
    shape = (fe_sizeA,)
    loss = 'categorical_crossentropy'
    clfA = clf_seq(shape, nClass=len(np.union1d(np.unique(S.Z), np.unique(T.Z))))
    clfA.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    def one_hot(y, nClass):
        y = np.array(y)
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
    
    # Source Model estimation
    oh_source = one_hot(Z_training_data_source.Z, len(np.union1d(np.unique(S.Z), np.unique(T.Z))))
    # no unlabelled anymore
    oh_target = one_hot(Z_training_data_Target.Z[Z_training_data_Target.Z != -1],
                        len(np.union1d(np.unique(S.Z), np.unique(T.Z))))

    mod, model1, results = jdcot_multitask_classif(modelB=clfA, modelA=clfB,
                                                   XB=np.array(Z_training_data_source.loc[:,
                                                               Z_training_data_source.columns != 'Z']),
                                                   YB=oh_source,
                                                   XA=np.array(Z_training_data_Target.loc[
                                                                   Z_training_data_Target.Z != -1, Z_training_data_Target.columns != 'Z']),
                                                   YA=oh_target,
                                                   yBtruth=S.Z,
                                                   yAtruth=T.loc[
                                                       Z_training_data_Target.loc[:, 'Z'] != -1, 'Z'],
                                                   algo='sinkhorn', reg=1, alpha=alpha)
    
    # Target Model estimation
    oh_source = one_hot(Z_training_data_source.loc[Z_training_data_source.loc[:, 'Z'] != -1, 'Z'],
                        len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
    # no unlabelled anymore
    oh_target = one_hot(Z_training_data_Target.loc[:, 'Z'],
                        len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
    # model1,model2,results=jdcot_multitask_classif(clfB,clfA,XA,YA,XB,YB,yAtruth,yBtruth)
    
    mod, model2, results = jdcot_multitask_classif(modelA=clfA, modelB=clfB,
                                                   XA=np.array(Z_training_data_source.loc[
                                                                   Z_training_data_source.loc[:,
                                                                   'Z'] != -1, Z_training_data_source.columns != 'Z']),
                                                   YA=oh_source,
                                                   XB=np.array(Z_training_data_Target.loc[:,
                                                               Z_training_data_Target.columns != 'Z']),
                                                   YB=oh_target,
                                                   yAtruth=S.loc[
                                                       Z_training_data_source.loc[:, 'Z'] != -1, 'Z'],
                                                   yBtruth=T['Z'], algo='sinkhorn', reg=1, alpha=alpha)
    
    if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target)) != 0:
    
        zpred_enc_target = model2.predict(Z_training_data_Target.loc[
                                              np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                           z_labelled_target), Z_training_data_Target.columns != 'Z'])
        zpred_target = one_hot_inv(zpred_enc_target) + min(np.unique(T['Z']))
    
        zpred_enc_source = model1.predict(Z_training_data_source.loc[
                                              np.setdiff1d(np.arange(0, np.shape(S)[0]),
                                                           z_labelled_source), Z_training_data_source.columns != 'Z'])
        zpred_source = one_hot_inv(zpred_enc_source) + min(np.unique(S['Z']))
    
        perf_jdcoot = (sum(zpred_source == S.loc[
            np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source), 'Z']) + sum(
            zpred_target == T.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z'])) / (
                                            len(zpred_source) + len(zpred_target))
        zt_test = one_hot_inv(model2.predict(T_test.loc[:, xcolumns(T_test)])) + min(np.unique(T.Z))
        zs_test = one_hot_inv(model1.predict(S_test.loc[:, xcolumns(S_test)])) + min(np.unique(T.Z))
        perf_jdcoot_test = (sum(zt_test == T_test.loc[:, 'Z']) + sum(zs_test == S_test.loc[:, 'Z'])) / (len(zt_test) + len(zs_test))
        return perf_jdcoot, perf_jdcoot_test
    
    
    
if __name__ == "__main__":

    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    S, T = jdcoot.Sref(INDEX_GENERATION)
    S_test, T_test = jdcoot.Sref_test(INDEX_GENERATION)
    
    S_test = S_test.loc[:, S.columns]
    T_test = T_test.loc[:, T.columns]
    perf_jdcoot, perf_jdcoot_test = discrete_crosspartial_jdcoot( S, T, S_test, T_test)
    perf_ref2, perf_ref = discrete_partial_reference( S, T, S_test, T_test)

    print("\n")
    print("Pure Performance JDCOOT : {} ".format(perf_jdcoot))
    print("Test Performance JDCOOT : {} ".format(perf_jdcoot_test))
    print("Pure Performance Reference : {} ".format(perf_ref2))
    print("Test Performance Reference : {} ".format(perf_ref))

