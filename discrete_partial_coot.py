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
from itertools import chain


def discrete_partial_coot(source, target, test_source, test_target):

    prop_source = 0.1
    prop_target = 0.1

    source_train, source_test = train_test_split(source, test_size = prop_source, stratify = source.Z)
    target_train, target_test = train_test_split(target, test_size = prop_target, stratify = target.Z)

    x_source_train = source_train.loc[:, xcolumns(source)].values
    z_source_train = enc.fit_transform(source_train.Z.values.reshape(-1, 1))

    x_target_train = target_train.loc[:, xcolumns(target)].values
    z_target_train = enc.fit_transform(target_train.Z.values.reshape(-1, 1))

    x_source_test = source_test.loc[:, xcolumns(source)].values
    z_source_test = source_test.Z.values

    x_target_test = target_test.loc[:, xcolumns(target)].values
    z_target_test = target_test.Z.values

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
    fe_sizeT = len(xcolumns(T))  # Nombre de variables de Target
    shapeT = (fe_sizeT,)
    loss = 'categorical_crossentropy'
    clfT = clf_seq(shapeT, nClass=len(np.union1d(np.unique(S.Z), np.unique(T.Z))))
    clfT.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    fe_sizeS = len(xcolumns(S))  # Nombre de variables de Target
    shapeS = (fe_sizeS,)
    loss = 'categorical_crossentropy'
    clfS = clf_seq(shapeS, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
    clfS.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    clfT.fit(T.loc[:, xcolumns(T)], enc.fit_transform(zt_estimated.reshape(-1, 1)), batch_size=10,
             epochs=20, verbose=0)  # we train the classifier with target data estimated
    clfS.fit(S.loc[:, xcolumns(S)], enc.fit_transform(zs_estimated.reshape(-1, 1)), batch_size=10,
             epochs=20, verbose=0)  # we train the classifier with target data estimated
    
    zt_test = clfT.predict(T_test.loc[:, xcolumns(T_test)])
    zs_test = clfS.predict(S_test.loc[:, xcolumns(S_test)])
    
    zt_test = enc.inverse_transform(zt_test).reshape(-1)
    zs_test = enc.inverse_transform(zs_test).reshape(-1)
    
    perf_coot_test = (sum(zt_test == T_test.Z) + sum(zs_test == S_test.Z)) / (len(zs_test) + len(zt_test))

    return perf_coot, perf_coot_test


if __name__ == "__main__":

    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    source, target = jdcoot.Sref(INDEX_GENERATION)
    source_test, target_test = jdcoot.Sref_test(INDEX_GENERATION)
    
    source_test = source_test.loc[:, source.columns]
    target_test = target_test.loc[:, target.columns]
    
    perf_coot, perf_coot_test = discrete_partial_coot(source, target, source_test, target_test)
    
    print("Pure Performance COOT : {} ".format(perf_coot))
    print("Test Performance COOT : {} ".format(perf_coot_test))
