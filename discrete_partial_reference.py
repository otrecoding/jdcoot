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
from jdcoot.utils import *
from itertools import chain

def discrete_labels(source, prop_source, target, prop_target):

    n_source = len(source.Z)
    n_target = len(target.Z)
    source_levels = np.unique(source.Z)
    target_levels = np.unique(target.Z)

    if len(source_levels) > 2:
    
        del_idx = [k for k in source_levels if sum(source.Z == k) < 0.01 * len(source.Z)]

        source = source.loc[~np.in1d(source.Z, del_idx), :].reset_index(drop=True)
    
    if len(target_levels) > 2:

        del_idx = [k for k in target_levels if sum(target.Z == k) < 0.01 * len(target.Z)]

        target = target.loc[~np.in1d(target.Z, del_idx), :].reset_index(drop=True)
    
    # number of observations kept referenced by the min number of available observation per class
    source_freq = min(np.unique(source.Z, return_counts=True)[1])
    target_freq = min(np.unique(target.Z, return_counts=True)[1])
    
    kept_source = []
    for k in source_levels:
        kept_source.append(np.random.choice(np.where(source.Z == k)[0], source_freq, replace=False))
    
    source = source.loc[chain(*kept_source), :].reset_index(drop=True)
    
    kept_target = []
    for k in target_levels:
        kept_target.append(np.random.choice(np.where(target.Z == k)[0], target_freq, replace=False))

    target = target.loc[chain(*kept_target), :].reset_index(drop=True)
    
    source_labels = np.array([]).astype(int)

    for k in source_levels:
        a = np.random.choice(np.where(source.Z == k)[0], 
                             math.ceil(prop_source * sum(source.Z == k)), 
                             replace=False)

        source_labels = np.append(source_labels, a)
    
    target_labels = np.array([]).astype(int)
    for k in target_levels:
        b = np.random.choice(np.where(target.Z == k)[0], 
                             math.ceil(prop_target * sum(target.Z == k)), 
                             replace=False)

        target_labels = np.append(target_labels, b)
    

    lsource = np.full(n_source, True)
    ltarget = np.full(n_target, True)

    lsource[source_labels] = False
    ltarget[target_labels] = False

    return lsource, ltarget


def discrete_partial_reference( source, target, source_test, target_test) :

    source_levels = np.unique(source.Z)
    target_levels = np.unique(target.Z)

    prop_source = 0.1
    prop_target = 0.1

    source_labels, target_labels = discrete_labels(source, prop_source, target, prop_target)

    source_train = source.drop(columns = 'Y')
    source_train.loc[source_labels, 'Z'] = -1
    
    target_train = target.drop(columns = 'Y')
    target_train.loc[target_labels, 'Z'] = -1
    
    def clf_seq(shape, nClass):
        model = tf_keras.Sequential([
            Dense(units=128, input_shape=shape, activation='sigmoid'),
            Dense(units=nClass, activation='sigmoid')])
        return model
    
    fe_size = len(xcolumns(target))  # Nombre de variables de Target
    shape = (fe_size,)
    loss = 'categorical_crossentropy'
    clf = clf_seq(shape, nClass=len(np.union1d(source_levels, target_levels)))
    clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    categories=[np.arange(len(np.union1d(source_levels, target_levels)))]

    enc = onehot(handle_unknown='ignore', sparse_output=False, categories=categories)
    
    xtrain_source = source_train.loc[source_train.Z != -1, source_train.columns != 'Z']
    ztrain_source = enc.fit_transform(source_train.loc[source_train.Z != -1, 'Z'].values.reshape(-1, 1))

    xtrain_target = target_train.loc[target_train.Z != -1, target_train.columns != 'Z']
    ztrain_target = enc.fit_transform(target_train.loc[target_train.Z != -1, 'Z'].values.reshape(-1, 1))

    clf.fit( xtrain_target, ztrain_target, batch_size=10, epochs=20, verbose=0)  
    
    xtest_target = target_test.loc[:, xcolumns(target)]
    z_test = enc.inverse_transform(clf.predict(xtest_target)).reshape(-1)
    
    fe_size = len(xcolumns(source))  # Nombre de variables de Targe
    shape = (fe_size,)
    loss = 'categorical_crossentropy'
    clf2 = clf_seq(shape, nClass=len(np.union1d(source_levels, target_levels)))
    
    clf2.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    clf2.fit( xtrain_source, ztrain_source, batch_size=10, epochs=20, verbose=0)  

    xtest_source = source_test.loc[:, xcolumns(source)]
    z_test2 = enc.inverse_transform(clf2.predict(xtest_source)).reshape(-1)
    
    perf_ref = (sum(z_test == target_test.Z) + sum(z_test2 == source_test.Z)) / (len(z_test) + len(z_test2))
    
    z_test = clf.predict(target_train.loc[target_train.Z == -1, target_train.columns != 'Z'])
    z_test = enc.inverse_transform(z_test).reshape(-1)
    
    z_test2 = clf2.predict(source_train.loc[source_train.Z == -1, source_train.columns != 'Z'])
    z_test2 = enc.inverse_transform(z_test2).reshape(-1)
    
    perf_ref2 = (  sum(z_test == target.loc[target_labels, 'Z']) 
                 + sum(z_test2 == source.loc[source_labels, 'Z'])) / (len(z_test) + len(z_test2))

    return perf_ref, perf_ref2


if __name__ == "__main__":

    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    source, target = jdcoot.Sref(INDEX_GENERATION)
    source_test, target_test = jdcoot.Sref_test(INDEX_GENERATION)
    
    source_test = target_test.loc[:, source.columns]
    target_test = target_test.loc[:, target.columns]

    perf_ref, perf_ref2 = discrete_partial_reference(source, target, source_test, target_test)

    print("Pure Performance Reference : {} ".format(perf_ref2))
    print("Test Performance Reference : {} ".format(perf_ref))
