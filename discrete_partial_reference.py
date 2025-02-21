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


def discrete_partial_reference( source, target, source_test, target_test) :

    source_levels = np.unique(source.Z)
    target_levels = np.unique(target.Z)

    prop_source = 0.1
    prop_target = 0.1

    source_labels, target_labels = discrete_labels(source, prop_source, target, prop_target)

    source_train = source.drop(columns = 'Y')
    source_train.loc[np.setdiff1d(np.arange(0, np.shape(source)[0]), source_labels), 'Z'] = -1
    
    target_train = target.drop(columns = 'Y')
    target_train.loc[np.setdiff1d(np.arange(0, np.shape(target)[0]), target_labels), 'Z'] = -1
    
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
    
    clf.fit(
        target_train.loc[target_train.Z != -1, target_train.columns != 'Z'],
        enc.fit_transform(target_train.loc[target_train.Z != -1, 'Z'].values.reshape(-1, 1)),
        batch_size=10, epochs=20, verbose=0)  # we train the classifier with target data estimated
    
    z_test = clf.predict(target_test.loc[:, xcolumns(target)])
    z_test = enc.inverse_transform(z_test).reshape(-1)
    
    fe_size = len(xcolumns(source))  # Nombre de variables de Targe
    shape = (fe_size,)
    loss = 'categorical_crossentropy'
    clf2 = clf_seq(shape, nClass=len(np.union1d(np.unique(source['Z']), np.unique(target['Z']))))
    
    clf2.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    clf2.fit(
        source_train.loc[source_train.Z != -1, source_train.columns != 'Z'],
        enc.fit_transform(source_train.loc[source_train.Z != -1, 'Z'].values.reshape(-1, 1)),
        batch_size=10, epochs=20, verbose=0)  # we train the classifier with target data estimated
    z_test2 = clf2.predict(source_test.loc[:, xcolumns(source)])
    z_test2 = enc.inverse_transform(z_test2).reshape(-1)
    
    perf_ref = (sum(z_test == target_test.Z) + sum(z_test2 == source_test.Z)) / (len(z_test) + len(z_test2))
    
    z_test = clf.predict(target_train.loc[target_train.Z == -1, target_train.columns != 'Z'])
    z_test = enc.inverse_transform(z_test).reshape(-1)
    
    z_test2 = clf2.predict(source_train.loc[source_train.Z == -1, source_train.columns != 'Z'])
    z_test2 = enc.inverse_transform(z_test2).reshape(-1)
    
    perf_ref2 = (sum(z_test == target.loc[np.setdiff1d(np.arange(0, np.shape(target)[0]), target_labels), 'Z']) + sum(
            z_test2 == source.loc[np.setdiff1d(np.arange(0, np.shape(source)[0]), source_labels), 'Z'])) / (
                                              len(z_test) + len(z_test2))

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
