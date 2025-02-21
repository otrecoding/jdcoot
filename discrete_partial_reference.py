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
from sklearn.model_selection import train_test_split


def discrete_partial_reference( source, target, test_source, test_target) :


    source_levels = np.unique(source.Z)
    target_levels = np.unique(target.Z)

    nClass = len(np.union1d(source_levels, target_levels))
    categories=[np.arange(nClass)]

    enc = onehot(handle_unknown='ignore', sparse_output=False, categories=categories)

    source_train, source_test = train_test_split(source, test_size = 0.2, stratify = source.Z)
    target_train, target_test = train_test_split(target, test_size = 0.2, stratify = target.Z)

    x_source_train = source_train.loc[:, xcolumns(source)].values
    z_source_train = enc.fit_transform(source_train.Z.values.reshape(-1, 1))

    x_target_train = target_train.loc[:, xcolumns(target)].values
    z_target_train = enc.fit_transform(target_train.Z.values.reshape(-1, 1))

    x_source_test = source_test.loc[:, xcolumns(source)].values
    z_source_test = source_test.Z.values

    x_target_test = target_test.loc[:, xcolumns(target)].values
    z_target_test = target_test.Z.values

    def clf_seq(shape, nClass):
        model = tf_keras.Sequential([
            Dense(units=128, input_shape=shape, activation='sigmoid'),
            Dense(units=nClass, activation='sigmoid')])
        return model
    
    fe_size = len(xcolumns(target))  # Nombre de variables de Target
    shape = (fe_size,)
    loss = 'categorical_crossentropy'

    clf_target = clf_seq(shape, nClass=nClass)
    clf_target.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    clf_target.fit( x_target_train, z_target_train, batch_size=10, epochs=20, verbose=0)  
    
    fe_size = len(xcolumns(source))  # Nombre de variables de Targe
    shape = (fe_size,)
    loss = 'categorical_crossentropy'

    clf_source = clf_seq(shape, nClass=nClass)
    clf_source.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    clf_source.fit( x_source_train, z_source_train, batch_size=10, epochs=20, verbose=0)  

    x_test_target = test_target.loc[:, xcolumns(test_target)]
    z_test_target = enc.inverse_transform(clf_target.predict(x_test_target)).reshape(-1)

    x_test_source = test_source.loc[:, xcolumns(test_source)]
    z_test_source = enc.inverse_transform(clf_source.predict(x_test_source)).reshape(-1)
    
    perf_test = ( sum(z_test_target == test_target.Z) 
                + sum(z_test_source == test_source.Z)) / (len(z_test_target) + len(z_test_source))
    
    z_target_pred = enc.inverse_transform(clf_target.predict(x_target_test)).reshape(-1)
    z_source_pred = enc.inverse_transform(clf_source.predict(x_source_test)).reshape(-1)
    
    perf_pure = (  sum(z_target_pred == z_target_test) 
                 + sum(z_source_pred == z_source_test)) / (len(z_target_pred) + len(z_source_pred))

    return perf_test, perf_pure


if __name__ == "__main__":

    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    source, target = jdcoot.Sref(INDEX_GENERATION)
    source_test, target_test = jdcoot.Sref_test(INDEX_GENERATION)
    
    source_test = target_test.loc[:, source.columns]
    target_test = target_test.loc[:, target.columns]

    perf_ref, perf_ref2 = discrete_partial_reference(source, target, source_test, target_test)

    print("Pure Performance Reference : {} ".format(perf_ref2))
    print("Test Performance Reference : {} ".format(perf_ref))
