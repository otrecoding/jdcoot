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
from jdcoot.utils import xcolumns
from sklearn.model_selection import train_test_split

def discrete_partial_jdcoot( source, target, test_source, test_target) :

    prop_source = 0.1
    prop_target = 0.1
    alpha = 2.875

    source_levels = np.unique(source.Z)
    target_levels = np.unique(target.Z)

    nClass = len(np.union1d(source_levels, target_levels))
    categories=[np.arange(nClass)]

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    z_source = source.Z.values
    z_target = target.Z.values

    n_source = len(z_source)
    n_target = len(z_target)

    l_source_train, l_source_test = train_test_split(np.arange(n_source), test_size = prop_source, stratify = z_source)
    l_target_train, l_target_test = train_test_split(np.arange(n_target), test_size = prop_target, stratify = z_target)

    x_source_train = source.loc[l_source_train, xcolumns(source)].values
    z_source_train = source.loc[l_source_train, 'Z'].values

    x_target_train = target.loc[l_target_train, xcolumns(target)].values
    z_target_train = target.loc[l_target_train, 'Z'].values

    x_source_test = source.loc[l_source_test, xcolumns(source)].values
    z_source_test = source.loc[l_source_test, 'Z'].values

    x_target_test = target.loc[l_target_test, xcolumns(target)].values
    z_target_test = target.loc[l_target_test, 'Z'].values

    def clf_seq(shape, nClass):
        model = tf_keras.Sequential([
            Dense(units=128, input_shape=shape, activation='relu'),
            Dense(units=nClass, activation='softmax')])
        return model
    
    fe_sizeB = len(xcolumns(target))  # Nombre de variables de Target
    shape = (fe_sizeB,)
    loss = 'categorical_crossentropy'
    clfB = clf_seq(shape, nClass=nClass)
    clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    fe_sizeA = len(xcolumns(source))  # Nombre de variables de Source
    shape = (fe_sizeA,)
    loss = 'categorical_crossentropy'
    clfA = clf_seq(shape, nClass=nClass)
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
    
    def one_cold(z_encoded):
        return np.vectorize(lambda i: np.argmax(z_encoded[i, :]))(np.arange(z_encoded.shape[0]))
    
    oh_source = one_hot(z_source_train, nClass)

    oh_target = one_hot(z_target_source, nClass)
    
    model1, model2, results = jdcot_multitask_classif(modelA=clfA, modelB=clfB,
                                                      XA=x_source_train, YA=oh_source,
                                                      XB=x_target_train, YB=oh_target,
                                                      yAtruth=z_source, yBtruth=z_target, 
                                                      algo='sinkhorn', reg=1, alpha=alpha)
    
    zpred_target = one_cold(model2.predict(x_target_test)) + min(target_levels)
    
    zpred_source = one_cold(model1.predict(x_source_test)) + min(source_levels)
    
    perf_jdcoot = (sum(zpred_source == z_source_test) 
                 + sum(zpred_target == z_target_test)) / (len(zpred_source) + len(zpred_target))

    zt_test = one_cold(model2.predict(test_target.loc[:, xcolumns(test_target)])) + min(target_levels)
    zs_test = one_cold(model1.predict(test_source.loc[:, xcolumns(test_source)])) + min(source_levels)

    perf_jdcoot_test = (sum(zt_test == test_target.Z) 
                      + sum(zs_test == test_source.Z)) / ( len(zt_test) + len(zs_test))


if __name__ == "__main__":

    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    S, T = jdcoot.Sref(INDEX_GENERATION)
    S_test, T_test = jdcoot.Sref_test(INDEX_GENERATION)
    
    S_test = S_test.loc[:, S.columns]
    T_test = T_test.loc[:, T.columns]

    perf_jdcoot, perf_jdcoot_test = discrete_partial_jdcoot( S, T, S_test, T_test)

    print("Pure Performance JDCOOT : {} ".format(perf_jdcoot))
    print("Test Performance JDCOOT : {} ".format(perf_jdcoot_test))
