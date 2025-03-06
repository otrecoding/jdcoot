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
from sklearn.model_selection import train_test_split
from jdcoot.utils import xcolumns


def discrete_partial_coot(source, target, test_source, test_target):

    prop_source = 0.1
    prop_target = 0.1

    source_levels = np.unique(source.Z)
    target_levels = np.unique(target.Z)

    nClass = len(np.union1d(source_levels, target_levels))
    categories=[np.arange(nClass)]

    encoder = onehot(handle_unknown='ignore', sparse_output=False, categories=categories)

    def one_hot(z):
        return encoder.fit_transform(z.reshape(-1, 1))

    def one_cold(z):
        return encoder.inverse_transform(z).reshape(-1)

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

    def compute_cost_matrix(ys, yt, v=10000):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_(v))
        return M
    
    M_lin = compute_cost_matrix(yt=z_target, ys=z_source_train)

    Ts, Tv, cost = cot_numpy( X1=x_source_train, X2=x_target,
         niter=100, C_lin=M_lin, algo='sinkhorn', reg=0.5, algo2='emd', verbose=False)

    zt_estimated = one_cold(n_target * np.dot(Ts.T, one_hot(z_source_train)))

    M_lin = compute_cost_matrix(yt=z_source, ys=z_target_train)

    zs_estimated = one_cold(n_source * np.dot(Ts.T, one_hot(z_target_train)))
     
    perf_coot = (sum(z_target_test == zt_estimated[l_target_test])
               + sum(z_source_test == zs_estimated[l_source_test])) / (len(z_target_test) + len(z_source_test))
     
    def clf_seq(shape, nClass):
        model = tf_keras.Sequential([
            Dense(units=128, input_shape=shape, activation='sigmoid'),
            Dense(units=nClass, activation='sigmoid')])
        return model
     
    fe_size_target = len(xcolumns(target))  
    shape_target = (fe_size_target,)
    loss = 'categorical_crossentropy'
    clf_target = clf_seq(shape_target, nClass=nClass)
    clf_target.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    fe_size_source = len(xcolumns(source))  # Nombre de variables de Target
    shape_source = (fe_size_source,)
    loss = 'categorical_crossentropy'
    clf_source = clf_seq(shape_source, nClass=nClass)
    clf_source.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
     
    # we train the classifier with target data estimated
    clf_target.fit(x_target, one_hot(zt_estimated), batch_size=10, epochs=20, verbose=0)  
    # we train the classifier with target data estimated
    clf_source.fit(x_source, one_hot(zs_estimated), batch_size=10, epochs=20, verbose=0)  
     
    zt_test = one_cold(clf_target.predict(test_target.loc[:, xcolumns(test_target)]))
    zs_test = one_cold(clf_source.predict(test_source.loc[:, xcolumns(test_source)]))
     
    perf_coot_test = (sum(zt_test == test_target.Z) + sum(zs_test == test_source.Z)) / (len(zs_test) + len(zt_test))

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
