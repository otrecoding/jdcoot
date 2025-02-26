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
from jdcoot.utils import xcolumns

def discrete_unsupervised_coot( source, target, source_test, target_test):

    source_levels = np.unique(source.Z)
    target_levels = np.unique(source.Z)
    
    size_target = target.Z.size

    x_source = source.loc[:, xcolumns(source)]
    x_target = target.loc[:, xcolumns(target)]

    z_source = source.Z.values
    z_target = target.Z.values
    
    nClass=len(np.union1d(source_levels, target_levels))
    categories=[np.arange(nClass)]

    encoder = onehot(handle_unknown='ignore', sparse_output = False, categories=categories)

    def one_hot(z):
        return encoder.fit_transform(z.reshape(-1, 1))

    def one_cold(z):
        return encoder.inverse_transform(z).reshape(-1)
    
    Ts, Tv, cost = cot_numpy(X1=x_source, X2=x_target, niter=100,
                             algo='sinkhorn', reg=1, algo2='emd', verbose=False)
    
    zpred_target = one_cold(size_target * np.dot(Ts.T, one_hot(z_source)))
    
    perf_coot = sum(z_target == zpred_target) / size_target
    
    def clf_seq(shape, nClass):
        model = tf_keras.Sequential([
            Dense(units=128, input_shape=shape, activation='sigmoid'),
            Dense(units=nClass, activation='sigmoid')])
        return model
    
    
    fe_size = len(xcolumns(target))  
    shape = (fe_size,)
    loss = 'categorical_crossentropy'
    clf = clf_seq(shape, nClass=nClass)
    clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    
    clf.fit(x_target, one_hot(zpred_target), batch_size=10, epochs=20, verbose=0)  

    z_test = one_cold(clf.predict(target_test.loc[:, xcolumns(target_test)]))
    perf_coot_test = sum(z_test == target_test.Z) / len(z_test)

    return perf_coot, perf_coot_test


if __name__ == '__main__':

    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    source, target = jdcoot.Sref(INDEX_GENERATION)
    source_test, target_test = jdcoot.Sref_test(INDEX_GENERATION)
    
    source_test = source_test.loc[:, source.columns]
    target_test = target_test.loc[:, target.columns]
    
    perf_coot, perf_coot_test = discrete_unsupervised_coot(source, target, source_test, target_test)

    print("Pure Performance COOT : {} ".format(perf_coot))
    print("Test Performance COOT : {} ".format(perf_coot_test))
    print("\n")
