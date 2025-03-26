import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath('src'))

from jdcoot.jdcot.multitask_classif import jdcot_multitask_classif
from jdcoot.utils import xcolumns, discrete_classifiers, discrete_accuracy

def discrete_partial_jdcoot( source, target, test_source, test_target) :

    prop_source = 0.1
    prop_target = 0.1
    alpha = 2.875

    source_levels = np.unique(source.Z)
    target_levels = np.unique(target.Z)

    nClass = len(np.union1d(source_levels, target_levels))

    z_source = source.Z.values
    z_target = target.Z.values

    n_source = len(z_source)
    n_target = len(z_target)

    l_source_train, l_source_test = train_test_split(np.arange(n_source), test_size = prop_source, stratify = z_source)
    l_target_train, l_target_test = train_test_split(np.arange(n_target), test_size = prop_target, stratify = z_target)

    x_source_train = source.loc[:, xcolumns(source)].values
    z_source_train = np.copy(z_source)
    z_source_train[l_source_test] = -1

    x_target_train = target.loc[:, xcolumns(target)].values
    z_target_train = np.copy(z_target)
    z_target_train[l_target_test] = -1

    x_source_test = source.loc[l_source_test, xcolumns(source)].values
    z_source_test = source.Z.values[l_source_test]

    x_target_test = target.loc[l_target_test, xcolumns(target)].values
    z_target_test = target.Z.values[l_target_test]

    clfA, clfB = discrete_classifiers(source, target, 'relu', 'softmax')
    
    def one_hot(y, nClass):
    
        levels = np.sort(np.unique(y))
        m = min(y)
        if m == -1:
            if len(levels) != 1:
                m = levels[1]
    
        Y = np.zeros((len(y), nClass))
        for i in range(len(y)):
            if y[i] != -1:
                Y[i, (y[i] - m).astype(int)] = 1
        return Y
    
    def one_cold(z_encoded):
        return np.vectorize(lambda i: np.argmax(z_encoded[i, :]))(np.arange(z_encoded.shape[0]))
    
    oh_source = one_hot(z_source_train, nClass)

    oh_target = one_hot(z_target_train, nClass)
    
    model1, model2, results = jdcot_multitask_classif(modelA=clfA, modelB=clfB,
                                                      XA=x_source_train, YA=oh_source,
                                                      XB=x_target_train, YB=oh_target,
                                                      yAtruth=z_source, yBtruth=z_target, 
                                                      algo='sinkhorn', reg=1, alpha=alpha)
    
    zpred_target = one_cold(model2.predict(x_target_test)) + min(target_levels)
    
    zpred_source = one_cold(model1.predict(x_source_test)) + min(source_levels)
    
    perf_pure = discrete_accuracy(zpred_source, z_source_test, zpred_target, z_target_test)

    zt_test = one_cold(model2.predict(test_target.loc[:, xcolumns(test_target)])) + min(target_levels)
    zs_test = one_cold(model1.predict(test_source.loc[:, xcolumns(test_source)])) + min(source_levels)

    perf_test = discrete_accuracy(zt_test, test_target.Z, zs_test, test_source.Z)

    return perf_pure, perf_test


if __name__ == "__main__":

    from scenario import generate_data
    
    data = generate_data()
    perf_pure, perf_test = discrete_partial_jdcoot( *data )

    print(f"Pure Performance JDCOOT : {perf_pure}")
    print(f"Test Performance JDCOOT : {perf_test}")
