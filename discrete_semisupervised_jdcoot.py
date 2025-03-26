import os
import sys
sys.path.append(os.path.abspath('src'))

import numpy as np
from sklearn.model_selection import train_test_split

from jdcoot.jdcot.multitask_classif import jdcot_multitask_classif
from jdcoot.utils import xcolumns, discrete_classifiers

def discrete_semisupervised_jdcoot(source, target, source_test, target_test):

    prop_target = 0.1 
    alpha = 3.335

    n_target = len(target.Z)

    z_target = target.Z.values

    l_train, l_test = train_test_split(np.arange(n_target), test_size = prop_target, stratify = z_target)
    
    xtrain_source = source.loc[:, xcolumns(source)].values
    ztrain_source = source.Z.values
    
    xtrain_target = target.loc[:, xcolumns(target)].values
    ztrain_target = target.Z.values.copy()
    ztrain_target[l_test] = -1
    
    clfA, clfB = discrete_classifiers(source, target, 'relu', 'softmax')
    
    def one_hot(y, nClass):
    
        m = min(y)
        levels = np.sort(np.unique(y))
        if m == -1:
            if len(levels) != 1:
                m = levels[1]
    
        Y = np.zeros((len(y), nClass))
        for i in range(len(y)):
            if y[i] != -1:
                Y[i, (y[i] - m).astype(int)] = 1
        return Y
    
    def one_hot_inv(z_encoded):
        return np.vectorize(lambda i: np.argmax(z_encoded[i, :]))(np.arange(z_encoded.shape[0]))
    
    nClass = len(np.union1d(np.unique(source.Z), np.unique(target.Z)))

    oh_source = one_hot(ztrain_source, nClass)

    oh_target = one_hot(ztrain_target, nClass)
    
    model1, model2, results = jdcot_multitask_classif(modelA=clfA, modelB=clfB,
                                                      XA=xtrain_source,
                                                      YA=oh_source,
                                                      XB=xtrain_target,
                                                      YB=oh_target,
                                                      yAtruth=source.Z.values,
                                                      yBtruth=target.Z.values, 
                                                      algo='sinkhorn', 
                                                      reg=1, 
                                                      alpha=alpha)
    
    zpred_enc_target = model2.predict(xtrain_target[l_test,:])
                                                           
    zpred_target = one_hot_inv(zpred_enc_target) + min(np.unique(source.Z))
    
    perf_pure = np.mean(zpred_target == target.Z.loc[l_test]) 

    zt_test = one_hot_inv(model2.predict(target_test.loc[:, xcolumns(target)])) + min(np.unique(target.Z))

    perf_test = np.mean(zt_test == target_test.Z)
    
    return perf_pure, perf_test
    



if __name__ == '__main__':

    from scenario import generate_data

    data = generate_data()

    perf_pure, perf_test = discrete_semisupervised_jdcoot( *data )

    print(f"Pure Performance JDCOOT : {perf_pure}")
    print(f"Test Performance JDCOOT : {perf_test}")
