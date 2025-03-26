import os
import sys
import numpy as np

sys.path.append(os.path.abspath('src'))

from jdcoot.jdcot.multitask_classif import jdcot_multitask_classif
from jdcoot.utils import xcolumns, discrete_classifiers


def discrete_unsupervised_jdcoot( source, target, source_test, target_test):

    alpha = 0.661
    nClass = len(np.union1d(np.unique(source.Z), np.unique(target.Z)))
    
    x_source = source.loc[:, xcolumns(source)].values
    z_source = source.Z.values
    
    x_target = target.loc[:, xcolumns(target)].values
    z_target = np.full_like(target.Z.values, -1)
    
    clf_source, clf_target = discrete_classifiers(source, target, 'relu', 'softmax')
    
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
    
    def one_hot_inv(z_encoded):
        return np.vectorize(lambda i: np.argmax(z_encoded[i, :]))(np.arange(z_encoded.shape[0]))
    
    oh_source = one_hot(z_source, nClass)
    oh_target = one_hot(z_target, nClass)
    
    model1, model2, results = jdcot_multitask_classif(modelA=clf_source, modelB=clf_target,
                                                      XA= x_source,
                                                      YA= oh_source,
                                                      XB= x_target,
                                                      YB = oh_target,
                                                      yAtruth=source.Z,
                                                      yBtruth=target.Z, algo='sinkhorn', reg=1, alpha=alpha)
    
    zpred_enc_target = model2.predict(x_target)
    zpred_target = one_hot_inv(zpred_enc_target) + min(np.unique(target.Z))
    
    perf_pure = np.mean(zpred_target == target.Z)

    zt_test = one_hot_inv(model2.predict(target_test.loc[:, xcolumns(target_test)])) + min(np.unique(target.Z))

    perf_test = np.mean(zt_test == target_test.Z)

    return perf_pure, perf_test


if __name__ == "__main__":

    from scenario import generate_data

    data = generate_data()

    perf_pure, perf_test = discrete_unsupervised_jdcoot( *data)

    print(f"Pure Performance JDCOOT : {perf_pure}")
    print(f"Test Performance JDCOOT : {perf_test}")
