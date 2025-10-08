import numpy as np
from sklearn.model_selection import train_test_split

import ot
from ..coot import init_matrix_np
from ..losses import loss_crossentropy2
from ..utils import xcolumns, discrete_classifiers, discrete_accuracy
from tf_keras.utils import to_categorical

def one_hot(z, nClass):

    return to_categorical(z, num_classes = nClass)

def one_cold(z_hot):

    return np.argmax(z_hot, axis=1)

def discrete_partial_jdcoot( source, target, test_source, test_target, **kwargs) :

    prop_source = kwargs.get('prop_source', 0.1)
    prop_target = kwargs.get('prop_target', 0.1)

    alpha = kwargs.get('alpha', 2.875)

    source_levels = np.sort(np.unique(source.Z))
    target_levels = np.sort(np.unique(target.Z))

    nClass = len(np.union1d(source_levels, target_levels))

    z_source = source.Z.values
    z_target = target.Z.values

    n_source = len(z_source)
    n_target = len(z_target)

    l_source_train, l_source_test = train_test_split(np.arange(n_source), 
                                                     train_size = prop_source, 
                                                     )

    l_target_train, l_target_test = train_test_split(np.arange(n_target), 
                                                     train_size = prop_target, 
                                                     )

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    x_source_train = x_source[l_source_train,:]
    x_target_train = x_target[l_target_train,:]

    z_source_train = one_hot(z_source[l_source_train], nClass)
    z_target_train = one_hot(z_target[l_target_train], nClass)

    x_source_test = x_source[l_source_test, :]
    z_source_test = z_source[l_source_test]

    x_target_test = x_target[l_target_test, :]
    z_target_test = z_target[l_target_test]

    clf_source, clf_target = discrete_classifiers(source, target, 'relu', 'softmax')
    
    algo='sinkhorn'
    reg=1

    algo2='emd'
    reg2=0
    numIterBCD=10
    nb_epoch=10
    batch_size=10

    nA, dA = x_source.shape
    nB, dB = x_target.shape

    vA = np.ones(dA) / dA 
    vB = np.ones(dB) / dB
    wA = np.ones(nA) / nA
    wB = np.ones(nB) / nB

    # original losses
    C_s, h1_s, h2_s = init_matrix_np(x_source, x_target, vA, vB)
    C_v, h1_v, h2_v = init_matrix_np(x_source.T, x_target.T, wA, wB)

    cost = np.inf

    Gs = np.ones((nA, nB)) / (nA * nB) 
    Gv = np.ones((dA, dB)) / (dA * dB)

    # we train the classifier with labelled examples only
    clf_source.fit(x_source_train, z_source_train, batch_size=batch_size, epochs=nb_epoch, verbose=0)  

    z_source_pred = clf_source.predict(x_source, verbose=0)  

    z_source_pred[l_source_train,:] = z_source_train  # injection of known labels in the classifier predictions

    # we train the classifier with labelled examples only
    clf_target.fit(x_target_train, z_target_train, batch_size=10, epochs=nb_epoch, verbose=0) 

    z_target_pred = clf_target.predict(x_target, verbose=0)  

    # injection of known labels in the classifier predictions
    z_target_pred[l_target_train] = z_target_train  

    log_out = {}
    log_out['cost'] = []

    fcost = loss_crossentropy2(z_source_pred, z_target_pred)

    for k in range(numIterBCD):

        Gsold = Gs
        Gvold = Gv
        costold = cost
        # step 1 : samples coupling optimization 
        Ms = alpha * (C_s - np.dot(h1_s, Gv).dot(h2_s.T)) + fcost  # is (nA,nB)
        if algo == 'emd':
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo == 'sinkhorn':
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

        # step 2 : features coupling optimization     
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)  # is (dA,dB)
        if algo2 == 'emd':
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Gv = ot.sinkhorn(vA, vB, Mv, reg2)

        delta = np.linalg.norm(Gs - Gsold) + np.linalg.norm(Gv - Gvold)
        cost = np.sum(Mv * Gv)


        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            print('converged at iter ', k)
            break

        # estimated labels of XA using transport map on samples
        z_source_hat = nA * Gs.dot(z_target_pred)
        z_source_hat[l_source_train] = z_source_train

        # update label estimate with models predictions
        clf_source.fit(x_source, z_source_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0)
        z_source_pred = clf_source.predict(x_source, verbose=0)

        z_source_pred[l_source_train] = z_source_train

        z_target_hat = nB * Gs.T.dot(z_source_pred)
        z_target_hat[l_target_train] = z_target_train

        clf_target.fit(x_target, z_target_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0)
        z_target_pred = clf_target.predict(x_target, verbose=0)

        z_target_pred[l_target_train] = z_target_train

        accuracy = np.mean((one_cold(z_target_pred[l_target_test]) + min(target_levels)) == z_target[l_target_test])

        print(f'Delta: {delta} \t  Loss: {cost} \t Accuracy: {accuracy}')

        fcost = loss_crossentropy2(z_source_pred, z_target_pred)

    zpred_target = one_cold(clf_target.predict(x_target_test, verbose=0)) + min(target_levels)
    
    zpred_source = one_cold(clf_source.predict(x_source_test, verbose=0)) + min(source_levels)
    
    perf_pure = discrete_accuracy(zpred_source, z_source_test, zpred_target, z_target_test)

    zt_test = one_cold(clf_target.predict(test_target.loc[:, xcolumns(test_target)], verbose=0)) + min(target_levels)
    zs_test = one_cold(clf_source.predict(test_source.loc[:, xcolumns(test_source)], verbose=0)) + min(source_levels)

    perf_test = discrete_accuracy(zt_test, test_target.Z, zs_test, test_source.Z)

    return perf_pure, perf_test


