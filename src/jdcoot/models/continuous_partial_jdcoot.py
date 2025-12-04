import numpy as np

from ..utils import xcolumns, continuous_classifiers, continuous_accuracy
import ot
from ..coot import init_matrix_np
from sklearn.model_selection import train_test_split

def continuous_partial_jdcoot(source, target, source_test, target_test, **kwargs):

    prop_source = kwargs.get('prop_source', 0.1)
    prop_target = kwargs.get('prop_target', 0.1)

    alpha = kwargs.get('alpha', 2.425)

    n_source = len(source.Y)
    n_target = len(target.Y)
    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values
    y_source = source.Y.values[:, np.newaxis]
    y_target = target.Y.values[:, np.newaxis]

    l_source_train, l_source_test = train_test_split(np.arange(n_source), train_size = prop_source)
    l_target_train, l_target_test = train_test_split(np.arange(n_target), train_size = prop_target)

    clf_source, clf_target = continuous_classifiers(source, target)

    x_source_train = x_source[l_source_train, :]
    y_source_train = y_source[l_source_train, :]
    x_target_train = x_target[l_target_train, :]
    y_target_train = y_target[l_target_train, :]

    algo1='sinkhorn'
    reg=100
    alpha=alpha
    algo2='emd'
    reg2=0
    numIterBCD=10
    nb_epoch=10
    batch_size=10

    # Initializations
    nA, dA = x_source.shape
    nB, dB = x_target.shape

    vA = np.ones(dA) / dA  # is (d,)
    vB = np.ones(dB) / dB  # is (d',)
    wA = np.ones(nA) / nA  # is (n,)
    wB = np.ones(nB) / nB  # is (n',)

    # original losses
    C_s, h1_s, h2_s = init_matrix_np(x_source, x_target, vA, vB)
    C_v, h1_v, h2_v = init_matrix_np(x_source.T, x_target.T, wA, wB)

    Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
    Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')

    clf_source.fit(x_source_train, y_source_train, batch_size=10, epochs=nb_epoch, verbose=0)

    y_source_pred = clf_source.predict(x_source, verbose=0)
    y_source_pred[l_source_train] = y_source_train

    clf_target.fit(x_target_train, y_target_train, batch_size=10, epochs=nb_epoch, verbose=0)  

    y_target_pred = clf_target.predict(x_target, verbose=0) 

    fcost = ot.dist(y_source_pred, y_target_pred, metric='sqeuclidean')  # is (nA,nB)

    cost = np.inf

    for k in range(numIterBCD):

        costold = cost
        Gsold = Gs
        Gvold = Gv

        # step 1 : samples coupling optimization 
        Ms = alpha * (C_s - np.dot(h1_s, Gv).dot(h2_s.T)) + fcost  # is (nA,nB)
        if algo1 == 'emd':
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo1 == 'sinkhorn':
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

        # step 2 : features coupling optimization     
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)  # is (dA,dB)
        if algo2 == 'emd':
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Gv = ot.sinkhorn(vA, vB, Mv, reg2)

        y_source_hat = nA * Gs.dot(y_target_pred)
        y_source_hat[l_source_train] = y_source_train

        clf_source.fit(x_source, y_source_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0)

        y_source_pred = clf_source.predict(x_source, verbose=0)
        y_source_pred[l_source_train] = y_source_train

        y_target_hat = nB * Gs.T.dot(y_source_pred)
        y_target_hat[l_target_train] = y_target_train

        clf_target.fit(x_target, y_target_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0)

        y_target_pred = clf_target.predict(x_target, verbose=0)
        y_target_pred[l_target_train] = y_target_train

        delta = np.linalg.norm(Gs - Gsold) + np.linalg.norm(Gv - Gvold)
        cost = np.sum(Mv * Gv)

        perf = continuous_accuracy(y_target_pred.ravel(), target.Y)

        print(f'Delta: {delta} \t  Loss: {cost} \t Accuracy: {perf}')

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            print('converged at iter ', k)
            break

        fcost = ot.dist(y_source_pred, y_target_pred, metric='sqeuclidean')  

    ypred_target = clf_target.predict(x_target[l_target_test, :], verbose=0).ravel()
    ypred_source = clf_source.predict(x_source[l_source_test, :], verbose=0).ravel()

    perf_pure = continuous_accuracy(ypred_source, source.loc[l_source_test, 'Y'], 
                                    ypred_target, target.loc[l_target_test, 'Y'])

    yt_test = clf_target.predict(target_test.loc[:, xcolumns(target_test)], verbose=0).ravel()
    ys_test = clf_source.predict(source_test.loc[:, xcolumns(source_test)], verbose=0).ravel()

    perf_source = continuous_accuracy(ys_test, source_test.Y)
    perf_target = continuous_accuracy(yt_test, target_test.Y)

    return perf_source, perf_target
