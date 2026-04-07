import numpy as np
from tf_keras.utils import to_categorical

import ot
from ..coot import init_matrix_np
from ..comp import comp_
from ..losses import loss_crossentropy2
from ..utils import xcolumns, discrete_classifier, discrete_accuracy

def one_hot(z, nClass):
    return to_categorical(z, num_classes=nClass)

def one_cold(z_hot):
    return np.argmax(z_hot, axis=1)

def discrete_semisupervised_jdcoot(source, target, source_test, target_test, 
    l_source_train, l_source_test, l_train, l_test, **kwargs):

    alpha = kwargs.get("alpha", 3.335)
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20) 
    classes = np.union1d(np.unique(source.Z), np.unique(target.Z))
    nClass = len(classes)

    x_source = source.loc[:, xcolumns(source)].values
    z_source = source.Z.values

    z_target = target.Z.values

    x_target = target.loc[:, xcolumns(target)].values

    x_target_train = x_target[l_train, :]

    z_source_train = one_hot(z_source, nClass).astype(np.float64)
    z_target_train = one_hot(z_target[l_train], nClass)

    clf = discrete_classifier(target, "relu", "softmax", nClass)

    algo = algo
    reg = reg
    algo2 = "emd"
    reg2 = 0

    numIterBCD = 100
    nb_epoch = 10
    batch_size = batch_size

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

    # we train the classifier with labelled examples only
    clf.fit(x_target_train, z_target_train, batch_size=batch_size, epochs=nb_epoch, verbose=0)
    z_target_pred = clf.predict(x_target, verbose=0)  # first estimate of XB labels
    z_target_pred[l_train] = z_target_train

    fcost = loss_crossentropy2(z_source_train, z_target_pred)
    cost = np.inf

    def compute_cost_matrix(ys, yt, v=10000):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_(v))
        return M
    z_target2 = z_target.copy()
    z_target2[l_test] = -1
    M_lin = compute_cost_matrix(yt=z_target2, ys=z_source)
    fcost  = M_lin
    for k in range(numIterBCD):
        Gsold = Gs.copy()
        Gvold = Gv.copy()
        costold = cost

        # step 1 : samples coupling optimization
        Ms =  (C_s - np.dot(h1_s, Gv).dot(h2_s.T)) + alpha *fcost  # is (nA,nB)

        if algo == "emd":
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo == "sinkhorn":
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

        # step 2 : features coupling optimization
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)  # is (dA,dB)
        if algo2 == "emd":
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == "sinkhorn":
            Gv = ot.sinkhorn(vA, vB, Mv, reg2)

        delta = np.linalg.norm(Gs - Gsold) + np.linalg.norm(Gv - Gvold)
        cost = np.sum(Mv * Gv)

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            print("converged at iter ", k)
            break

        z_target_hat = nB * Gs.T.dot(z_source_train)
        z_target_hat[l_train] = z_target_train
        clf.fit(
            x_target, z_target_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0
        )

        z_target_pred = clf.predict(x_target, verbose=0)
        z_target_pred[l_train] = z_target_train

        perf = np.mean(
            (one_cold(z_target_pred[l_test]) + min(classes)) == z_target[l_test]
        )
        print(f"Delta: {delta} \t Loss: {cost} \t Accuracy: {perf}")

        fcost = loss_crossentropy2(z_source_train, z_target_pred)

    z_target_test = one_cold(clf.predict(x_target[l_test, :], verbose=0)) + min(
        np.unique(z_source)
    )

    perf_pure_source = 1.0
    perf_pure_target = discrete_accuracy(z_target_test, z_target[l_test])

    z_target_test = one_cold(
        clf.predict(target_test.loc[:, xcolumns(target)], verbose=0)
    ) + min(np.unique(z_source))

    perf_test_source = 1.0
    perf_test_target = discrete_accuracy(z_target_test, target_test.Z)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
