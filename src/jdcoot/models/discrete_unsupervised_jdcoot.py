import numpy as np
import ot

from ..utils import xcolumns, discrete_classifier, discrete_accuracy
from ..coot import init_matrix_np
from ..losses import loss_crossentropy2
from tf_keras.utils import to_categorical


def one_hot(y, nClass):
    return to_categorical(y, num_classes=nClass)


def one_cold(z_encoded):
    return np.argmax(z_encoded, axis=1)


def discrete_unsupervised_jdcoot(source, target, source_test, target_test, **kwargs):
    alpha = kwargs.get("alpha", 0.661)
    classes = np.union1d(np.unique(source.Z), np.unique(target.Z))
    nClass = len(classes)

    x_source = source.loc[:, xcolumns(source)].values
    z_source = source.Z.values

    x_target = target.loc[:, xcolumns(target)].values
    z_target = target.Z.values

    clf = discrete_classifier(target,"relu", "softmax", nClass)

    z_source_train = one_hot(z_source, nClass).astype(np.float64)
    x_target_train = x_target

    algo = "sinkhorn"
    reg = 0.1

    algo2 = "emd"
    reg2 = 0
    numIterBCD = 100
    nb_epoch = 20
    batch_size = 20

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

    # step 1 : samples coupling optimization

    Ms =  (C_s - np.dot(h1_s, Gv).dot(h2_s.T))  # is (nA,nB)
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

    z_source_pred = (
        z_source_train  # injection of known labels in the classifier predictions
    )

    z_target_hat = nB * np.dot(Gs.T, z_source_pred)
    clf.fit(
        x_target_train, z_target_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0
    )
    z_target_pred = clf.predict(x_target, verbose=0)

    fcost = loss_crossentropy2(z_source_pred, z_target_pred)
    cost = np.inf

    for k in range(numIterBCD):
        Gsold = Gs
        Gvold = Gv
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

        z_source_pred = z_source_train
        z_target_hat = nB * np.dot(Gs.T, z_source_pred)

        clf.fit(
            x_target, z_target_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0
        )

        z_target_pred = clf.predict(x_target, verbose=0)

        fcost = loss_crossentropy2(z_source_pred, z_target_pred)
        z_target_pred = one_cold(clf.predict(x_target, verbose=0)) + min(classes)
        perf_pure = np.mean(z_target_pred == z_target)

        print(f"Delta: {delta} \t  Loss: {cost} \t Accuracy: {perf_pure}")

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            print("converged at iter ", k)
            break

    z_target_test = one_cold(
        clf.predict(target_test.loc[:, xcolumns(target_test)], verbose=0)
    ) + min(classes)

    perf_pure_source = 1.0
    perf_pure_target = discrete_accuracy(z_target_pred, z_target)
    perf_test_source = 1.0
    perf_test_target = discrete_accuracy(z_target_test, target_test.Z)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
