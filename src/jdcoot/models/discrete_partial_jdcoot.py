import numpy as np
from jdcoot.models.discrete_unsupervised_jdcoot import one_hot, one_cold
import ot

from ..comp import comp_
from ..utils import (
    xcolumns,
    discrete_accuracy,
    discrete_classifiers,
)
from ..losses import loss_crossentropy2
from ..coot import init_matrix_np

def discrete_partial_jdcoot(
    source,
    target,
    test_source,
    test_target,
    l_source_train,
    l_source_test,
    l_target_train,
    l_target_test,
    **kwargs,
):
    alphaS = kwargs.get("alphaS", 3.335)
    alphaT = kwargs.get("alphaT", 3.335)
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20)
    nb_epoch = 10
    classes = np.union1d(np.unique(source.Z), np.unique(target.Z))
    nClass = len(classes)
    numIterBCD = 100
    algo2 = "emd"
    reg2 = 0
    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values
    z_source = source.Z.values.copy()
    z_target = target.Z.values.copy()

    x_source_train = source.loc[l_source_train, xcolumns(source)].values
    x_target_train = target.loc[l_target_train, xcolumns(target)].values
    z_source_train = one_hot(z_source[l_source_train], nClass).astype(np.float64)
    z_target_train = one_hot(z_target[l_target_train], nClass).astype(np.float64)
    x_source_test = x_source[l_source_test, :]
    x_target_test = x_target[l_target_test, :]
    z_source_test = source.loc[l_source_test, "Z"].values
    z_target_test = target.loc[l_target_test, "Z"].values

    clf_source, clf_target = discrete_classifiers(source, target, "relu", "softmax")

    def compute_cost_matrix(ys, yt, v=10000):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_(v))
        return M

    z_target_2 = z_target.copy()
    z_target_2[l_target_test] = -1
    z_source_2 = z_source[l_source_train]

    M_lin = compute_cost_matrix(yt=z_target_2, ys=z_source_2)

    nA, dA = x_source_train.shape
    nB, dB = x_target.shape

    vA = np.ones(dA) / dA  # is (d,)
    vB = np.ones(dB) / dB  # is (d',)
    wA = np.ones(nA) / nA  # is (n,)
    wB = np.ones(nB) / nB  # is (n',)

    # original losses
    C_s, h1_s, h2_s = init_matrix_np(x_source_train, x_target, vA, vB)
    C_v, h1_v, h2_v = init_matrix_np(x_source_train.T, x_target.T, wA, wB)

    Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
    Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')

    fcost = M_lin
    cost = np.inf
    for k in range(numIterBCD):
        Gsold = Gs.copy()
        Gvold = Gv.copy()
        costold = cost

        # step 1 : samples coupling optimization
        Ms = (C_s - np.dot(h1_s, Gv).dot(h2_s.T)) + M_lin + alphaT * fcost  # is (nA,nB)

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
        z_target_hat[l_target_train] = z_target_train
        clf_target.fit(
            x_target, z_target_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0
        )

        z_target_pred = clf_target.predict(x_target, verbose=0)
        z_target_pred[l_target_train] = z_target_train

        perf = np.mean(
            (one_cold(z_target_pred[l_target_test]) + min(classes))
            == z_target[l_target_test]
        )
        print(f"Delta: {delta} \t Loss: {cost} \t Accuracy: {perf}")

        fcost = loss_crossentropy2(z_source_train, z_target_pred)

    z_target_2 = z_target[l_target_train]
    z_source_2 = z_source.copy()
    z_source_2[l_source_test] = -1
    M_lin = compute_cost_matrix(yt=z_source_2, ys=z_target_2)

    nA, dA = x_source.shape
    nB, dB = x_target_train.shape

    vA = np.ones(dA) / dA  # is (d,)
    vB = np.ones(dB) / dB  # is (d',)
    wA = np.ones(nA) / nA  # is (n,)
    wB = np.ones(nB) / nB  # is (n',)

    # original losses
    C_s, h1_s, h2_s = init_matrix_np(x_target_train, x_source, vB, vA)
    C_v, h1_v, h2_v = init_matrix_np(x_target_train.T, x_source.T, wB, wA)

    Gs = np.ones((nB, nA)) / (nA * nB)  # is (n,n')
    Gv = np.ones((dB, dA)) / (dA * dB)  # is (d,d')
    cost = np.inf
    fcost = M_lin
    for k in range(numIterBCD):
        Gsold = Gs.copy()
        Gvold = Gv.copy()
        costold = cost

        # step 1 : samples coupling optimization
        Ms = (C_s - np.dot(h1_s, Gv).dot(h2_s.T)) + M_lin + alphaS * fcost  # is (nA,nB)

        if algo == "emd":
            Gs = ot.emd(wB, wA, Ms, numItermax=1e7)
        elif algo == "sinkhorn":
            Gs = ot.sinkhorn(wB, wA, Ms, reg)

        # step 2 : features coupling optimization
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)  # is (dA,dB)
        if algo2 == "emd":
            Gv = ot.emd(vB, vA, Mv, numItermax=1e7)
        elif algo2 == "sinkhorn":
            Gv = ot.sinkhorn(vB, vA, Mv, reg2)

        delta = np.linalg.norm(Gs - Gsold) + np.linalg.norm(Gv - Gvold)
        cost = np.sum(Mv * Gv)

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            print("converged at iter ", k)
            break

        z_source_hat = nA * Gs.T.dot(z_target_train)
        z_source_hat[l_source_train] = z_source_train
        clf_source.fit(
            x_source, z_source_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0
        )

        z_source_pred = clf_source.predict(x_source, verbose=0)
        z_source_pred[l_source_train] = z_source_train

        perf = np.mean(
            (one_cold(z_source_pred[l_source_test]) + min(classes))
            == z_source[l_source_test]
        )
        print(f"Delta: {delta} \t Loss: {cost} \t Accuracy: {perf}")

        fcost = loss_crossentropy2(z_target_train, z_source_pred)

    zpred_target = one_cold(
        clf_target.predict(x_target_test, verbose=0)
    )  # + min(target_levels    )
    zpred_source = one_cold(
        clf_source.predict(x_source_test, verbose=0)
    )  # + min(source_levels)

    perf_pure_source = discrete_accuracy(zpred_source, z_source_test)
    perf_pure_target = discrete_accuracy(zpred_target, z_target_test)

    zt_test = one_cold(
        clf_target.predict(test_target.loc[:, xcolumns(test_target)], verbose=0)
    )  # + min(target_levels)

    zs_test = one_cold(
        clf_source.predict(test_source.loc[:, xcolumns(test_source)], verbose=0)
    )  # + min(source_levels)

    perf_test_source = discrete_accuracy(zs_test, test_source.Z)
    perf_test_target = discrete_accuracy(zt_test, test_target.Z)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
