import numpy as np

from ..utils import continuous_accuracy, xcolumns, continuous_classifiers
import ot
from ..comp import comp_regression
from ..coot import init_matrix_np


def continuous_semisupervised_jdcoot(
    source, target, source_test, target_test, l_source_train, l_source_test, l_train, l_test, **kwargs
):

    alpha = kwargs.get("alpha", 1e-5)
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20)

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values
    y_source = source.Y.values[:, np.newaxis]
    y_target = target.Y.values[:, np.newaxis]

    x_target_train = x_target[l_train, :]
    y_target_train = y_target[l_train, :]

    clf_source, clf_target = continuous_classifiers(source, target)

    algo1 = algo
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

    clf_source.fit(x_source, y_source, batch_size=batch_size, epochs=nb_epoch, verbose=0)

    clf_target.fit(
        x_target_train, y_target_train, batch_size=batch_size, epochs=nb_epoch, verbose=0
    )

    y_target_pred = clf_target.predict(x_target, verbose=0)
    y_target_pred[l_train] = y_target_train

    fcost = ot.dist(y_source, y_target_pred, metric="sqeuclidean")

    cost = np.inf

    y_target2 = y_target.copy()
    y_target2[l_test] = -1
    def compute_cost_matrix(ys, yt):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_regression())
        return M

    M_lin = compute_cost_matrix(yt=y_target2, ys=y_source)
    fcost= M_lin
    for k in range(numIterBCD):
        costold = cost
        Gsold = Gs.copy()
        Gvold = Gv.copy()

        # step 1 : samples coupling optimization
        Ms =  (C_s - np.dot(h1_s, Gv).dot(h2_s.T)) +  alpha *fcost  # is (nA,nB)
        if algo1 == "emd":
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo1 == "sinkhorn":
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

        # step 2 : features coupling optimization
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)  # is (dA,dB)
        if algo2 == "emd":
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == "sinkhorn":
            Gv = ot.sinkhorn(vA, vB, Mv, reg2)

        y_target_hat = nB * Gs.T.dot(y_source)
        y_target_hat[l_train] = y_target_train

        clf_target.fit(
            x_target, y_target_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0
        )

        y_target_pred = clf_target.predict(x_target, verbose=0)
        y_target_pred[l_train] = y_target_train

        perf = clf_target.evaluate(x_target, y_target, verbose=0)

        delta = np.linalg.norm(Gs - Gsold) + np.linalg.norm(Gv - Gvold)
        cost = np.sum(Mv * Gv)

        perf = clf_target.evaluate(x_target, target.Y, verbose=0)

        print(f"Delta: {delta:15.7f} \t  Loss: {cost} \t Accuracy: {perf[0]}")

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            print("converged at iter ", k)
            break

        fcost = ot.dist(y_source, y_target_pred, metric="sqeuclidean")

    xtest_target = x_target[l_test, :]
    ytest_target = y_target[l_test]
    ypred_target = clf_target.predict(xtest_target, verbose=0).ravel()

    perf_pure_source = 0.0
    perf_pure_target = continuous_accuracy(ypred_target, ytest_target.ravel())

    zt_test = clf_target.predict(
        target_test.loc[:, xcolumns(target_test)], verbose=0
    ).ravel()
    zs_test = clf_source.predict(
        source_test.loc[:, xcolumns(source_test)], verbose=0
    ).ravel()

    perf_test_source = continuous_accuracy(zs_test, source_test.Y)
    perf_test_target = continuous_accuracy(zt_test, target_test.Y)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
