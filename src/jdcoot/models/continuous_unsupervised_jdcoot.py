import numpy as np
from ..utils import xcolumns, continuous_accuracy, continuous_classifiers
import ot
from ..coot import cot_numpy
from ..coot import init_matrix_np

def continuous_unsupervised_jdcoot(source, target, test_source, test_target,
    l_source_target, l_source_test, l_target_train, l_target_test, **kwargs):

    alpha = kwargs.get("alpha", 1e-5)
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20)
    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    y_source = source.Y.values[:, np.newaxis]
    y_target = target.Y.values[:, np.newaxis]

    clf_source, clf_target = continuous_classifiers(source, target)

    algo1 = algo

    algo2 = "emd"
    reg2 = 1
    numIterBCD = 100
    nb_epoch = 10
    batch_size = batch_size

    nA, dA = x_source.shape
    nB, dB = x_target.shape

    vA = np.ones(dA) / dA
    vB = np.ones(dB) / dB
    wA = np.ones(nA) / nA
    wB = np.ones(nB) / nB

    C_s, h1_s, h2_s = init_matrix_np(x_source, x_target, vA, vB)
    C_v, h1_v, h2_v = init_matrix_np(x_source.T, x_target.T, wA, wB)

    Gs = np.ones((nA, nB)) / (nA * nB)
    Gv = np.ones((dA, dB)) / (dA * dB)

    clf_source.fit(x_source, y_source, batch_size=batch_size, epochs=nb_epoch, verbose=0)

    Ts, Tv, cost = cot_numpy(
        X1=x_source,
        X2=x_target,
        niter=100,
        C_lin=None,
        algo="emd",
        reg=1,
        algo2="emd",
        verbose=False,
    )

    y_target_pred = nB * np.dot(Ts.T, y_source).reshape((-1, 1))

    fcost = ot.dist(y_source, y_target_pred, metric="sqeuclidean")

    cost = np.inf

    for k in range(numIterBCD):
        costold = cost
        Gsold = Gs.copy()
        Gvold = Gv.copy()

        # step 1 : samples coupling optimization
        Ms =  (C_s - np.dot(h1_s, Gv).dot(h2_s.T)) + alpha *fcost  # is (nA,nB)
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

        clf_target.fit(
            x_target, y_target_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0
        )

        y_target_pred = clf_target.predict(x_target, verbose=0)

        fcost = ot.dist(y_source, y_target_pred, metric="sqeuclidean")

        delta = np.linalg.norm(Gs - Gsold) + np.linalg.norm(Gv - Gvold)
        cost = np.sum(Mv * Gv)

        perf = clf_target.evaluate(x_target, target.Y, verbose=0)

        print(f"Delta: {delta} \t  Loss: {cost} \t Accuracy: {perf[0]}")

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            print("converged at iter ", k)
            break

    y_target_pred = clf_target.predict(x_target, verbose=0).ravel()

    perf_pure_source = 0.0
    perf_pure_target = continuous_accuracy(y_target_pred, target.Y)

    x_source = test_source.loc[:, xcolumns(test_source)].values
    x_target = test_target.loc[:, xcolumns(test_target)].values
    y_target = clf_target.predict(x_target, verbose=0).ravel()
    y_source = clf_source.predict(x_source, verbose=0).ravel()

    perf_test_source = continuous_accuracy(y_source, test_source.Y)
    perf_test_target = continuous_accuracy(y_target, test_target.Y)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
