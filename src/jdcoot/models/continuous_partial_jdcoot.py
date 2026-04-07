import numpy as np
import ot

from jdcoot.comp import comp_regression

from ..utils import xcolumns, continuous_classifiers, continuous_accuracy
from ..coot import init_matrix_np
from ..coot import cot_numpy


def continuous_partial_jdcoot(source, target, source_test, target_test,  l_source_train, l_source_test,l_target_train, l_target_test, **kwargs):

    alpha = kwargs.get("alpha", 2.425)
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20)

    n_source = len(source.Y)
    n_target = len(target.Y)
    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values
    y_source = source.Y.values[:, np.newaxis]
    y_target = target.Y.values[:, np.newaxis]

    clf_source, clf_target = continuous_classifiers(source, target)

    x_source_train = x_source[l_source_train, :]
    y_source_train = y_source[l_source_train, :]
    x_target_train = x_target[l_target_train, :]
    y_target_train = y_target[l_target_train, :]

    algo1 = algo
    reg = reg
    alpha = alpha
    algo2 = "emd"
    reg2 = 0
    numIterBCD = 100
    nb_epoch = 10
    batch_size = batch_size

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

    def compute_cost_matrix(ys, yt):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_regression())
        return M

    y_target2 = y_target.copy()
    y_target2[l_target_test] = -1
    y_source2 = y_source_train.copy()
    M_lin = compute_cost_matrix(yt=y_target2, ys=y_source2)

    Ts, Tv, cost = cot_numpy(
        X1=x_source_train,
        X2=x_target,
        niter=100,
        C_lin=M_lin,
        algo=algo,
        reg=reg,
        algo2="emd",
        verbose=False,
    )

    zt_estimated = n_target * np.dot(Ts.T, y_source_train)
    zt_estimated[l_target_train] = y_target_train

    y_target2 = y_target_train.copy()
    y_source2 = y_source.copy()
    y_source2[l_source_test] = -1
    M_lin = compute_cost_matrix(yt=y_source2, ys=y_target2)

    Ts, Tv, cost = cot_numpy(
         X1=x_target_train,
         X2=x_source,
         niter=100,
         C_lin=M_lin,
         algo=algo,
         reg=reg,
         algo2="emd",
         verbose=False,
     )

    zs_estimated = n_source * np.dot(Ts.T, y_target_train)
    zs_estimated[l_source_train] = y_source_train

    clf_target.fit(x_target, zt_estimated, batch_size=batch_size, epochs=10, verbose=0)
    clf_source.fit(x_source, zs_estimated, batch_size=batch_size, epochs=10, verbose=0)

    y_target_pred = clf_target.predict(x_target, verbose=0).ravel()
    y_source_pred = clf_source.predict(x_source, verbose=0).ravel()

    y_source_pred[l_source_train] = y_source_train.ravel()
    y_target_pred[l_target_train] = y_target_train.ravel()
    fcost = ot.dist(y_source_pred.reshape(-1,1), y_target_pred.reshape(-1,1), metric="sqeuclidean")  # is (nA,nB)

    cost = np.inf

    def compute_cost_matrix(ys, yt):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_regression())
        return M

    y_target2 = y_target.copy()
    y_target2[l_target_test] = -1
    y_source2 = y_source.copy()
    y_source2[l_source_test] = -1
    M_lin = compute_cost_matrix(yt=y_target2, ys=y_source2)
    fcost = M_lin

    for k in range(numIterBCD):
        costold = cost
        Gsold = Gs.copy()
        Gvold = Gv.copy()

        # step 1 : samples coupling optimization
        Ms = (C_s - np.dot(h1_s, Gv).dot(h2_s.T)) +  alpha * fcost  # is (nA,nB)
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

        y_source_hat = nA * Gs.dot(y_target_pred.reshape(-1,1))
        y_source_hat[l_source_train] = y_source_train
        y_target_hat = nB * Gs.T.dot(y_source_pred.reshape(-1,1))
        y_target_hat[l_target_train] = y_target_train
        
        clf_source.fit(
            x_source, y_source_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0
        )

        y_source_pred = clf_source.predict(x_source, verbose=0).ravel()
        y_source_pred[l_source_train] = y_source_train.ravel()

        clf_target.fit(
            x_target, y_target_hat, batch_size=batch_size, epochs=nb_epoch, verbose=0
        )

        y_target_pred = clf_target.predict(x_target, verbose=0).ravel()
        y_target_pred[l_target_train] = y_target_train.ravel()

        delta = np.linalg.norm(Gs - Gsold) + np.linalg.norm(Gv - Gvold)
        cost = np.sum(Mv * Gv)

        perf = continuous_accuracy(y_target_pred.ravel(), target.Y)

        print(f"Delta: {delta} \t  Loss: {cost} \t Accuracy: {perf}")

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            print("converged at iter ", k)
            break

        fcost = ot.dist(y_source_pred.reshape(-1,1), y_target_pred.reshape(-1,1), metric="sqeuclidean")

    ypred_target = clf_target.predict(x_target[l_target_test, :], verbose=0).ravel()
    ypred_source = clf_source.predict(x_source[l_source_test, :], verbose=0).ravel()

    perf_pure_source = continuous_accuracy(ypred_source, source.loc[l_source_test, "Y"])
    perf_pure_target = continuous_accuracy(ypred_target, target.loc[l_target_test, "Y"])

    yt_test = clf_target.predict(
        target_test.loc[:, xcolumns(target_test)], verbose=0
    ).ravel()
    ys_test = clf_source.predict(
        source_test.loc[:, xcolumns(source_test)], verbose=0
    ).ravel()

    perf_test_source = continuous_accuracy(ys_test, source_test.Y)
    perf_test_target = continuous_accuracy(yt_test, target_test.Y)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
