import numpy as np
from sklearn.model_selection import train_test_split

import ot
from ..coot import init_matrix_np
from ..comp import comp_
from ..coot import cot_numpy
from ..losses import loss_crossentropy2
from ..utils import xcolumns, discrete_classifiers, discrete_accuracy, discrete_classifier
from tf_keras.utils import to_categorical


def one_hot(z, nClass):
    return to_categorical(z, num_classes=nClass)

def one_hot2(z, seen_levels):
    """
    One-hot encoding sur les classes vues dans ce fold.

    z : array-like, labels (float ou int)
    seen_levels : array-like, original labels present
    """
    z = np.array(z)
    seen_levels = np.array(seen_levels)
    indices = np.searchsorted(seen_levels, z)
    return to_categorical(indices, num_classes=len(seen_levels))
def one_cold(z_hot):
    return np.argmax(z_hot, axis=1)

def one_cold2(z_hot, seen_levels):
    indices = np.argmax(z_hot, axis=1)
    seen_levels = np.array(seen_levels)
    return seen_levels[indices]


def discrete_partial_jdcoot3(source, target, test_source, test_target,l_source_train, l_source_test,l_target_train, l_target_test, **kwargs):
    #prop_source = kwargs.get("prop_source", 0.1)
    #prop_target = kwargs.get("prop_target", 0.1)
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20)
    alpha = kwargs.get("alpha", 2.875)
    algo2 = "emd"
    reg2 = 0
    numIterBCD = 100
    nb_epoch = 10
    source_levels = np.sort(np.unique(source.Z))
    target_levels = np.sort(np.unique(target.Z))

    nClass = len(np.union1d(source_levels, target_levels))

    z_source = source.Z.values
    z_target = target.Z.values

    n_source = len(z_source)
    n_target = len(z_target)

    #l_source_train, l_source_test = train_test_split(
    #    np.arange(n_source),
    #    train_size=prop_source
    #)

    #l_target_train, l_target_test = train_test_split(
    #    np.arange(n_target),
    #    train_size=prop_target
    #)

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values
    z_source = source.Z.values.copy()
    z_target = target.Z.values.copy()

    n_source = len(z_source)
    n_target = len(z_target)

    #l_source_train, l_source_test = train_test_split(
    #    np.arange(n_source), train_size=prop_source
    #)
    #l_target_train, l_target_test = train_test_split(
    #    np.arange(n_target), train_size=prop_target
    #)

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

    zs_onehot = z_source_train
    zt_onehot_estimated = n_target * np.dot(Ts.T, zs_onehot)
    zt_onehot_estimated [l_target_train] = z_target_train
    zt_estimated = one_cold(zt_onehot_estimated)+ min(target_levels)

    z_target_2 = z_target[l_target_train]
    z_source_2 = z_source.copy()
    z_source_2[l_source_test] = -1
    M_lin = compute_cost_matrix(yt=z_source_2, ys=z_target_2)

    Ts, Tv, cost = cot_numpy(
         X1=x_target_train,
         X2=x_source,
         niter=100,
         C_lin=M_lin,
         algo="sinkhorn",
         reg=1,
         algo2="emd",
         verbose=False,
    )

    zt_onehot = z_target_train
    zs_onehot_estimated = n_source * np.dot(Ts.T, zt_onehot)
    zs_onehot_estimated[l_source_train] = z_source_train
    zs_estimated = one_cold(zs_onehot_estimated)+ min(source_levels)


    clf_source.fit(x_source, zs_onehot_estimated, batch_size=batch_size, epochs=10, verbose=0)
    clf_target.fit(x_target, zt_onehot_estimated, batch_size=batch_size, epochs=10, verbose=0)

    z_target_pred = clf_target.predict(x_target, verbose=0)
    
    z_source_pred = clf_source.predict(x_source, verbose=0)
    
    #algo = "sinkhorn"
     #reg = 1




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

    
    log_out = {}
    log_out["cost"] = []

    fcost = loss_crossentropy2(z_source_pred, z_target_pred)

    def compute_cost_matrix(ys, yt, v=10000):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_(v))
        return M
    z_target_2 = z_target.copy()
    z_target_2[l_target_test] = -1
    z_source_2 = z_source.copy()
    z_source_2[l_source_test] = -1
    M_lin = compute_cost_matrix(yt=z_target_2, ys=z_source_2)
    #fcost = M_lin

    for k in range(numIterBCD):
        Gsold = Gs.copy()
        Gvold = Gv.copy()
        costold = cost
        # step 1 : samples coupling optimization
        Ms = (C_s - np.dot(h1_s, Gv).dot(h2_s.T))
        Ms= Ms + fcost #+  M_lin# + alpha * fcost  # is (nA,nB)
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

        # estimated labels of XA using transport map on samples
        #zs_onehot_estimated = nA * Gs.dot(z_target_pred_init) {si je mets init j'ai la meme chose que COOT}
        zs_onehot_estimated = nA * Gs.dot(z_target_pred)
        zs_onehot_estimated[l_source_train] = z_source_train
            #zt_onehot_estimated = nB * Gs.T.dot(z_source_pred_init)
        zt_onehot_estimated = nB * Gs.T.dot(z_source_pred)
        zt_onehot_estimated[l_target_train] = z_target_train
            # update label estimate with models predictions
        clf_source.fit(
                x_source, zs_onehot_estimated, batch_size=batch_size, epochs=nb_epoch, verbose=0
            )
        z_source_pred = clf_source.predict(x_source, verbose=0)

        z_source_pred[l_source_train] = z_source_train

        

        clf_target.fit(
                x_target, zt_onehot_estimated, batch_size=batch_size, epochs=nb_epoch, verbose=0
            )
        z_target_pred = clf_target.predict(x_target, verbose=0)

        z_target_pred[l_target_train] = z_target_train

        accuracy = np.mean(
                (one_cold(z_target_pred[l_target_test]) + min(target_levels))
                == z_target[l_target_test]
            )

        print(f"Delta: {delta} \t  Loss: {cost} \t Accuracy: {accuracy}")

        fcost = loss_crossentropy2(z_source_pred, z_target_pred)
    
    print("Gs sum:", Gs.sum())
    print("Gs max:", Gs.max())
    print("Gs min:", Gs.min())
    
    zpred_target = one_cold(clf_target.predict(x_target_test, verbose=0)) + min(
        target_levels
    )

    zpred_source = one_cold(clf_source.predict(x_source_test, verbose=0)) + min(
        source_levels
    )

    perf_pure_source = discrete_accuracy(zpred_source, z_source_test)
    perf_pure_target = discrete_accuracy(zpred_target, z_target_test)

    zt_test = one_cold(
        clf_target.predict(test_target.loc[:, xcolumns(test_target)], verbose=0)
    ) + min(target_levels)
    zs_test = one_cold(
        clf_source.predict(test_source.loc[:, xcolumns(test_source)], verbose=0)
    ) + min(source_levels)

    perf_test_source = discrete_accuracy(zs_test, test_source.Z)
    perf_test_target = discrete_accuracy(zt_test, test_target.Z)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
