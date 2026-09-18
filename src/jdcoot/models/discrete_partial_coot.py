import numpy as np
import ot

from ..comp import comp_
from ..coot import cot_numpy
from ..utils import xcolumns, discrete_accuracy, discrete_classifiers
from tf_keras.utils import to_categorical


def one_hot(z, nClass):
    return to_categorical(z, num_classes=nClass)


def one_cold(z_hot):
    return np.argmax(z_hot, axis=1)


def discrete_partial_coot(
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
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20)
    source_levels = np.unique(source.Z)
    target_levels = np.unique(target.Z)

    nClass = len(np.union1d(source_levels, target_levels))

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values
    z_source = source.Z.values.copy()
    z_target = target.Z.values.copy()

    n_source = len(z_source)
    n_target = len(z_target)

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
    zt_onehot_estimated[l_target_train] = z_target_train

    z_target_2 = z_target[l_target_train]
    z_source_2 = z_source.copy()
    z_source_2[l_source_test] = -1
    M_lin = compute_cost_matrix(yt=z_source_2, ys=z_target_2)

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

    zt_onehot = z_target_train
    zs_onehot_estimated = n_source * np.dot(Ts.T, zt_onehot)
    zs_onehot_estimated[l_source_train] = z_source_train

    clf_source.fit(
        x_source, zs_onehot_estimated, batch_size=batch_size, epochs=10, verbose=0
    )
    clf_target.fit(
        x_target, zt_onehot_estimated, batch_size=batch_size, epochs=10, verbose=0
    )

    zpred_target = one_cold(clf_target.predict(x_target_test, verbose=0))

    zpred_source = one_cold(clf_source.predict(x_source_test, verbose=0))

    perf_pure_source = discrete_accuracy(zpred_source, z_source_test)
    perf_pure_target = discrete_accuracy(zpred_target, z_target_test)

    zt_test = one_cold(
        clf_target.predict(test_target.loc[:, xcolumns(test_target)], verbose=0)
    )

    zs_test = one_cold(
        clf_source.predict(test_source.loc[:, xcolumns(test_source)], verbose=0)
    )

    perf_test_source = discrete_accuracy(zs_test, test_source.Z)
    perf_test_target = discrete_accuracy(zt_test, test_target.Z)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
