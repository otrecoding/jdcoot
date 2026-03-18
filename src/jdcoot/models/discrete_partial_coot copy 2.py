import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot
from jdcoot.models.discrete_unsupervised_jdcoot import one_hot
import ot

from ..comp import comp_
from ..coot import cot_numpy
from sklearn.model_selection import train_test_split
from ..utils import xcolumns, discrete_accuracy, discrete_classifiers
from tf_keras.utils import to_categorical

def one_hot(z, nClass):
    return to_categorical(z, num_classes=nClass)


def one_cold(z_hot):
    return np.argmax(z_hot, axis=1)

def discrete_partial_coot(source, target, test_source, test_target, **kwargs):
    prop_source = kwargs.get("prop_source", 0.1)
    prop_target = kwargs.get("prop_target", 0.1)
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20)
    source_levels = np.unique(source.Z)
    target_levels = np.unique(target.Z)

    nClass = len(np.union1d(source_levels, target_levels))
    categories = [np.arange(nClass)]

    encoder = onehot(
        handle_unknown="ignore", sparse_output=False, categories=categories
    )

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values
    z_source = source.Z.values.copy()
    z_target = target.Z.values.copy()

    n_source = len(z_source)
    n_target = len(z_target)

    l_source_train, l_source_test = train_test_split(
        np.arange(n_source), train_size=prop_source
    )
    l_target_train, l_target_test = train_test_split(
        np.arange(n_target), train_size=prop_target
    )

    z_source[l_source_test] = -1
    z_target[l_target_test] = -1

    x_source_train = source.loc[l_source_train, xcolumns(source)].values
    z_source_train = source.loc[l_source_train, "Z"].values

    x_target_train = target.loc[l_target_train, xcolumns(target)].values
    z_target_train = target.loc[l_target_train, "Z"].values

    z_source_test = source.loc[l_source_test, "Z"].values
    z_target_test = target.loc[l_target_test, "Z"].values

    def compute_cost_matrix(ys, yt, v=10000):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_(v))
        return M

    M_lin = compute_cost_matrix(yt=z_target, ys=z_source)

    Ts, Tv, cost = cot_numpy(
        X1=x_source,
        X2=x_target,
        niter=100,
        C_lin=M_lin,
        algo=algo,
        reg=reg,
        algo2="emd",
        verbose=False,
    )

    from tf_keras.utils import to_categorical
    zs_onehot = one_hot(z_source, nClass).astype(np.float64)
    zt_onehot_estimated = n_target * np.dot(Ts.T, zs_onehot)
    zt_onehot_estimated [l_target_train] = one_hot(z_target_train, nClass).astype(np.float64)
    zt_estimated = one_cold(zt_onehot_estimated)+ min(target_levels)
    M_lin = compute_cost_matrix(yt=z_source, ys=z_target)

    Ts, Tv, cost = cot_numpy(
        X1=x_target,
        X2=x_source,
        niter=100,
        C_lin=M_lin,
        algo="sinkhorn",
        reg=1,
        algo2="emd",
        verbose=False,
    )

    zt_onehot = one_hot(z_target, nClass).astype(np.float64)
    zs_onehot_estimated = n_source * np.dot(Ts.T, zt_onehot)
    zs_onehot_estimated[l_source_train] = one_hot(z_source_train, nClass).astype(np.float64)
    zs_estimated = one_cold(zs_onehot_estimated)+ min(source_levels)

    perf_pure_source = discrete_accuracy(z_source_test, zs_estimated[l_source_test])
    perf_pure_target = discrete_accuracy(z_target_test, zt_estimated[l_target_test])

    clf_source, clf_target = discrete_classifiers(source, target, "relu", "softmax")

    clf_source.fit(x_source, zs_onehot_estimated, batch_size=batch_size, epochs=20, verbose=0)
    clf_target.fit(x_target, zt_onehot_estimated, batch_size=batch_size, epochs=20, verbose=0)

    zt_test = one_cold(
        clf_target.predict(test_target.loc[:, xcolumns(test_target)], verbose=0)+ min(target_levels)
    )
    zs_test = one_cold(
        clf_source.predict(test_source.loc[:, xcolumns(test_source)], verbose=0)+ min(source_levels)
    )

    perf_test_source = discrete_accuracy(zs_test, test_source.Z)
    perf_test_target = discrete_accuracy(zt_test, test_target.Z)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
