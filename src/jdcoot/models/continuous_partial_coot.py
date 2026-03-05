import numpy as np
import ot

from ..comp import comp_regression
from ..coot import cot_numpy
from ..utils import xcolumns, continuous_classifiers, continuous_accuracy
from sklearn.model_selection import train_test_split


def continuous_partial_coot(source, target, source_test, target_test, algo,reg, **kwargs):
    prop_source = kwargs.get("prop_source", 0.1)
    prop_target = kwargs.get("prop_target", 0.1)
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20)

    n_source = source.Y.size
    n_target = target.Y.size

    l_source_train, l_source_test = train_test_split(
        np.arange(n_source), train_size=prop_source
    )
    l_target_train, l_target_test = train_test_split(
        np.arange(n_target), train_size=prop_target
    )

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    x_source_train = x_source[l_source_train, :]
    x_target_train = x_target[l_target_train, :]

    y_source = source.Y.values
    y_target = target.Y.values

    y_source_train = source.Y.values.copy()
    y_source_train[l_source_test] = np.nan
    y_target_train = target.Y.values.copy()
    y_target_train[l_target_test] = np.nan

    y_source_test = source.Y.values[l_source_test]
    y_target_test = target.Y.values[l_target_test]

    def compute_cost_matrix(ys, yt):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_regression())
        return M

    M_lin = compute_cost_matrix(yt=y_target_train, ys=y_source[l_source_train])

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

    zt_estimated = n_target * np.dot(Ts.T, y_source[l_source_train])

    M_lin = compute_cost_matrix(yt=y_source_train, ys=y_target[l_target_train])

    Ts, Tv, cost = cot_numpy(
        X1=x_target_train,
        X2=x_source,
        niter=100,
        C_lin=M_lin,
        algo="emd",
        reg=1,
        algo2="emd",
        verbose=False,
    )

    zs_estimated = n_source * np.dot(Ts.T, y_target[l_target_train])

    perf_pure_target = continuous_accuracy(y_target_test, zt_estimated[l_target_test])
    perf_pure_source = continuous_accuracy(y_source_test, zs_estimated[l_source_test])

    clf_source, clf_target = continuous_classifiers(source, target)

    clf_target.fit(x_target, zt_estimated, batch_size=batch_size, epochs=10, verbose=0)
    clf_source.fit(x_source, zs_estimated, batch_size=batch_size, epochs=10, verbose=0)

    zt_test = clf_target.predict(
        target_test.loc[:, xcolumns(target_test)], verbose=0
    ).ravel()
    zs_test = clf_source.predict(
        source_test.loc[:, xcolumns(source_test)], verbose=0
    ).ravel()

    perf_test_source = continuous_accuracy(zs_test, source_test.Y)
    perf_test_target = continuous_accuracy(zt_test, target_test.Y)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
