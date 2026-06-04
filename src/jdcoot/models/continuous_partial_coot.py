import numpy as np
import ot
from ..comp import comp_regression
from ..coot import cot_numpy
from ..utils import xcolumns, continuous_classifiers, continuous_accuracy


def continuous_partial_coot(
    source,
    target,
    source_test,
    target_test,
    l_source_train,
    l_source_test,
    l_target_train,
    l_target_test,
    **kwargs,
):
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

    ypred_target = clf_target.predict(x_target[l_target_test, :], verbose=0).ravel()
    ypred_source = clf_source.predict(x_source[l_source_test, :], verbose=0).ravel()

    perf_pure_source = continuous_accuracy(ypred_source, source.loc[l_source_test, "Y"])
    perf_pure_target = continuous_accuracy(ypred_target, target.loc[l_target_test, "Y"])

    zt_test = clf_target.predict(
        target_test.loc[:, xcolumns(target_test)], verbose=0
    ).ravel()
    zs_test = clf_source.predict(
        source_test.loc[:, xcolumns(source_test)], verbose=0
    ).ravel()

    perf_test_source = continuous_accuracy(zs_test, source_test.Y)
    perf_test_target = continuous_accuracy(zt_test, target_test.Y)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
