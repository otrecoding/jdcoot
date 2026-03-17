import numpy as np
import ot

from ..coot import cot_numpy
from ..comp import comp_regression
from ..utils import xcolumns, continuous_classifier, continuous_accuracy
from sklearn.model_selection import train_test_split


def continuous_semisupervised_coot(source, target, source_test, target_test,  **kwargs):
    prop_target = kwargs.get("prop_target", 0.1)
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20)

    y_source = source.Y.values
    y_target = target.Y.values

    n_target = len(y_target)

    l_target_train, l_target_test = train_test_split(
        np.arange(n_target), train_size=prop_target
    )

    ytrain_target = target.Y.values.copy()
    ytrain_target[l_target_test] = np.nan

    def compute_cost_matrix(ys, yt):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_regression())
        return M

    M_lin = compute_cost_matrix(yt=ytrain_target, ys=y_source)

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

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

    y_target_pred = n_target * np.dot(Ts.T, y_source)
    y_target_pred[l_train] = y_target_train
    perf_pure_source = 0.0
    perf_pure_target = continuous_accuracy(
        y_target_pred[l_target_test], y_target[l_target_test]
    )

    clf_target = continuous_classifier(target)

    clf_target.fit(x_target, y_target_pred, batch_size=batch_size, epochs=10, verbose=0)

    y_target_test_pred = clf_target.predict(
        target_test.loc[:, xcolumns(target_test)], verbose=0
    ).ravel()

    perf_test_source = 0.0
    perf_test_target = continuous_accuracy(y_target_test_pred, target_test.Y)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
