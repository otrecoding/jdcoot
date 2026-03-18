import numpy as np
import ot
from tf_keras.utils import to_categorical

from ..comp import comp_
from ..coot import cot_numpy
from ..utils import xcolumns, discrete_classifier, discrete_accuracy
from sklearn.model_selection import train_test_split
from tf_keras.utils import to_categorical

def one_hot(z, nClass):
    return to_categorical(z, num_classes=nClass)


def one_cold(z_hot):
    return np.argmax(z_hot, axis=1)

def discrete_semisupervised_coot(source, target, source_test, target_test,l_target_train, l_target_test, **kwargs):
    #prop_target = kwargs.get("prop_target", 0.1)
    algo = kwargs.get("algo", "emd")
    reg = kwargs.get("reg", 1)
    batch_size = kwargs.get("batch_size", 20) 

    source_levels = np.sort(np.unique(source.Z))
    target_levels = np.sort(np.unique(target.Z))

    categories = np.union1d(source_levels, target_levels)
    nClass = len(categories)

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    z_source = source.Z.values
    z_target = target.Z.values

    n_target = len(z_target)

    #l_target_train, l_target_test = train_test_split(
    #    np.arange(n_target), train_size=prop_target
    #)

    z_target_train = z_target.copy()
    z_target_train[l_target_test] = -1

    def compute_cost_matrix(ys, yt, v=10000):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_(v))
        return M

    M_lin = compute_cost_matrix(yt=z_target_train, ys=z_source)

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

    z_target_pred = n_target * np.dot(Ts.T, one_hot(z_source, nClass).astype(np.float64))
    z_target_pred[l_target_train] = one_hot(z_target[l_target_train], nClass).astype(np.float64)
    clf = discrete_classifier(target, "relu", "softmax", nClass)

    clf.fit(x_target, z_target_pred, batch_size=batch_size, epochs=10, verbose=0)

    z_target_test = target.loc[l_target_test, "Z"].values

    z_target_pred = one_cold(z_target_pred)+ min(target_levels)

    perf_pure_source = 1.0
    perf_pure_target = discrete_accuracy(z_target_test, z_target_pred[l_target_test])

    z_test = clf.predict(target_test.loc[:, xcolumns(target)], verbose=0)

    perf_test_source = 1.0
    perf_test_target = discrete_accuracy(one_cold(z_test)+ min(target_levels), target_test.Z)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
