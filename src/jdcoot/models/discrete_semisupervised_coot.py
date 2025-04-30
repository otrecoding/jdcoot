import numpy as np
import ot
from tf_keras.utils import to_categorical

from ..comp import comp_
from ..coot import cot_numpy
from ..utils import xcolumns, discrete_classifier
from sklearn.model_selection import train_test_split


def discrete_semisupervised_coot( source, target, source_test, target_test, **kwargs):

    prop_target = kwargs.get('prop_target', 0.1)

    source_levels = np.sort(np.unique(source.Z))
    target_levels = np.sort(np.unique(target.Z))

    categories = np.union1d(source_levels, target_levels)
    nClass = len(categories)

    def one_hot(z):
        return to_categorical(z, num_classes=nClass)

    def one_cold(z):
        return np.argmax(z, axis=1) + min(categories)

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    z_source = source.Z.values
    z_target = target.Z.values

    n_target = len(z_target)

    l_target_train, l_target_test = train_test_split(np.arange(n_target), 
                                                     test_size = prop_target, 
                                                     stratify = z_target)

    z_target_train = z_target.copy()
    z_target_train[l_target_test] = -1

    def compute_cost_matrix(ys, yt, v=10000):
        M = ot.dist(ys.reshape(-1, 1), yt.reshape(-1, 1), metric=comp_(v))
        return M
    
    M_lin = compute_cost_matrix(yt=z_target_train, ys=z_source)
    
    Ts, Tv, cost = cot_numpy(X1=x_source,
                             X2=x_target,
                             niter=100, C_lin=M_lin,
                             algo='sinkhorn', reg=1,
                             algo2='emd', verbose=False)
    
    z_target_pred = n_target * np.dot(Ts.T, one_hot(z_source))

    clf = discrete_classifier(target, 'sigmoid', 'sigmoid', nClass)
    
    clf.fit(x_target, z_target_pred, batch_size=10, epochs=10, verbose=0) 

    z_target_test = target.loc[l_target_test, 'Z'].values

    z_target_pred = one_cold(z_target_pred)

    perf_pure = np.mean(z_target_test == z_target_pred[l_target_test])
    
    z_test = clf.predict(target_test.loc[:, xcolumns(target)], verbose=0)
    perf_test = np.mean(one_cold(z_test) == target_test.Z) 

    return perf_pure, perf_test

