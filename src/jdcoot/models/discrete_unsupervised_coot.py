import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot

from ..coot import cot_numpy
from ..utils import discrete_classifier, xcolumns

def discrete_unsupervised_coot( source, target, source_test, target_test, **kwargs):

    source_levels = np.unique(source.Z)
    target_levels = np.unique(source.Z)
    
    size_target = target.Z.size

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values

    z_source = source.Z.values
    z_target = target.Z.values
    
    nClass=len(np.union1d(source_levels, target_levels))
    categories=[np.arange(nClass)]

    encoder = onehot(handle_unknown='ignore', sparse_output = False, categories=categories)

    def one_hot(z):
        return encoder.fit_transform(z.reshape(-1, 1))

    def one_cold(z):
        return encoder.inverse_transform(z).ravel()
    
    Ts, Tv, cost = cot_numpy(X1=x_source, X2=x_target, niter=100,
                             algo='sinkhorn', reg=1, algo2='emd', verbose=False)
    
    z_target_pred = size_target * np.dot(Ts.T, one_hot(z_source))
    
    perf_pure = np.mean(z_target == one_cold(z_target_pred))
    
    clf = discrete_classifier(target, 'sigmoid', 'sigmoid', nClass)

    clf.fit(x_target, z_target_pred, batch_size=10, epochs=20, verbose=0)  

    z_test = clf.predict(target_test.loc[:, xcolumns(target_test)], verbose=0)

    perf_test = np.mean(one_cold(z_test) == target_test.Z)

    return perf_pure, perf_test


