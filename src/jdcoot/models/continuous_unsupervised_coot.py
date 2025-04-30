import numpy as np
import ot
from ..comp import comp_regression
from ..coot import cot_numpy
from ..utils import xcolumns, continuous_accuracy, continuous_classifier

    
def continuous_unsupervised_coot( source, target, source_test, target_test, **kwargs) :

    x_source = source.loc[:, xcolumns(source)]
    x_target = target.loc[:, xcolumns(target)]

    y_source = source.Y.values
    y_target = target.Y.values
    
    ytrain_source = y_source
    ytrain_target = np.full_like(y_target, np.nan)
    
    def compute_cost_matrix(ys, yt):
        return ot.dist(ys[:, np.newaxis], yt[:, np.newaxis], metric=comp_regression())  
    
    M_lin = compute_cost_matrix(yt=ytrain_target, ys=ytrain_source)

    Ts, Tv, cost = cot_numpy(X1=x_source, X2=x_target, niter=100, 
                             C_lin=M_lin,
                             algo='sinkhorn', reg=1,
                             algo2='emd', verbose=False)
    
    ytrue = y_target
    ypred = y_target.size * np.dot(Ts.T, y_source)
    
    perf_pure = continuous_accuracy(ypred, ytrue)
    
    clf = continuous_classifier(target)
    
    clf.fit(x_target, ypred, batch_size=10, epochs=10, verbose=0) 
    xtest = target_test.loc[:, xcolumns(target)].values
    ytest = clf.predict(xtest, verbose=0).ravel()
    perf_test = continuous_accuracy(ytest, target_test.Y.values)

    return perf_pure, perf_test


