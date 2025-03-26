import os
import sys
sys.path.append(os.path.abspath('src'))

import numpy as np

from jdcoot.jdcot.multitask_reg import jdcot_multitask_reg
from jdcoot.utils import continuous_accuracy, xcolumns, continuous_classifiers
from sklearn.model_selection import train_test_split


def continuous_semisupervised_jdcoot( source, target, source_test, target_test ):

    prop_target = 0.1 
    alpha = 2.625

    n_target = len(target.Y)
    
    l_target_train, l_target_test = train_test_split(np.arange(n_target), test_size = prop_target)

    x_source = source.loc[:, xcolumns(source)].values
    x_target = target.loc[:, xcolumns(target)].values
    y_source = source.Y.values
    y_target = target.Y.values

    ytrain_target = np.copy(y_target)
    ytrain_target[l_target_test] = np.nan
    
    clf_source, clf_target = continuous_classifiers(source, target)
    
    model1, model2, results = jdcot_multitask_reg(modelA=clf_source, 
                                                  modelB=clf_target,
                                                  XA=x_source,
                                                  YA=y_source.reshape((-1, 1)),
                                                  XB=x_target,
                                                  YB=ytrain_target.reshape((-1, 1)),
                                                  yAtruth=y_source,
                                                  yBtruth=y_target, 
                                                  reshape_data=False, 
                                                  algo='sinkhorn',
                                                  reg=100, 
                                                  alpha=alpha)
    
    xtest_target = x_target[l_target_test,:]
    ytest_target = y_target[l_target_test]

    ypred_target = model2.predict(xtest_target).ravel()
    perf_pure = continuous_accuracy(ypred_target, ytest_target)
    
    zt_test = model2.predict(target_test.loc[:, xcolumns(target_test)]).ravel()
    zs_test = model1.predict(source_test.loc[:, xcolumns(source_test)]).ravel()

    perf_test = continuous_accuracy(zs_test, source_test.Y, zt_test, target_test.Y) 

    return perf_pure, perf_test
    


if __name__ == '__main__':

    from scenario import generate_data

    data = generate_data()
    perf_pure, perf_test = continuous_semisupervised_jdcoot( *data )

    print(f"Pure Performance JDCOOT : {perf_pure}")
    print(f"Test Performance JDCOOT : {perf_test}")
