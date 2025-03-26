import os
import sys
sys.path.append(os.path.abspath('src'))
import numpy as np
from jdcoot.jdcot.multitask_reg import jdcot_multitask_reg
from jdcoot.utils import xcolumns, continuous_accuracy, continuous_classifiers


def continuous_unsupervised_jdcoot( source, target, test_source, test_target):

    alpha = 0.3
    
    xtrain_source = source.loc[:, xcolumns(source)].values
    xtrain_target = target.loc[:, xcolumns(target)].values
    
    ytrain_source = source.Y.values[:, np.newaxis]
    ytrain_target = np.full_like(target.Y.values, np.nan)[:, np.newaxis]
    
    clf_source, clf_target = continuous_classifiers( source, target ) 
    
    model_source, model_target, results = jdcot_multitask_reg(
                                                  modelA=clf_source, 
                                                  modelB=clf_target,
                                                  XA = xtrain_source,
                                                  YA = ytrain_source,
                                                  XB = xtrain_target,
                                                  YB = ytrain_target,
                                                  yAtruth=source.Y,
                                                  yBtruth=target.Y, 
                                                  reshape_data=False, 
                                                  algo='sinkhorn',
                                                  reg=100, alpha=alpha)
    
    zpred_target = model_target.predict(xtrain_target).ravel()
    
    perf_pure = continuous_accuracy(zpred_target, target.Y)
    
    xsource = test_source.loc[:, xcolumns(test_source)].values
    xtarget = test_target.loc[:, xcolumns(test_target)].values

    ytarget = model_target.predict(xtarget).ravel()
    ysource = model_source.predict(xsource).ravel()
    
    perf_test = continuous_accuracy(ytarget, test_target.Y, ysource, test_source.Y)

    return perf_pure, perf_test


if __name__ == "__main__":

    from scenario import generate_data
    
    data = generate_data()

    perf_pure, perf_test = continuous_unsupervised_jdcoot( *data )

    print(f"Pure Performance JDCOOT : {perf_pure}")
    print(f"Test Performance JDCOOT : {perf_test}")
