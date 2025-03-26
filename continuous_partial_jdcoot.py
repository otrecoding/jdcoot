import math
import os
import sys
import numpy as np

sys.path.append(os.path.abspath('src'))

from jdcoot.jdcot.multitask_reg import jdcot_multitask_reg
from jdcoot.utils import xcolumns, continuous_classifiers, continuous_labels, continuous_accuracy

def continuous_partial_jdcoot( source, target, source_test, target_test):
    """
    jdcot multi-task for multi regression problems
    npreds : number of parameters to predict (it has to be the same number 
    for both datasets)
    yAtruth is (nA,npreds),yBtruth is (nB,npreds) : true value of 
    the parameters to estimate
    YA is (nA,npreds), YB is (nB,npreds) : line of 0 if non observed labels 
    and true values if observed labels (semi supervision)
    """

    prop_source = 0.1 
    prop_target = 0.1 
    alpha = 2.425
    
    l_source, l_target = continuous_labels(source, prop_source, target, prop_target)
    data_source = source.drop(columns = 'Z')
    data_source.loc[l_source, 'Y'] = np.nan
    data_target = target.drop(columns = 'Z')
    data_target.loc[l_target, 'Y'] = np.nan
    
    clf_source, clf_target = continuous_classifiers(source, target)
    
    xtrain_source = data_source.loc[:, xcolumns(source)].values
    ytrain_source = data_source.Y.values[:, np.newaxis]
    xtrain_target = data_source.loc[:, xcolumns(source)].values
    ytrain_target = data_target.Y.values[:, np.newaxis]

    model1, model2, results = jdcot_multitask_reg(modelA=clf_source, modelB=clf_target,
                                                  XA=xtrain_source,
                                                  YA=ytrain_source,
                                                  XB=xtrain_target,
                                                  YB=ytrain_target,
                                                  yAtruth=source.Y,
                                                  yBtruth=target.Y, 
                                                  reshape_data=False, algo='sinkhorn',
                                                  reg=100, alpha=alpha)
    
    zpred_target = model2.predict(data_target.loc[l_target, xcolumns(target)]).ravel()
    zpred_source = model1.predict(data_source.loc[l_source, xcolumns(source)]).ravel()

    perf_pure = continuous_accuracy(zpred_source, source.loc[l_source, 'Y'], zpred_target, target.loc[l_target, 'Y'])
    
    zt_test = model2.predict(target_test.loc[:, xcolumns(target_test)])[:, 0]
    zs_test = model1.predict(source_test.loc[:, xcolumns(source_test)])[:, 0]
    perf_test = continuous_accuracy(zt_test, target_test.Y, zs_test, source_test.Y)

    return perf_pure, perf_test


if __name__ == "__main__":

    from scenario import generate_data

    np.random.seed(2025)

    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    data = generate_data()

    perf_pure, perf_test = continuous_partial_jdcoot(*data)

    print(f"Pure Performance JDCOOT : {perf_pure}")
    print(f"Test Performance JDCOOT : {perf_test}")
