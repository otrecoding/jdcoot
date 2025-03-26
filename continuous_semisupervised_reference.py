import os
import sys
sys.path.append(os.path.abspath('src'))

import numpy as np

from jdcoot.utils import xcolumns, continuous_accuracy, continuous_classifier
from sklearn.model_selection import train_test_split

def continuous_semisupervised_reference( source, target, source_test, target_test):

    prop_target = 0.1 
    
    x_target = target.loc[:, xcolumns(target)].values

    y_target = target.Y.values

    n_target = len(y_target)

    l_target_train, l_target_test = train_test_split(np.arange(n_target), test_size = prop_target)

    xtrain_target = x_target[l_target_train, :]
    ytrain_target = y_target[l_target_train]

    xtest_target = x_target[l_target_test, :]
    ytest_target = y_target[l_target_test]

    clf = continuous_classifier(target)

    clf.fit(xtrain_target, ytrain_target, batch_size=10, epochs=20, verbose=0) 

    z_test = clf.predict(xtest_target).ravel()
    perf_ref1 = continuous_accuracy(z_test, ytest_target)
    
    x_target_test = target_test.loc[:, xcolumns(target)].values
    y_target_test = target_test.Y.values

    z_test = clf.predict(x_target_test).ravel()
    perf_ref2 = continuous_accuracy(z_test, y_target_test)
    
    return perf_ref1, perf_ref2


if __name__ == "__main__":

    from scenario import generate_data

    data = generate_data()

    pure, test = continuous_semisupervised_reference(*data)

    print(f"Pure Performance Reference : {pure}")
    print(f"Test Performance Reference : {test}")
