import os
import sys
sys.path.append(os.path.abspath('src'))

import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.model_selection import train_test_split

from jdcoot.utils import xcolumns, discrete_classifier


def discrete_semisupervised_reference( source, target, source_test, target_test):

     prop_target = 0.1
     z_target = target.Z.values
     n_target = len(target.Z)

     l_train, l_test = train_test_split(np.arange(n_target), test_size = prop_target, stratify = z_target)

     xtrain_target = target.loc[l_train, xcolumns(target)].values
     ztrain_target = target.Z.values[l_train]

     xtest_target = target.loc[l_test, xcolumns(target)].values
     ztest_target = target.Z.values[l_test]
     
     nClass=len(np.union1d(np.unique(source.Z), np.unique(target.Z)))
     categories=[np.arange(nClass)]

     clf = discrete_classifier(target, 'sigmoid', 'sigmoid', nClass)

     enc = onehot(handle_unknown='ignore', sparse_output = False, categories=categories)
     
     clf.fit( xtrain_target, enc.fit_transform(ztrain_target[:,np.newaxis]),
              batch_size=10, epochs=20, verbose=0)  
     
     z_test = clf.predict(target_test.loc[:, xcolumns(target)])
     z_test = enc.inverse_transform(z_test).ravel()
     perf_test = np.mean(z_test == target_test.Z)
     
     z_test = clf.predict(xtest_target)
     z_test = enc.inverse_transform(z_test).ravel()
     perf_pure = np.mean(z_test == ztest_target)

     return perf_pure, perf_test


if __name__ == '__main__':

    from scenario import generate_data

    data = generate_data()

    perf_pure, perf_test = discrete_semisupervised_reference( *data ) 

    print(f"Pure Performance Reference : {perf_pure}")
    print(f"Test Performance Reference : {perf_test}")
