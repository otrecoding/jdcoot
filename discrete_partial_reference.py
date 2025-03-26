import os
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot

sys.path.append(os.path.abspath('src'))

from jdcoot.utils import xcolumns, discrete_accuracy, discrete_classifiers
from sklearn.model_selection import train_test_split


def discrete_partial_reference( source, target, test_source, test_target) :

    source_levels = np.sort(np.unique(source.Z))
    target_levels = np.sort(np.unique(target.Z))

    nClass = len(np.union1d(source_levels, target_levels))
    categories=[np.arange(nClass)]

    enc = onehot(handle_unknown='ignore', sparse_output=False, categories=categories)

    source_train, source_test = train_test_split(source, test_size = 0.2, stratify = source.Z)
    target_train, target_test = train_test_split(target, test_size = 0.2, stratify = target.Z)

    x_source_train = source_train.loc[:, xcolumns(source)].values
    z_source_train = enc.fit_transform(source_train.Z.values[:, np.newaxis])

    x_target_train = target_train.loc[:, xcolumns(target)].values
    z_target_train = enc.fit_transform(target_train.Z.values[:, np.newaxis])

    x_source_test = source_test.loc[:, xcolumns(source)].values
    z_source_test = source_test.Z.values

    x_target_test = target_test.loc[:, xcolumns(target)].values
    z_target_test = target_test.Z.values

    clf_source, clf_target = discrete_classifiers(source, target, 'sigmoid', 'sigmoid')

    clf_target.fit( x_target_train, z_target_train, batch_size=10, epochs=20, verbose=0)  
    clf_source.fit( x_source_train, z_source_train, batch_size=10, epochs=20, verbose=0)  

    z_target_pred = enc.inverse_transform(clf_target.predict(x_target_test)).ravel()
    z_source_pred = enc.inverse_transform(clf_source.predict(x_source_test)).ravel()
    
    perf_pure = discrete_accuracy(z_target_pred, z_target_test, z_source_pred, z_source_test)

    x_test_source = test_source.loc[:, xcolumns(source)]
    x_test_target = test_target.loc[:, xcolumns(target)]

    z_test_source = enc.inverse_transform(clf_source.predict(x_test_source)).ravel()
    z_test_target = enc.inverse_transform(clf_target.predict(x_test_target)).ravel()
    
    perf_test = discrete_accuracy(z_test_target, test_target.Z, z_test_source, test_source.Z)
    

    return perf_pure, perf_test


if __name__ == "__main__":

    from scenario import generate_data

    data = generate_data()
    perf_pure, perf_test = discrete_partial_reference(*data)

    print(f"Pure Performance Reference : {perf_pure}")
    print(f"Test Performance Reference : {perf_test}")
