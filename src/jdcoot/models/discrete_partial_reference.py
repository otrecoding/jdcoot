import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot

from ..utils import xcolumns, discrete_accuracy, discrete_classifiers
from sklearn.model_selection import train_test_split


def discrete_partial_reference(source, target, test_source, test_target,l_source_train, l_source_test,l_target_train, l_target_test, **kwargs):
    #prop_source = kwargs.get("prop_source", 0.1)
    #prop_target = kwargs.get("prop_target", 0.1)
    batch_size = kwargs.get("batch_size", 20) 
    source_levels = np.sort(np.unique(source.Z))
    target_levels = np.sort(np.unique(target.Z))

    nClass = len(np.union1d(source_levels, target_levels))
    categories = [np.arange(nClass)]

    enc = onehot(handle_unknown="ignore", sparse_output=False, categories=categories)

    #source_train, source_test = train_test_split(source, train_size=prop_source)
    #target_train, target_test = train_test_split(target, train_size=prop_target)
    source_train = source.iloc[l_source_train, :]
    target_train = target.iloc[l_target_train, :]
    source_test = source.iloc[l_source_test, :]
    target_test = target.iloc[l_target_test, :]

    x_source_train = source_train.loc[:, xcolumns(source)].values
    z_source_train = enc.fit_transform(source_train.Z.values[:, np.newaxis])

    x_target_train = target_train.loc[:, xcolumns(target)].values
    z_target_train = enc.fit_transform(target_train.Z.values[:, np.newaxis])

    x_source_test = source_test.loc[:, xcolumns(source)].values
    z_source_test = source_test.Z.values

    x_target_test = target_test.loc[:, xcolumns(target)].values
    z_target_test = target_test.Z.values

    clf_source, clf_target = discrete_classifiers(source, target, "relu", "softmax")

    clf_target.fit(x_target_train, z_target_train, batch_size=batch_size, epochs=10, verbose=0)
    clf_source.fit(x_source_train, z_source_train, batch_size=batch_size, epochs=10, verbose=0)

    z_target_pred = enc.inverse_transform(
        clf_target.predict(x_target_test, verbose=0)
    ).ravel()
    z_source_pred = enc.inverse_transform(
        clf_source.predict(x_source_test, verbose=0)
    ).ravel()

    perf_pure_source = discrete_accuracy(z_source_pred, z_source_test)
    perf_pure_target = discrete_accuracy(z_target_pred, z_target_test)

    x_test_source = test_source.loc[:, xcolumns(source)]
    x_test_target = test_target.loc[:, xcolumns(target)]

    z_test_source = enc.inverse_transform(
        clf_source.predict(x_test_source, verbose=0)
    ).ravel()
    z_test_target = enc.inverse_transform(
        clf_target.predict(x_test_target, verbose=0)
    ).ravel()

    perf_test_source = discrete_accuracy(z_test_source, test_source.Z)
    perf_test_target = discrete_accuracy(z_test_target, test_target.Z)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
