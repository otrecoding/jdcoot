import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.model_selection import train_test_split
from tf_keras.utils import to_categorical
from ..utils import xcolumns, discrete_classifier, discrete_accuracy


def one_hot(z, seen_levels):
    """
    One-hot encoding sur les classes vues dans ce fold.

    z : array-like, labels (float ou int)
    seen_levels : array-like, original labels present
    """
    z = np.array(z)
    seen_levels = np.array(seen_levels)
    indices = np.searchsorted(seen_levels, z)
    return to_categorical(indices, num_classes=len(seen_levels))

def one_cold(z_hot, seen_levels):
    indices = np.argmax(z_hot, axis=1)
    seen_levels = np.array(seen_levels)
    return seen_levels[indices]

def discrete_semisupervised_reference(
    source, target, source_test, target_test,
    l_source_train, l_source_test, l_train, l_test, **kwargs
):

    batch_size = kwargs.get("batch_size", 20)
    n_target = len(target.Z)

    xtrain_target = target.loc[l_train, xcolumns(target)].values
    ztrain_target = target.Z.values[l_train]

    xtest_target = target.loc[l_test, xcolumns(target)].values
    ztest_target = target.Z.values[l_test]

    nClass = len(np.union1d(np.unique(source.Z), np.unique(target.Z)))
    categories = [np.arange(nClass)]

    enc = onehot(handle_unknown="ignore", sparse_output=False, categories=categories)
    z_target_train = one_hot(ztrain_target[:, np.newaxis], np.sort(np.unique(ztrain_target))).astype(np.float64)
    target_levels_train = np.sort(np.unique(ztrain_target))
    clf = discrete_classifier(target, "relu", "softmax", len(target_levels_train))
    clf.fit(
        xtrain_target,
        z_target_train,
        batch_size=batch_size,
        epochs=10,
        verbose=0,
    )

    z_test = clf.predict(target_test.loc[:, xcolumns(target)], verbose=0)
    z_test = one_cold(
        clf.predict(target_test.loc[:, xcolumns(target)], verbose=0),target_levels_train)


    perf_pure_source = 1.0
    perf_pure_target = discrete_accuracy(z_test, target_test.Z)

    z_test = clf.predict(xtest_target, verbose=0)
    z_test = one_cold(z_test,target_levels_train)
    
    perf_test_source = 1.0
    perf_test_target = discrete_accuracy(z_test, ztest_target)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
