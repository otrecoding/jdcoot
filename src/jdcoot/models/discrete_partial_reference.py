import numpy as np
from tf_keras.utils import to_categorical
from ..utils import discrete_classifier, xcolumns, discrete_accuracy

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

def discrete_partial_reference(source, target, test_source, test_target, l_source_train, 
                               l_source_test, l_target_train, l_target_test, **kwargs):
    
    batch_size = kwargs.get("batch_size", 20) 

    source_train = source.iloc[l_source_train, :]
    target_train = target.iloc[l_target_train, :]
    source_test = source.iloc[l_source_test, :]
    target_test = target.iloc[l_target_test, :]

    x_source_train = source_train.loc[:, xcolumns(source)].values

    source_levels_train = np.sort(np.unique(source_train.Z))
    target_levels_train = np.sort(np.unique(target_train.Z))

    z_source_train = one_hot(source_train.Z.values[:, np.newaxis], source_levels_train).astype(np.float64)

    x_target_train = target_train.loc[:, xcolumns(target)].values
    z_target_train = one_hot(target_train.Z.values[:, np.newaxis], target_levels_train).astype(np.float64)

    x_source_test = source_test.loc[:, xcolumns(source)].values
    z_source_test = source_test.Z.values

    x_target_test = target_test.loc[:, xcolumns(target)].values
    z_target_test = target_test.Z.values

    clf_source = discrete_classifier(source, "relu", "softmax", len(source_levels_train))
    clf_target = discrete_classifier(target, "relu", "softmax", len(target_levels_train))
    clf_target.fit(x_target_train, z_target_train, batch_size=batch_size, epochs=10, verbose=0)
    clf_source.fit(x_source_train, z_source_train, batch_size=batch_size, epochs=10, verbose=0)

    z_target_pred = one_cold(
        clf_target.predict(x_target_test, verbose=0),target_levels_train) 
    z_source_pred = one_cold(
        clf_source.predict(x_source_test, verbose=0),source_levels_train) 

    perf_pure_source = discrete_accuracy(z_source_pred, z_source_test)
    perf_pure_target = discrete_accuracy(z_target_pred, z_target_test)

    x_test_source = test_source.loc[:, xcolumns(source)]
    x_test_target = test_target.loc[:, xcolumns(target)]

    z_test_source = one_cold(
        clf_source.predict(x_test_source, verbose=0),source_levels_train) 
    z_test_target = one_cold(
        clf_target.predict(x_test_target, verbose=0),target_levels_train) 
  
    perf_test_source = discrete_accuracy(z_test_source, test_source.Z)
    perf_test_target = discrete_accuracy(z_test_target, test_target.Z)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
