from ..utils import xcolumns, continuous_classifiers, continuous_accuracy


def continuous_partial_reference(source, target, test_source, test_target, l_source_train, l_source_test, l_target_train, l_target_test, **kwargs):

    batch_size = kwargs.get("batch_size", 20)

    x_source = source.loc[:, xcolumns(source)].values
    y_source = source.Y.values
    x_target = target.loc[:, xcolumns(target)].values
    y_target = target.Y.values

    x_target_train = x_target[l_target_train, :]
    y_target_train = y_target[l_target_train, :]
    x_source_train = x_source[l_source_train, :]
    y_source_train = y_source[l_source_train, :]

    x_target_test = x_target[l_target_test, :]
    y_target_test = y_target[l_target_test, :]
    x_source_test = x_source[l_source_test, :]
    y_source_test = y_source[l_source_test, :]

    clf_source, clf_target = continuous_classifiers(source, target)

    clf_target.fit(x_target_train, y_target_train, batch_size=batch_size, epochs=20, verbose=0)

    y_target_pred = clf_target.predict(x_target_test, verbose=0).ravel()

    clf_source.fit(x_source_train, y_source_train, batch_size=batch_size, epochs=20, verbose=0)

    y_source_pred = clf_source.predict(x_source_test, verbose=0).ravel()

    perf_pure_source = continuous_accuracy(y_source_pred, y_source_test)
    perf_pure_target = continuous_accuracy(y_target_pred, y_target_test)

    ytest1 = clf_target.predict(test_target.loc[:, xcolumns(target)], verbose=0).ravel()
    ytest2 = clf_source.predict(test_source.loc[:, xcolumns(source)], verbose=0).ravel()

    perf_test_source = continuous_accuracy(ytest2, test_source.Y)
    perf_test_target = continuous_accuracy(ytest1, test_target.Y)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
