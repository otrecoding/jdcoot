from ..utils import xcolumns, continuous_accuracy, continuous_classifier


def continuous_semisupervised_reference(
    source,
    target,
    source_test,
    target_test,
    l_source_train,
    l_source_test,
    l_target_train,
    l_target_test,
    **kwargs,
):
    batch_size = kwargs.get("batch_size", 20)

    x_target = target.loc[:, xcolumns(target)].values

    y_target = target.Y.values

    xtrain_target = x_target[l_target_train, :]
    ytrain_target = y_target[l_target_train]

    xtest_target = x_target[l_target_test, :]
    ytest_target = y_target[l_target_test]

    clf = continuous_classifier(target)

    clf.fit(xtrain_target, ytrain_target, batch_size=batch_size, epochs=10, verbose=0)

    z_test = clf.predict(xtest_target, verbose=0).ravel()

    perf_pure_source = 0.0
    perf_pure_target = continuous_accuracy(z_test, ytest_target)

    x_target_test = target_test.loc[:, xcolumns(target)].values
    y_target_test = target_test.Y.values

    z_test = clf.predict(x_target_test, verbose=0).ravel()

    perf_test_source = 0.0
    perf_test_target = continuous_accuracy(z_test, y_target_test)

    return perf_pure_source, perf_pure_target, perf_test_source, perf_test_target
