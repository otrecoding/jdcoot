from ..utils import xcolumns, continuous_classifiers, continuous_accuracy
from sklearn.model_selection import train_test_split


def continuous_partial_reference(source, target, source_test, target_test, **kwargs):

    prop_source = kwargs.get('prop_source', 0.1)
    prop_target = kwargs.get('prop_target', 0.1)
    
    x_source = source.loc[:, xcolumns(source)].values
    y_source = source.Y.values
    x_target = target.loc[:, xcolumns(target)].values
    y_target = target.Y.values

    x_source_train, x_source_test, y_source_train, y_source_test = train_test_split(x_source, y_source, train_size = prop_source)
    x_target_train, x_target_test, y_target_train, y_target_test = train_test_split(x_target, y_target, train_size = prop_target)

    clf_source, clf_target = continuous_classifiers(source, target)
    
    clf_target.fit(x_target_train, y_target_train, batch_size=10, epochs=20, verbose=0)  
    
    # y_target_pred = clf_target.predict(x_target_test, verbose=0).ravel()
    
    clf_source.fit(x_source_train, y_source_train, batch_size=10, epochs=20, verbose=0)  
    
    # y_source_pred = clf_source.predict(x_source_test, verbose=0).ravel()
    
    # perf_pure = continuous_accuracy(y_source_pred, y_source_test, y_target_pred, y_target_test)
    
    ytest1 = clf_target.predict(target_test.loc[:, xcolumns(target)], verbose=0).ravel()
    ytest2 = clf_source.predict(source_test.loc[:, xcolumns(source)], verbose=0).ravel()

    perf_source = continuous_accuracy(ytest2, source_test.Y)
    perf_target = continuous_accuracy(ytest1, target_test.Y)

    return perf_source, perf_target
