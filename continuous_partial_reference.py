import sys
import os
sys.path.append(os.path.abspath('src'))
from jdcoot.utils import xcolumns, continuous_classifiers, continuous_accuracy
from sklearn.model_selection import train_test_split


def continuous_partial_reference(source, target, source_test, target_test):

    prop_source = 0.1 
    prop_target = 0.1 
    
    x_source = source.loc[:, xcolumns(source)].values
    y_source = source.Y.values
    x_target = target.loc[:, xcolumns(target)].values
    y_target = target.Y.values

    x_source_train, x_source_test, y_source_train, y_source_test = train_test_split(x_source, y_source, test_size = prop_source)
    x_target_train, x_target_test, y_target_train, y_target_test = train_test_split(x_target, y_target, test_size = prop_target)

    clf_source, clf_target = continuous_classifiers(source, target)
    
    clf_target.fit(x_target_train, y_target_train, batch_size=10, epochs=20, verbose=0)  
    
    y_target_pred = clf_target.predict(x_target_test).ravel()
    
    clf_source.fit(x_source_train, y_source_train, batch_size=10, epochs=20, verbose=0)  
    
    y_source_pred = clf_source.predict(x_source_test).ravel()
    
    perf_pure = continuous_accuracy(y_source_pred, y_source_test, y_target_pred, y_target_test)
    
    ytest1 = clf_target.predict(target_test.loc[:, xcolumns(target)]).ravel()
    ytest2 = clf_source.predict(source_test.loc[:, xcolumns(source)]).ravel()
    
    perf_test = continuous_accuracy(ytest1, target_test.Y, ytest2, source_test.Y)

    return perf_pure, perf_test


if __name__ == "__main__":

    from scenario import generate_data

    data = generate_data()
    perf_pure, perf_test = continuous_partial_reference(*data)

    print(f"Pure Performance Reference : {perf_pure}")
    print(f"Test Performance Reference : {perf_test}")
