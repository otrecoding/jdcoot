import math
import numpy as np
import tf_keras
from tf_keras.layers import Dense

def continuous_labels(source, prop_source, target, prop_target):

    size_s = source.Y.size
    size_t = target.Y.size
    
    l_source = np.full(size_s, True)
    l_target = np.full(size_t, True)
    
    l_source[np.random.choice(np.arange(size_s), math.ceil(prop_source * size_s), replace=False)] = False
    l_target[np.random.choice(np.arange(size_t), math.ceil(prop_target * size_t), replace=False)] = False

    return l_source, l_target


def rmse(ypred, ytest):
    n = ytest.size
    assert ypred.size == ytest.size
    return math.sqrt(sum((ypred - ytest) ** 2) / n)


def xcolumns(df):
    return [a for a in df.columns if 'X' in a]


def continuous_classifier( data ):

    def clf_seq(shape):
        model = tf_keras.Sequential([Dense(units=128, input_shape=shape, 
                                     activation='linear'),
                                     Dense(units=1, activation='linear')])
        return model

    fe_size = len(xcolumns(data))
    shape = (fe_size,)
    loss = 'MeanSquaredError'
    clf = clf_seq(shape)
    clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

    return clf
    
def continuous_classifiers( source, target):

    return continuous_classifier(source), continuous_classifier(target)

def continuous_accuracy_2( ypred_source, ytrue_source, ypred_target, ytrue_target):

    return (sum((ypred_source - ytrue_source) ** 2) 
          + sum((ypred_target - ytrue_target) ** 2)) / (len(ypred_source) + len(ytrue_target))

def continuous_accuracy_1( ypred, ytrue):
    assert len(ypred) == len(ytrue)
    return sum((ypred - ytrue) ** 2) / len(ypred)

def continuous_accuracy(*args):

    if len(args)==2:
        return continuous_accuracy_1(args[0], args[1])
    elif len(args)==4:
        return continuous_accuracy_2(args[0], args[1], args[2], args[3])
    else:
        return np.nan


def discrete_classifier(data, f_in, f_out, nClass):

    def clf_seq(shape):
        model = tf_keras.Sequential([
          Dense(units=128, input_shape=shape, activation=f_in),
          Dense(units=nClass, activation=f_out)])
        return model

    fe_size = len(xcolumns(data))
    shape = (fe_size,)
    loss = 'categorical_crossentropy'
    clf = clf_seq(shape)
    clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    return clf

def discrete_classifiers(source, target, f_in, f_out):

    nClass=len(np.union1d(np.unique(source.Z), np.unique(target.Z)))

    return discrete_classifier(source, f_in, f_out, nClass), discrete_classifier(target, f_in, f_out, nClass)

def discrete_accuracy_1( zpred, ztrue):
    return np.mean(zpred == ztrue)

def discrete_accuracy_2( zpred_source, ztrue_source, zpred_target, ztrue_target):

    return (sum(zpred_source == ztrue_source) 
          + sum(zpred_target == ztrue_target)) / (len(zpred_source) + len(zpred_target))

def discrete_accuracy(*args):

    if len(args) == 2:
        return discrete_accuracy_1(*args)
    elif len(args) == 4:
        return discrete_accuracy_2(*args)
    else:
        return np.nan
    

