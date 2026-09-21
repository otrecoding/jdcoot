import math
import numpy as np
import tf_keras
from tf_keras.layers import Dense


def rmse(ypred, ytest):
    n = ytest.size
    assert ypred.size == ytest.size
    return math.sqrt(sum((ypred - ytest) ** 2) / n)


def xcolumns(df):
    return [a for a in df.columns if "X" in a]


def discrete_classifier(data, f_in, f_out, nClass):
    def clf_seq(shape):
        model = tf_keras.Sequential(
            [
                Dense(units=128, input_shape=shape, activation=f_in),
                Dense(units=nClass, activation=f_out),
            ]
        )
        return model

    fe_size = len(xcolumns(data))
    shape = (fe_size,)
    is_binary = nClass <= 2
    loss = "binary_crossentropy" if is_binary else "categorical_crossentropy"
    units_out = 1 if is_binary else nClass
    f_out_effective = "sigmoid" if is_binary else f_out

    clf = clf_seq(shape, units_out, f_out_effective)
    clf.compile(optimizer="Adam", loss=loss, metrics=["accuracy"])
    return clf


def discrete_classifiers(source, target, f_in, f_out):
    nClass = len(np.union1d(np.unique(source.Z), np.unique(target.Z)))

    return discrete_classifier(source, f_in, f_out, nClass), discrete_classifier(
        target, f_in, f_out, nClass
    )


def discrete_accuracy_1(zpred, ztrue):
    return np.mean(zpred == ztrue)


def discrete_accuracy_2(zpred_source, ztrue_source, zpred_target, ztrue_target):
    return (sum(zpred_source == ztrue_source) + sum(zpred_target == ztrue_target)) / (
        len(zpred_source) + len(zpred_target)
    )


def discrete_accuracy(*args):
    if len(args) == 2:
        return discrete_accuracy_1(*args)
    elif len(args) == 4:
        return discrete_accuracy_2(*args)
    else:
        return np.nan
