"""Classifier construction, covariate selection, and accuracy helpers."""

import numpy as np
import tf_keras
from tf_keras.layers import Dense


def xcolumns(df):
    """Return the names of observed covariate columns."""
    return [a for a in df.columns if "X" in a]


def discrete_classifier(data, f_in, f_out, nClass):
    """Build and compile a two-layer classifier for the given covariates."""
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
    loss = "categorical_crossentropy"
    clf = clf_seq(shape)
    clf.compile(optimizer="Adam", loss=loss, metrics=["accuracy"])
    return clf


def discrete_classifiers(source, target, f_in, f_out):
    """Build source and target classifiers with a common class count."""
    n_classes = len(np.union1d(np.unique(source.Z), np.unique(target.Z)))

    return discrete_classifier(source, f_in, f_out, n_classes), discrete_classifier(
        target, f_in, f_out, n_classes
    )


def discrete_accuracy_1(zpred, ztrue):
    """Return the fraction of correctly classified observations."""
    return np.mean(zpred == ztrue)


def discrete_accuracy_2(zpred_source, ztrue_source, zpred_target, ztrue_target):
    """Return sample-weighted accuracy across source and target domains."""
    return (sum(zpred_source == ztrue_source) + sum(zpred_target == ztrue_target)) / (
        len(zpred_source) + len(zpred_target)
    )


def discrete_accuracy(*args):
    """Compute accuracy for one or two domains, or NaN for other arities."""
    if len(args) == 2:
        return discrete_accuracy_1(*args)
    if len(args) == 4:
        return discrete_accuracy_2(*args)
    return np.nan
