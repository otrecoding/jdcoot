import numpy as np
from sklearn.model_selection import train_test_split

from jdcoot import discrete_partial_coot
from jdcoot import discrete_partial_jdcoot
from jdcoot import discrete_partial_reference
from jdcoot import discrete_semisupervised_coot
from jdcoot import discrete_semisupervised_jdcoot
from jdcoot import discrete_semisupervised_reference
from jdcoot import discrete_unsupervised_coot
from jdcoot import discrete_unsupervised_jdcoot
from jdcoot import generate_data

prop_source, prop_target = 0.1, 0.1
data = generate_data(size=500)
source, target = data[0], data[1]
n_source = len(source.Z)
n_target = len(target.Z)
l_source = train_test_split(np.arange(n_source), train_size=prop_source)
l_target = train_test_split(np.arange(n_target), train_size=prop_target)

def test_partial_coot():
    print("discrete partial coot")
    pure_source, pure_target, test_source, test_target = discrete_partial_coot(*data, *l_source, *l_target)
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_partial_jdcoot():
    print("discrete partial jdcoot")
    pure_source, pure_target, test_source, test_target = discrete_partial_jdcoot(*data, *l_source, *l_target)
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_partial_reference():
    print("discrete partial reference")
    pure_source, pure_target, test_source, test_target = discrete_partial_reference(
        *data, *l_source, *l_target
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_semisupervised_coot():
    print("discrete semi-supervised coot")
    pure_source, pure_target, test_source, test_target = discrete_semisupervised_coot(
        *data, *l_source, *l_target
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_semisupervised_jdcoot():
    print("discrete semi-supervised jdcoot")
    pure_source, pure_target, test_source, test_target = discrete_semisupervised_jdcoot(
        *data, *l_source, *l_target
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_semisupervised_reference():
    print("discrete semi-supervised reference")
    pure_source, pure_target, test_source, test_target = (
        discrete_semisupervised_reference(*data, *l_source, *l_target)
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_unsupervised_coot():
    print("discrete unsupervised coot")
    pure_source, pure_target, test_source, test_target = discrete_unsupervised_coot(
        *data, *l_source, *l_target
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_unsupervisel_jdcoot():
    print("discrete unsupervised jdcoot")
    pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(
        *data, *l_source, *l_target
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True
