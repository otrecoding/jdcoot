import numpy as np
from sklearn.model_selection import train_test_split

from jdcoot import continuous_partial_coot
from jdcoot import continuous_partial_jdcoot
from jdcoot import continuous_partial_reference
from jdcoot import continuous_semisupervised_coot
from jdcoot import continuous_semisupervised_jdcoot
from jdcoot import continuous_semisupervised_reference
from jdcoot import continuous_unsupervised_coot
from jdcoot import continuous_unsupervised_jdcoot
from jdcoot import generate_data


def test_partial_coot():
    print("continuous partial coot")
    data = generate_data(size=500)
    source, target = data[0], data[1]
    n_source = len(source.Y)
    n_target = len(target.Y)
    l_source = train_test_split(np.arange(n_source), train_size=prop_source)
    l_target = train_test_split(np.arange(n_target), train_size=prop_target)

    pure_source, pure_target, test_source, test_target = continuous_partial_coot(*data, *l_source, *l_target)
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_partial_jdcoot():
    print("continuous partial jdcoot")
    data = generate_data(size=500)
    source, target = data[0], data[1]
    n_source = len(source.Y)
    n_target = len(target.Y)
    l_source = train_test_split(np.arange(n_source), train_size=prop_source)
    l_target = train_test_split(np.arange(n_target), train_size=prop_target)

    pure_source, pure_target, test_source, test_target = continuous_partial_jdcoot(
        *data, *l_source, *l_target
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_partial_reference():
    print("continuous partial reference")
    data = generate_data(size=500)
    source, target = data[0], data[1]
    n_source = len(source.Y)
    n_target = len(target.Y)
    l_source = train_test_split(np.arange(n_source), train_size=prop_source)
    l_target = train_test_split(np.arange(n_target), train_size=prop_target)
    pure_source, pure_target, test_source, test_target = continuous_partial_reference(
        *data, *l_source, *l_target
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_semisupervised_coot():
    data = generate_data(size=500)
    source, target = data[0], data[1]
    n_source = len(source.Y)
    n_target = len(target.Y)
    l_source = train_test_split(np.arange(n_source), train_size=prop_source)
    l_target = train_test_split(np.arange(n_target), train_size=prop_target)
    print("continuous semi-supervised coot")
    pure_source, pure_target, test_source, test_target = continuous_semisupervised_coot(
        *data, *l_source, *l_target
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_semisupervised_jdcoot():
    print("continuous semisupervised jdcoot")
    data = generate_data(size=500)
    source, target = data[0], data[1]
    n_source = len(source.Y)
    n_target = len(target.Y)
    l_source = train_test_split(np.arange(n_source), train_size=prop_source)
    l_target = train_test_split(np.arange(n_target), train_size=prop_target)
    pure_source, pure_target, test_source, test_target = (
        continuous_semisupervised_jdcoot(*data, *l_source, *l_target)
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_semisupervised_reference():
    print("continuous semisupervised reference")
    data = generate_data(size=500)
    source, target = data[0], data[1]
    n_source = len(source.Y)
    n_target = len(target.Y)
    l_source = train_test_split(np.arange(n_source), train_size=prop_source)
    l_target = train_test_split(np.arange(n_target), train_size=prop_target)
    pure_source, pure_target, test_source, test_target = (
        continuous_semisupervised_reference(*data, *l_source, *l_target)
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_unsupervised_coot():
    print("continuous unsupervised coot")
    data = generate_data(size=500)
    source, target = data[0], data[1]
    n_source = len(source.Y)
    n_target = len(target.Y)
    l_source = train_test_split(np.arange(n_source), train_size=prop_source)
    l_target = train_test_split(np.arange(n_target), train_size=prop_target)
    pure_source, pure_target, test_source, test_target = continuous_unsupervised_coot(
        *data, *l_source, *l_target
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True


def test_unsupervised_jdcoot(size=500):
    print("continuous unsupervised jdcoot")
    data = generate_data()
    source, target = data[0], data[1]
    n_source = len(source.Y)
    n_target = len(target.Y)
    l_source = train_test_split(np.arange(n_source), train_size=prop_source)
    l_target = train_test_split(np.arange(n_target), train_size=prop_target)
    pure_source, pure_target, test_source, test_target = continuous_unsupervised_jdcoot(
        *data, *l_source, *l_target
    )
    print(
        f"pure_source, pure_target, test_source, test_target = {pure_source:7.3f}, {pure_target:7.3f}, {test_source:7.3f}, {test_target:7.3f}"
    )
    assert True
