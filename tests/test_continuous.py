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
    pure, test =  continuous_partial_coot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_partial_jdcoot():

    print("continuous partial jdcoot")
    data = generate_data(size=500)
    pure, test =  continuous_partial_jdcoot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True


def test_partial_reference():

    print("continuous partial reference")
    data = generate_data(size=500)
    pure, test =  continuous_partial_reference(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_semisupervised_coot():

    data = generate_data(size=500)
    print("continuous semi-supervised coot")
    pure, test =  continuous_semisupervised_coot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_semisupervised_jdcoot():

    print("continuous semisupervised jdcoot")
    data = generate_data(size=500)
    pure, test =  continuous_semisupervised_jdcoot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_semisupervised_reference():

    print("continuous semisupervised reference")
    data = generate_data(size=500)
    pure, test =  continuous_semisupervised_reference(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_unsupervised_coot():

    print("continuous unsupervised coot")
    data = generate_data(size=500)
    pure, test =  continuous_unsupervised_coot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_unsupervised_jdcoot(size=500):

    print("continuous unsupervised jdcoot")
    data = generate_data()
    pure, test =  continuous_unsupervised_jdcoot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

