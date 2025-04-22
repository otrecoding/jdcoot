from jdcoot import discrete_partial_coot
from jdcoot import discrete_partial_jdcoot
from jdcoot import discrete_partial_reference
from jdcoot import discrete_semisupervised_coot
from jdcoot import discrete_semisupervised_jdcoot
from jdcoot import discrete_semisupervised_reference
from jdcoot import discrete_unsupervised_coot
from jdcoot import discrete_unsupervised_jdcoot
from jdcoot import generate_data

def test_partial_coot():

    print("discrete partial coot")
    data = generate_data(size=500)
    pure, test =  discrete_partial_coot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_partial_jdcoot():

    print("discrete partial jdcoot")
    data = generate_data(size=500)
    pure, test =  discrete_partial_jdcoot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_partial_reference():

    print("discrete partial reference")
    data = generate_data(size=500)
    pure, test =  discrete_partial_reference(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_semisupervised_coot():

    print("discrete semi-supervised coot")
    data = generate_data(size=500)
    pure, test =  discrete_semisupervised_coot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_semisupervised_jdcoot():

    print("discrete semi-supervised jdcoot")
    data = generate_data(size=500)
    pure, test =  discrete_semisupervised_jdcoot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_semisupervised_reference():

    print("discrete semi-supervised reference")
    data = generate_data(size=500)
    pure, test =  discrete_semisupervised_reference(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_unsupervised_coot():

    print("discrete unsupervised coot")
    data = generate_data(size=500)
    pure, test =  discrete_unsupervised_coot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True

def test_unsupervisel_jdcoot():

    print("discrete unsupervised jdcoot")
    data = generate_data(size=500)
    pure, test =  discrete_unsupervised_jdcoot(*data)
    print(f"pure, test = {pure:7.3f}, {test:7.3f}")
    assert True
