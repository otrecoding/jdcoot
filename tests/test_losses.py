import numpy as np
import pytest
from jdcoot.losses import loss_crossentropy


def test_loss_crossentropy_basic():
    # Test with a simple case
    Y = np.array([[1, 0], [0, 1]])  # One-hot encoded true labels
    F = np.array([[0.8, 0.2], [0.3, 0.7]])  # Predicted probabilities
    expected = np.array([[0.22314355, 1.2039728], [1.60943791, 0.35667494]])
    result = loss_crossentropy(Y, F)
    np.testing.assert_almost_equal(result, expected, decimal=6)


def test_loss_crossentropy_with_zeros():
    # Test with probabilities close to zero
    Y = np.array([[1, 0], [0, 1]])
    F = np.array([[1e-10, 1 - 1e-10], [1 - 1e-10, 1e-10]])
    result = loss_crossentropy(Y, F)
    assert np.all(result >= 0)  # Loss should be non-negative


def test_loss_crossentropy_with_uniform_distribution():
    # Test with uniform probabilities
    Y = np.array([[1, 0], [0, 1]])
    F = np.array([[0.5, 0.5], [0.5, 0.5]])
    result = loss_crossentropy(Y, F)
    expected = np.array([[0.69314718, 0.69314718], [0.69314718, 0.69314718]])
    np.testing.assert_almost_equal(result, expected, decimal=6)

