"""Regression tests for package exports and the numerical transport helpers."""

import numpy as np
import pytest

import jdcoot
from jdcoot.coot import cot_numpy, init_matrix_np, random_gamma_init
from jdcoot.sinkhorn import sinkhorn_scaling


@pytest.fixture(autouse=True)
def deterministic_random_state():
    state = np.random.get_state()
    np.random.seed(2025)
    yield
    np.random.set_state(state)


def test_public_exports():
    namespace = {}
    exec("from jdcoot import *", namespace)  # pylint: disable=exec-used
    for name in jdcoot.__all__:
        assert isinstance(name, str)
        assert namespace[name] is getattr(jdcoot, name)


def test_init_matrix_matches_pairwise_squared_cost():
    source = np.array([[0.0, 2.0], [3.0, 1.0], [4.0, 5.0]])
    target = np.array([[1.0, 3.0], [2.0, 6.0]])
    source_weights = np.array([0.3, 0.7])
    target_weights = np.array([0.6, 0.4])
    feature_coupling = np.outer(source_weights, target_weights)
    constant, source_term, target_term = init_matrix_np(
        source, target, source_weights, target_weights
    )
    expected = np.sum(
        (source[:, None, :, None] - target[None, :, None, :]) ** 2
        * feature_coupling[None, None, :, :],
        axis=(2, 3),
    )
    np.testing.assert_allclose(
        constant - source_term @ feature_coupling @ target_term.T, expected
    )


@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("random_init", [False, True])
@pytest.mark.parametrize("algo,algo2", [("emd", "emd"), ("sinkhorn", "sinkhorn")])
def test_cot_preserves_marginals_and_squared_cost(log, random_init, algo, algo2):
    source = np.array([[0.0, 0.2], [0.3, 0.1], [0.4, 0.5]])
    target = np.array([[0.1, 0.3], [0.2, 0.6]])
    w1 = np.array([0.2, 0.3, 0.5])
    w2 = np.array([0.6, 0.4])
    v1 = np.array([0.3, 0.7])
    v2 = np.array([0.6, 0.4])
    result = cot_numpy(
        source, target, w1=w1, w2=w2, v1=v1, v2=v2, niter=5,
        algo=algo, algo2=algo2, reg=1.0, reg2=1.0,
        log=log, random_init=random_init, verbose=False,
    )
    samples, features, cost = result[:3]
    np.testing.assert_allclose(samples.sum(axis=1), w1, atol=1e-7)
    np.testing.assert_allclose(samples.sum(axis=0), w2, atol=1e-7)
    np.testing.assert_allclose(features.sum(axis=1), v1, atol=1e-7)
    np.testing.assert_allclose(features.sum(axis=0), v2, atol=1e-7)
    expected = np.einsum(
        "ijkl,ij,kl->",
        (source[:, None, :, None] - target[None, :, None, :]) ** 2,
        samples,
        features,
    )
    np.testing.assert_allclose(cost, expected, atol=1e-7)
    if log:
        assert len(result) == 4
        assert result[3]["cost"]
        assert result[3]["cost"][-1] == cost
    else:
        assert len(result) == 3


def test_random_gamma_init_supports_scipy_sparse_matrices():
    source_weights = np.array([0.2, 0.3, 0.5])
    target_weights = np.array([0.6, 0.4])
    coupling = random_gamma_init(
        source_weights, target_weights, stopThr=1e-16, numItermax=10000
    )
    assert coupling.shape == (3, 2)
    assert np.all(coupling >= 0)
    np.testing.assert_allclose(coupling.sum(axis=1), source_weights, atol=1e-7)
    np.testing.assert_allclose(coupling.sum(axis=0), target_weights, atol=1e-7)


@pytest.mark.parametrize("log", [False, True])
def test_sinkhorn_scaling_single_histogram(log):
    source_weights = np.array([0.2, 0.3, 0.5])
    target_weights = np.array([0.6, 0.4])
    kernel = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = sinkhorn_scaling(source_weights, target_weights, kernel, log=log)
    coupling = result[0] if log else result
    np.testing.assert_allclose(coupling.sum(axis=1), source_weights, atol=1e-7)
    np.testing.assert_allclose(coupling.sum(axis=0), target_weights, atol=1e-7)
    if log:
        np.testing.assert_allclose(
            result[1]["u"][:, None] * kernel * result[1]["v"][None, :], coupling
        )
        assert result[1]["err"]


@pytest.mark.parametrize("log", [False, True])
def test_sinkhorn_scaling_multiple_histograms(log):
    source_weights = np.array([0.2, 0.3, 0.5])
    target_weights = np.array([[0.6, 0.1], [0.4, 0.9]])
    kernel = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = sinkhorn_scaling(source_weights, target_weights, kernel, log=log)
    masses = result[0] if log else result
    np.testing.assert_allclose(masses, np.ones(2))
    if log:
        details = result[1]
        for index in range(2):
            coupling = (
                details["u"][:, index, None]
                * kernel
                * details["v"][None, :, index]
            )
            np.testing.assert_allclose(
                coupling.sum(axis=1), source_weights, atol=1e-7
            )
            np.testing.assert_allclose(
                coupling.sum(axis=0), target_weights[:, index], atol=1e-7
            )
