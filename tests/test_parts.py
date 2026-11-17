"""Kernel fit, negative-sample schedule, initial layouts and the HNSW backend."""

import numpy as np
import pytest
from scipy.optimize import curve_fit

from ncvis_plus import _core
from ncvis_plus.init import pca_init, random_init
from ncvis_plus.kernel import find_ab
from ncvis_plus.knn import prepare
from ncvis_plus.schedule import noise_plan


@pytest.mark.parametrize("min_dist", [0.001, 0.1, 0.5])
def test_find_ab_matches_scipy_curve_fit(min_dist):
    d = np.linspace(0, 3, 300)
    target = np.where(d < min_dist, 1.0, np.exp(-(d - min_dist)))
    (a_ref, b_ref), _ = curve_fit(lambda x, a, b: 1 / (1 + a * x ** (2 * b)), d, target)
    a, b = find_ab(1.0, min_dist)
    assert a == pytest.approx(a_ref, rel=1e-3)
    assert b == pytest.approx(b_ref, rel=1e-3)


def test_find_ab_rejects_bad_shape():
    with pytest.raises(ValueError):
        find_ab(spread=0.0)


def test_noise_plan_ramps_up_with_requested_mean():
    plan = noise_plan(50, 10)
    assert plan.dtype == np.uint32 and len(plan) == 50
    assert plan.min() >= 1
    assert np.all(np.diff(plan.astype(int)) >= 0)
    assert plan.mean() == pytest.approx(10, rel=0.05)


def test_pca_init_follows_leading_component():
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(16).astype(np.float32)
    X = np.outer(rng.standard_normal(5000), direction).astype(np.float32)
    X += 0.01 * rng.standard_normal(X.shape).astype(np.float32)
    Y = pca_init(X, 2, scale=10.0, seed=0)
    assert Y.shape == (5000, 2) and Y.dtype == np.float32
    assert Y[:, 0].std() == pytest.approx(10.0, rel=1e-4)
    assert abs(np.corrcoef(Y[:, 0], X @ direction)[0, 1]) > 0.999


def test_pca_init_pads_when_features_are_fewer_than_components():
    X = np.random.default_rng(0).standard_normal((100, 1)).astype(np.float32)
    assert pca_init(X, 2, scale=1.0, seed=0).shape == (100, 2)


def test_random_init_is_seeded():
    assert np.array_equal(random_init(10, 2, 1.0, seed=3), random_init(10, 2, 1.0, seed=3))


def test_cosine_prepare_normalizes_rows():
    X = prepare(np.array([[3.0, 4.0], [0.0, 2.0]]), "cosine")
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


def test_hnsw_knn_has_high_recall_and_no_self():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 16)).astype(np.float32)
    k = 10
    found = _core.hnsw_knn(X, k, 16, 200, 64, 0, 4)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    exact = np.argsort(d2, axis=1)[:, :k]
    recall = np.mean([len(set(f) & set(e)) / k for f, e in zip(found, exact)])
    assert recall > 0.95
    assert not np.any(found == np.arange(len(X))[:, None])
