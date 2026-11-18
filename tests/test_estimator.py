"""End-to-end: NCVisPlus separates clusters and preserves neighbourhoods on synthetic data."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier

from ncvis_plus import NCVisPlus

N_POINTS = 3000
N_FEATURES = 32
N_CLUSTERS = 6
THREADS = 4


@pytest.fixture(scope="module")
def blobs():
    X, labels = make_blobs(N_POINTS, N_FEATURES, centers=N_CLUSTERS, cluster_std=2.0, random_state=0)
    return X.astype(np.float32), labels


def cluster_accuracy(Y, labels):
    half = len(Y) // 2
    model = KNeighborsClassifier(10).fit(Y[:half], labels[:half])
    return model.score(Y[half:], labels[half:])


@pytest.mark.parametrize("knn", ["rinnd", "hnsw"])
def test_embedding_keeps_clusters_and_neighbourhoods(blobs, knn):
    X, labels = blobs
    Y = NCVisPlus(knn=knn, n_jobs=THREADS, random_state=0).fit_transform(X)
    assert Y.shape == (N_POINTS, 2) and Y.dtype == np.float32
    assert np.isfinite(Y).all()
    assert cluster_accuracy(Y, labels) > 0.98
    assert trustworthiness(X, Y, n_neighbors=10) > 0.85


def test_precomputed_knn_skips_search(blobs):
    X, labels = blobs
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    knn = np.argsort(d2, axis=1)[:, :15]
    Y = NCVisPlus(n_jobs=THREADS, random_state=0).fit_transform(X, knn_indices=knn)
    assert cluster_accuracy(Y, labels) > 0.98


def test_single_thread_is_deterministic(blobs):
    X, _ = blobs
    first = NCVisPlus(knn="hnsw", n_jobs=1, random_state=1).fit_transform(X)
    second = NCVisPlus(knn="hnsw", n_jobs=1, random_state=1).fit_transform(X)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("options", [{"init": "random"}, {"metric": "cosine"}, {"a": 1.0, "b": 1.0},
                                     {"n_components": 3}, {"n_components": 4}, {"local_epochs": 50},
                                     {"local_share": 0.0}, {"local_cell_points": 1}])
def test_options_produce_finite_layouts(blobs, options):
    X, _ = blobs
    Y = NCVisPlus(n_jobs=THREADS, random_state=0, **options).fit_transform(X)
    assert Y.shape == (N_POINTS, options.get("n_components", 2))
    assert np.isfinite(Y).all()


@pytest.mark.parametrize("n_components", [2, 3])
def test_local_phase_is_deterministic_and_sharpens_neighbourhoods(blobs, n_components):
    X, _ = blobs
    options ={"knn": "hnsw", "n_components": n_components, "n_jobs": 1, "random_state": 2}
    uniform = NCVisPlus(local_epochs=0, **options).fit_transform(X)
    local = NCVisPlus(**options).fit_transform(X)
    assert np.array_equal(local, NCVisPlus(**options).fit_transform(X))
    assert not np.array_equal(local, uniform)
    assert trustworthiness(X, local, n_neighbors=10) >= trustworthiness(X, uniform, n_neighbors=10)


def test_array_init_is_used(blobs):
    X, _ = blobs
    init = np.random.default_rng(0).standard_normal((N_POINTS, 2)).astype(np.float32)
    model = NCVisPlus(init=init, n_epochs=1, n_negative=1, learning_rate=1e-6, n_jobs=THREADS)
    assert np.allclose(model.fit_transform(X), init, atol=1e-3)


@pytest.mark.parametrize("options", [{"knn": "annoy"}, {"init": "spectral"}, {"a": 1.0},
                                     {"metric": "manhattan"}, {"local_epochs": -1}, {"local_share": 1.0},
                                     {"local_cell_points": 0}])
def test_invalid_options_raise(blobs, options):
    X, _ = blobs
    with pytest.raises(ValueError):
        NCVisPlus(**options).fit_transform(X)


def test_bad_knn_indices_raise(blobs):
    X, _ = blobs
    with pytest.raises(ValueError):
        NCVisPlus().fit_transform(X, knn_indices=np.full((N_POINTS, 5), N_POINTS))
