"""kNN graph backends: RINND (default, NN-descent in Rust) or the bundled HNSW.

Both return int64 indices [n_points, k], nearest first, without the point itself. Cosine
distance is handled by normalizing the rows first, which gives the same neighbours as cosine.

The HNSW defaults are denser than stock NCVis's (M=8, ef_construction=100, query ef=k+1): those
reach only ~0.85 recall@15 on 512-d embeddings, these ~0.99.
"""

import numpy as np

from ncvis_plus import _core

BACKENDS = ("rinnd", "hnsw")
METRICS = ("euclidean", "cosine")
HNSW_DEFAULTS = {"M": 16, "ef_construction": 200, "ef_search": 64}


def prepare(X: np.ndarray, metric: str) -> np.ndarray:
    """Data as contiguous float32, rows normalized for cosine."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    X = np.ascontiguousarray(X, dtype=np.float32)
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(norms > 0, norms, 1)
    return X


def knn_graph(X: np.ndarray, k: int, backend: str, n_threads: int, seed: int,
              params: dict | None = None) -> np.ndarray:
    if backend == "rinnd":
        return _rinnd(X, k, n_threads, seed, params or {})
    if backend == "hnsw":
        return _hnsw(X, k, n_threads, seed, params or {})
    raise ValueError(f"knn must be one of {BACKENDS} or an index array, got {backend!r}")


def _rinnd(X: np.ndarray, k: int, n_threads: int, seed: int, params: dict) -> np.ndarray:
    from rinnd import RINND

    index = RINND(X, n_neighbors=k, n_jobs=n_threads, random_state=seed, graph_only=True, **params)
    indices, _ = index.neighbor_graph
    return np.ascontiguousarray(indices, dtype=np.int64)


def _hnsw(X: np.ndarray, k: int, n_threads: int, seed: int, params: dict) -> np.ndarray:
    unknown = set(params) - set(HNSW_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown HNSW parameters {sorted(unknown)}; known: {sorted(HNSW_DEFAULTS)}")
    p = HNSW_DEFAULTS | params
    return _core.hnsw_knn(X, k, p["M"], p["ef_construction"], p["ef_search"], seed, n_threads)
