# cython: language_level=3, boundscheck=False, wraparound=False
"""Thin Cython bindings to the C++ core (csrc/ncvis.hpp). Validation lives in the Python API."""

import numpy as np

cimport numpy as cnp
from libc.stdint cimport int64_t, uint32_t, uint64_t
from libcpp.vector cimport vector

cnp.import_array()


cdef extern from "ncvis.hpp" namespace "ncvis":
    cdef cppclass Graph:
        vector[size_t] offsets
        vector[uint32_t] targets

    cdef cppclass LayoutParams:
        size_t dim
        float a, b, learning_rate, learning_rate_Q
        uint64_t seed
        int n_threads
        vector[uint32_t] noise_plan

    void c_hnsw_knn "ncvis::hnsw_knn"(const float *X, size_t n_points, size_t dim, size_t k,
                                      size_t M, size_t ef_construction, size_t ef_search,
                                      size_t seed, int n_threads, int64_t *indices_out) except + nogil
    Graph c_symmetrize "ncvis::symmetrize"(const int64_t *indices, size_t n_points, size_t k,
                                           int n_threads) except + nogil
    void c_optimize "ncvis::optimize"(const Graph &graph, float *Y, size_t n_points,
                                      const LayoutParams &params) except + nogil


def hnsw_knn(const float[:, ::1] X, size_t k, size_t M, size_t ef_construction, size_t ef_search,
             size_t seed, int n_threads):
    """kNN indices [n_points, k] of every row of X, nearest first, without the point itself."""
    indices = np.empty((X.shape[0], k), dtype=np.int64)
    cdef int64_t[:, ::1] out = indices
    with nogil:
        c_hnsw_knn(&X[0, 0], X.shape[0], X.shape[1], k, M, ef_construction, ef_search, seed,
                   n_threads, &out[0, 0])
    return indices


def embed(const int64_t[:, ::1] knn, float[:, ::1] Y, float a, float b, float learning_rate,
          float learning_rate_Q, const uint32_t[::1] noise_plan, uint64_t seed, int n_threads):
    """Symmetrize the kNN graph and optimize Y [n_points, dim] in place."""
    cdef LayoutParams params
    params.dim = Y.shape[1]
    params.a = a
    params.b = b
    params.learning_rate = learning_rate
    params.learning_rate_Q = learning_rate_Q
    params.seed = seed
    params.n_threads = n_threads
    params.noise_plan.assign(&noise_plan[0], &noise_plan[0] + noise_plan.shape[0])
    cdef Graph graph
    with nogil:
        graph = c_symmetrize(&knn[0, 0], knn.shape[0], knn.shape[1], n_threads)
        c_optimize(graph, &Y[0, 0], Y.shape[0], params)
