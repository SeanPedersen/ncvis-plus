// NCVis+ core: kNN search (HNSW), graph symmetrization and the noise-contrastive layout.
//
// The layout is NCVis (Artemenkov & Panov, 2020): Hogwild SGD on a noise-contrastive estimation
// objective with the kernel q(d) = 1 / (1 + a d^(2b)) and a learned normalization constant Q.
// Python builds the kNN graph (RINND by default, or `hnsw_knn` here) and the initial layout;
// this module only symmetrizes the graph and optimizes the layout in place.
#ifndef NCVIS_PLUS_HPP
#define NCVIS_PLUS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ncvis {

// Approximate kNN of every row of X [n_points, dim] (squared L2), written to
// indices_out [n_points, k] without the point itself.
void hnsw_knn(const float *X, size_t n_points, size_t dim, size_t k, size_t M,
              size_t ef_construction, size_t ef_search, size_t seed, int n_threads,
              int64_t *indices_out);

// Undirected neighbour lists as CSR: row i's neighbours are targets[offsets[i] .. offsets[i+1]).
struct Graph {
    std::vector<size_t> offsets;
    std::vector<uint32_t> targets;
};

// Union of the kNN graph [n_points, k] and its transpose, without duplicates or self loops.
Graph symmetrize(const int64_t *indices, size_t n_points, size_t k, int n_threads);

struct LayoutParams {
    size_t dim;                       // embedding dimensions
    float a, b;                       // kernel q(d) = 1 / (1 + a d^(2b))
    float learning_rate;              // decays as learning_rate * (1 - (epoch / n_epochs)^2)
    float learning_rate_Q;            // step of the log normalization constant
    uint64_t seed;
    int n_threads;
    std::vector<uint32_t> noise_plan; // negative samples per edge, one entry per epoch
};

// Optimizes Y [n_points, dim] in place, starting from its current values.
void optimize(const Graph &graph, float *Y, size_t n_points, const LayoutParams &params);

}  // namespace ncvis

#endif  // NCVIS_PLUS_HPP
