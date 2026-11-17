// NCVis+ core. See ncvis.hpp for the interface.
//
// Layout changes over stock NCVis 1.5.13 (same objective and update rule):
// - one powf per sample instead of two, none for the Cauchy kernel (a = b = 1);
// - pcg32 with a multiply-shift range instead of pcg64 with std::uniform_int_distribution;
// - a CSR graph with 32-bit targets instead of a vector of size_t pairs;
// - the negatives of an edge are drawn up front and their rows prefetched. With every thread
//   writing to random rows, the load of a negative's position otherwise stalls the loop on
//   cross-core cache traffic (half of the loop's time in a profile at 100k x 36 threads).
#include "ncvis.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <hnswlib/hnswlib.h>
#include <pcg_random.hpp>

namespace ncvis {

void hnsw_knn(const float *X, size_t n_points, size_t dim, size_t k, size_t M,
              size_t ef_construction, size_t ef_search, size_t seed, int n_threads,
              int64_t *indices_out) {
    if (k == 0 || k >= n_points) {
        throw std::invalid_argument("hnsw_knn: need 0 < k < n_points");
    }
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, n_points, M, ef_construction, seed);
    index.addPoint(X, 0);  // the first point goes in alone: hnswlib needs an entry point
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 256)
    for (long long i = 1; i < (long long)n_points; ++i) {
        index.addPoint(X + i * dim, (size_t)i);
    }
    index.setEf(std::max(ef_search, k + 1));
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 256)
    for (long long i = 0; i < (long long)n_points; ++i) {
        auto result = index.searchKnn(X + i * dim, k + 1);  // max-heap: farthest on top
        std::vector<int64_t> found;
        found.reserve(k + 1);
        while (!result.empty()) {
            found.push_back((int64_t)result.top().second);
            result.pop();
        }
        std::reverse(found.begin(), found.end());  // nearest first
        auto self = std::find(found.begin(), found.end(), (int64_t)i);
        if (self != found.end()) {
            found.erase(self);
        }
        found.resize(k, found.empty() ? 0 : found.back());
        std::copy(found.begin(), found.end(), indices_out + i * k);
    }
}

Graph symmetrize(const int64_t *indices, size_t n_points, size_t k, int n_threads) {
    std::vector<size_t> degree(n_points, 0);
    for (size_t i = 0; i < n_points; ++i) {
        for (size_t j = 0; j < k; ++j) {
            const int64_t other = indices[i * k + j];
            if (other < 0 || (size_t)other >= n_points) {
                throw std::invalid_argument("symmetrize: neighbour index out of range");
            }
            ++degree[i];
            ++degree[other];
        }
    }
    Graph graph;
    graph.offsets.assign(n_points + 1, 0);
    for (size_t i = 0; i < n_points; ++i) {
        graph.offsets[i + 1] = graph.offsets[i] + degree[i];
    }
    graph.targets.resize(graph.offsets[n_points]);
    std::vector<size_t> fill(graph.offsets.begin(), graph.offsets.end() - 1);
    for (size_t i = 0; i < n_points; ++i) {
        for (size_t j = 0; j < k; ++j) {
            const size_t other = (size_t)indices[i * k + j];
            graph.targets[fill[i]++] = (uint32_t)other;
            graph.targets[fill[other]++] = (uint32_t)i;
        }
    }
    // Sort each row, drop duplicates and self loops, then compact.
    std::vector<size_t> kept(n_points);
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1024)
    for (long long i = 0; i < (long long)n_points; ++i) {
        uint32_t *row_begin = graph.targets.data() + graph.offsets[i];
        uint32_t *row_end = graph.targets.data() + graph.offsets[i + 1];
        std::sort(row_begin, row_end);
        row_end = std::unique(row_begin, row_end);
        row_end = std::remove(row_begin, row_end, (uint32_t)i);
        kept[i] = (size_t)(row_end - row_begin);
    }
    size_t write = 0;
    for (size_t i = 0; i < n_points; ++i) {
        const size_t read = graph.offsets[i];
        std::copy(graph.targets.begin() + read, graph.targets.begin() + read + kept[i],
                  graph.targets.begin() + write);
        graph.offsets[i] = write;
        write += kept[i];
    }
    graph.offsets[n_points] = write;
    graph.targets.resize(write);
    graph.targets.shrink_to_fit();
    return graph;
}

void optimize(const Graph &graph, float *Y, size_t n_points, const LayoutParams &params) {
    const size_t dim = params.dim;
    const float a = params.a, b = params.b;
    const bool cauchy = (a == 1.f && b == 1.f);
    const float min_d2 = 1e-12f;
    const float max_shift = 4.f;
    const size_t prefetch_ahead = 8;
    const size_t n_edges = graph.targets.size();
    const int n_epochs = (int)params.noise_plan.size();
    if (n_points < 2 || n_edges == 0 || n_epochs == 0) {
        return;
    }
    const uint32_t max_noise = *std::max_element(params.noise_plan.begin(), params.noise_plan.end());
    const uint32_t *targets = graph.targets.data();
    const size_t *offsets = graph.offsets.data();
    float Q = 0.f;
    float Q_sum = 0.f;

#pragma omp parallel num_threads(params.n_threads)
    {
        const int n_threads = omp_get_num_threads();
        pcg32 rng(params.seed, (uint64_t)omp_get_thread_num());
        const uint64_t n_points64 = n_points;
        std::vector<uint32_t> others(max_noise + 1);

        for (int epoch = 0; epoch < n_epochs; ++epoch) {
            const float progress = (float)epoch / n_epochs;
            const float step = params.learning_rate * (1 - progress * progress);
            const uint32_t n_noise = params.noise_plan[epoch];
            const float inv_noise = n_noise ? 1.f / n_noise : 0.f;
            float Q_local = Q;
#pragma omp for nowait schedule(static)
            for (long long i = 0; i < (long long)n_points; ++i) {
                float *yi = Y + (size_t)i * dim;
                for (size_t edge = offsets[i]; edge < offsets[i + 1]; ++edge) {
                    if (edge + prefetch_ahead < n_edges) {
                        __builtin_prefetch(Y + (size_t)targets[edge + prefetch_ahead] * dim);
                    }
                    others[0] = targets[edge];
                    for (uint32_t s = 1; s <= n_noise; ++s) {
                        uint32_t other;
                        do {
                            other = (uint32_t)(((uint64_t)rng() * n_points64) >> 32);
                        } while (other == (uint32_t)i);
                        others[s] = other;
                        __builtin_prefetch(Y + (size_t)other * dim);
                    }
                    for (uint32_t s = 0; s <= n_noise; ++s) {
                        float *yo = Y + (size_t)others[s] * dim;
                        float d2 = 0;
                        for (size_t c = 0; c < dim; ++c) {
                            const float diff = yo[c] - yi[c];
                            d2 += diff * diff;
                        }
                        float q, kernel_grad;  // kernel_grad = 2ab q d2^(b-1)
                        if (cauchy) {
                            q = 1 / (1 + d2);
                            kernel_grad = 2 * q;
                        } else {
                            const float d2_safe = d2 > min_d2 ? d2 : min_d2;
                            const float d2_pow_b = powf(d2_safe, b);
                            q = 1 / (1 + a * d2_pow_b);
                            kernel_grad = 2 * a * b * q * d2_pow_b / d2_safe;
                        }
                        float w = 1.f;  // no negatives this epoch: plain attraction
                        if (n_noise != 0) {
                            const float ratio = q * inv_noise * expf(-Q_local);
                            w = (s == 0) ? 1 / (1 + ratio) : -1 / (1 + 1 / ratio);
                            Q_local -= w * params.learning_rate_Q;
                            w *= kernel_grad;
                        }
                        for (size_t c = 0; c < dim; ++c) {
                            float shift = (yo[c] - yi[c]) * w * step;
                            shift = shift > max_shift ? max_shift : (shift < -max_shift ? -max_shift : shift);
                            yi[c] += shift;
                            yo[c] -= shift;
                        }
                    }
                }
            }
#pragma omp atomic
            Q_sum += Q_local;
#pragma omp barrier
#pragma omp single
            {
                Q = Q_sum / n_threads;
                Q_sum = 0.f;
            }
        }
    }
}

}  // namespace ncvis
