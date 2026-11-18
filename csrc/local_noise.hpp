// Local noise for the final layout epochs: a uniform grid over Y and the cell block around a point.
//
// In the last epochs of the layout most negatives are drawn from the 3^DIM cell block around the
// source point instead of uniformly. Uniform negatives almost never land in a small cluster (a
// 500-point cluster in 512k points gets ~0.1% of the draws), so the arrangement inside clusters
// gets almost no repulsion signal. The mixture noise density enters the NCE ratio, which makes
// the classifier discriminate a point's graph neighbours from points that are merely close in Y.
//
// The grid is rebuilt every local epoch with a random offset. Its spacing is set from the 1%-99%
// box of Y so that a cell holds `cell_points` points on average; points are sorted by cell
// (row-major), so a block is 3^(DIM-1) contiguous ranges of the sorted order. A block is centred
// on its point, which matters: a fixed partition (k-means or grid cells) leaves points at the
// borders without repulsion from the other side and lets neighbouring cells grow into each other.
#ifndef NCVIS_PLUS_LOCAL_NOISE_HPP
#define NCVIS_PLUS_LOCAL_NOISE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <pcg_random.hpp>
#include <vector>

namespace ncvis::local_noise {

constexpr uint32_t MAX_SIDE = 65535;      // cell coordinates are stored as uint16
constexpr float CORE_QUANTILE = 0.01f;    // spacing from the 1%-99% box: robust to outliers
constexpr size_t MAX_CELLS_PER_POINT = 4;
constexpr float SPACING_GROWTH = 1.25f;   // spacing step when the grid would get too many cells

template <size_t DIM>
struct Grid {
    uint32_t side[DIM] = {};
    std::vector<uint16_t> coords;  // [n_points, DIM] cell coordinates of each point
    std::vector<uint32_t> order;   // point ids sorted by cell, row-major (x fastest)
    std::vector<uint32_t> start;   // cell id -> first slot in `order`; one entry per cell + 1

    size_t cell_id(const uint32_t *cell) const {
        size_t id = cell[DIM - 1];
        for (size_t c = DIM - 1; c-- > 0;) id = id * side[c] + cell[c];
        return id;
    }
};

inline float quantile(std::vector<float> &values, float q) {
    const size_t at = (size_t)(q * (values.size() - 1));
    std::nth_element(values.begin(), values.begin() + at, values.end());
    return values[at];
}

// Bins Y [n_points, DIM] into a grid with a random offset and sorts the points by cell.
template <size_t DIM>
void rebuild(Grid<DIM> &grid, const float *Y, size_t n_points, float cell_points, pcg32 &rng) {
    float lo[DIM], hi[DIM], core[DIM];
    std::vector<float> axis(n_points);
    for (size_t c = 0; c < DIM; ++c) {
        lo[c] = hi[c] = Y[c];
        for (size_t i = 0; i < n_points; ++i) {
            axis[i] = Y[i * DIM + c];
            lo[c] = std::min(lo[c], axis[i]);
            hi[c] = std::max(hi[c], axis[i]);
        }
        core[c] = quantile(axis, 1 - CORE_QUANTILE) - quantile(axis, CORE_QUANTILE);
    }
    float core_volume = core[0];
    for (size_t c = 1; c < DIM; ++c) core_volume *= core[c];
    const float cell_volume = std::max(core_volume, 1e-12f) * cell_points / n_points;
    float spacing = DIM == 2 ? std::sqrt(cell_volume) : std::pow(cell_volume, 1.f / DIM);
    float shift[DIM];
    for (size_t c = 0; c < DIM; ++c) shift[c] = spacing * std::ldexp((float)rng(), -32);

    size_t n_cells;
    for (;;) {
        n_cells = 1;
        bool fits = true;
        for (size_t c = 0; c < DIM; ++c) {
            const size_t side = (size_t)((hi[c] - lo[c] + shift[c]) / spacing) + 1;
            fits = fits && side <= MAX_SIDE;
            grid.side[c] = (uint32_t)std::min<size_t>(side, MAX_SIDE);
            n_cells *= side;
        }
        if (fits && n_cells <= MAX_CELLS_PER_POINT * n_points) break;
        spacing *= SPACING_GROWTH;
    }

    const float inv_spacing = 1 / spacing;
    grid.coords.resize(n_points * DIM);
    grid.order.resize(n_points);
    grid.start.assign(n_cells + 1, 0);
    std::vector<uint32_t> cell_of(n_points);
    for (size_t i = 0; i < n_points; ++i) {
        uint32_t cell[DIM];
        for (size_t c = 0; c < DIM; ++c) {
            cell[c] = std::min((uint32_t)((Y[i * DIM + c] - (lo[c] - shift[c])) * inv_spacing), grid.side[c] - 1);
            grid.coords[i * DIM + c] = (uint16_t)cell[c];
        }
        cell_of[i] = (uint32_t)grid.cell_id(cell);
        ++grid.start[cell_of[i] + 1];
    }
    for (size_t cell = 0; cell < n_cells; ++cell) grid.start[cell + 1] += grid.start[cell];
    std::vector<uint32_t> fill(grid.start.begin(), grid.start.end() - 1);
    for (size_t i = 0; i < n_points; ++i) grid.order[fill[cell_of[i]]++] = (uint32_t)i;
}

// The 3^DIM cell block around a point: up to 3^(DIM-1) contiguous ranges of the sorted order.
template <size_t DIM>
struct Block {
    static constexpr size_t MAX_RANGES = DIM == 2 ? 3 : 9;
    uint32_t lo[DIM] = {}, hi[DIM] = {};
    uint32_t begin[MAX_RANGES] = {}, size[MAX_RANGES] = {};
    uint32_t total = 0;

    Block() = default;
    Block(const Grid<DIM> &grid, const uint16_t *center) {
        for (size_t c = 0; c < DIM; ++c) {
            lo[c] = center[c] ? center[c] - 1u : 0u;
            hi[c] = std::min(center[c] + 1u, grid.side[c] - 1);
        }
        // Rows run along x; walk the other axes (y fastest), one range per row.
        uint32_t cell[DIM];
        size_t n_ranges = 0;
        for (size_t c = 1; c < DIM; ++c) cell[c] = lo[c];
        for (;;) {
            cell[0] = lo[0];
            begin[n_ranges] = grid.start[grid.cell_id(cell)];
            cell[0] = hi[0];
            size[n_ranges] = grid.start[grid.cell_id(cell) + 1] - begin[n_ranges];
            total += size[n_ranges++];
            size_t c = 1;
            while (c < DIM && cell[c] == hi[c]) cell[c] = lo[c], ++c;
            if (c == DIM) break;
            ++cell[c];
        }
    }

    bool contains(const uint16_t *coords) const {
        for (size_t c = 0; c < DIM; ++c) {
            if (coords[c] < lo[c] || coords[c] > hi[c]) return false;
        }
        return true;
    }

    uint32_t slot(uint32_t u) const {  // u in [0, total)
        for (size_t r = 0;; ++r) {
            if (u < size[r]) return begin[r] + u;
            u -= size[r];
        }
    }
};

}  // namespace ncvis::local_noise

#endif  // NCVIS_PLUS_LOCAL_NOISE_HPP
