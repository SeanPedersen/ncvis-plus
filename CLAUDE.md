# CLAUDE.md

NCVis+: fork of NCVis 1.5.13 (noise-contrastive 2D projection) with PCA init, retuned defaults, a
prefetching layout kernel and RINND kNN graphs. Python 3.12 + uv; C++17 core via Cython.

## Commands
- Build and test on hartmut, not locally: `uv sync`, `uv run pytest` (24 tests), `uv run ruff check src tests`.
- After editing `csrc/` or `_core.pyx`: `uv sync --reinstall-package ncvis-plus`.

## Layout
- `csrc/ncvis.{hpp,cpp}`: `hnsw_knn`, `symmetrize` (kNN -> deduplicated undirected CSR, uint32 targets), `optimize` (Hogwild NCE SGD on Y in place).
- `src/ncvis_plus/_core.pyx`: thin bindings, GIL released. Validation lives in Python.
- `src/ncvis_plus/estimator.py`: `NCVisPlus`; orchestrates kNN -> init -> `_core.embed`.
- `knn.py` (RINND default, HNSW optional; cosine = row-normalized euclidean), `init.py` (PCA fitted on 20k rows, projected without a centred copy of X), `kernel.py` (numpy Levenberg-Marquardt for (a, b), no scipy), `schedule.py` (cubic negative-sample ramp).
- `lib/`: vendored hnswlib and pcg-cpp headers, each with its own license.

## Gotchas
- Compiled with `-ffast-math`: `std::isinf` / `std::isnan` are unreliable. Use finite sentinels.
- The hot loop is bound by the latency of loading random negatives' rows (cross-core traffic), not by arithmetic. Negatives are drawn up front per edge and prefetched; keep that if you touch the loop.
- Output is deterministic only with `n_jobs=1` (Hogwild).
- Benchmarks (README tables) come from drbench (`~/projects/projection-benchmark`), with the RINND graph built inside `fit_transform`; the experiment scripts live in `~/projects/playground/experiments/ncvis-tsne/`.
