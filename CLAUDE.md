# CLAUDE.md

NCVis+: fork of NCVis 1.5.13 (noise-contrastive 2D projection) with PCA init, retuned defaults, a
prefetching layout kernel with a local-noise final phase and RINND kNN graphs. Python 3.12 + uv; C++17 core via Cython.

## Commands
- Build and test on hartmut, not locally: `uv sync`, `uv run pytest` (33 tests), `uv run ruff check src tests`.
- After editing `csrc/` or `_core.pyx`: `uv sync --reinstall-package ncvis-plus`.

## Layout
- `csrc/ncvis.{hpp,cpp}`: `hnsw_knn`, `symmetrize` (kNN -> deduplicated undirected CSR, uint32 targets), `optimize` (Hogwild NCE SGD on Y in place; dispatches `optimize_dim<2|3|0>`, 0 = runtime dim).
- `csrc/local_noise.hpp`: grid over Y and the 3^DIM cell block around a point, for the local phase (last `local_epochs`, 2D/3D only): a share `local_share` of the negatives comes from the block and the mixture density enters the NCE ratio.
- `src/ncvis_plus/_core.pyx`: thin bindings, GIL released. Validation lives in Python.
- `src/ncvis_plus/estimator.py`: `NCVisPlus`; orchestrates kNN -> init -> `_core.embed`.
- `knn.py` (RINND default, HNSW optional; cosine = row-normalized euclidean), `init.py` (PCA fitted on 20k rows, projected without a centred copy of X), `kernel.py` (numpy Levenberg-Marquardt for (a, b), no scipy), `schedule.py` (cubic negative-sample ramp).
- `lib/`: vendored hnswlib and pcg-cpp headers, each with its own license.

## Gotchas
- Compiled with `-ffast-math`: `std::isinf` / `std::isnan` are unreliable. Use finite sentinels.
- The hot loop is latency-bound, not throughput-bound: (1) loading random negatives' rows (cross-core traffic), so negatives are drawn up front per edge and prefetched; (2) the sample-to-sample chain through `powf`/`expf`/divisions, so an edge's samples run in blocks of `SNAPSHOT_BLOCK` (8) sharing one snapshot of yi and Q. Keep both if you touch the loop. Wider SIMD (`-march`), write-intent prefetch and cross-edge lookahead were measured neutral or slower.
- Kernel experiments (variants, harness, hartmut results): `~/projects/playground/experiments/ncvis-kernel/`, results in `~/experiments/ncvis-kernel-20261117/results/` on hartmut.
- Output is deterministic only with `n_jobs=1` (Hogwild).
- Local phase (issue #1): keep it short and last (10 epochs; starting earlier collapses local), keep the block centred on each point (fixed partitions such as k-means or grid cells let clusters grow into each other at borders), keep `local_share` < 1 (it divides by 1 - share; `-ffast-math` makes inf unsafe). Cell size must grow with N (`max(64, N / 2000)`). Exaggeration, a lambda ramp, early local noise and unweighted denser graphs did not help global structure; init scale and kernel tail trade local for global.
- Bit-identity checks against the experiment kernel need the exact same float expression order (`-ffast-math` reassociates).
- Benchmarks (README tables) come from drbench (`~/projects/projection-benchmark`): NCVis+ rows from `ncvis-plus-rinnd-default` / `-n20-rinnd-default` on hartmut, means of 3 seeds (`~/experiments/projection-benchmark-fz8-20261117/`: jinaclip from `results-scaling/`, dbpedia from `results-fz8/`; cached RINND graph, build time charged), t-SNE / NCVis rows from the scaling run. Older experiment scripts live in `~/projects/playground/experiments/ncvis-tsne/`. After a kernel change: push, bump the rev in drbench's `pyproject.toml`, rerun those two variants.
