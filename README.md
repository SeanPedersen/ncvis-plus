# NCVis+

2D projections with t-SNE-level local and global structure, 3–5× faster than t-SNE FFT and
using less than half its memory at 512k points.

NCVis+ is a fork of [NCVis](https://github.com/stat-ml/ncvis) (Artemenkov & Panov, 2020). It keeps NCVis's
noise-contrastive objective and changes four things around it:

- **PCA initialization:** with random or spectral starts NCVis loses global structure as N grows.
- **Retuned defaults:** `min_dist` 0.001, learning rate 2, twice the negative samples.
- **A faster layout kernel:** the random negative rows are prefetched, plus a cheaper kernel and RNG.
- **RINND as the default kNN graph.**

```python
from ncvis_plus import NCVisPlus

Y = NCVisPlus(random_state=0).fit_transform(X)                 # balanced default
Y = NCVisPlus(n_negative=20, random_state=0).fit_transform(X)  # higher local quality, ~1.6x the time
```

## Quality vs t-SNE

The measures:

- **local:** mean trustworthiness at k = 5, 15, 30.
- **global:** Spearman ρ between high-dimensional and 2D distances of 1M random point pairs.

Setup: mean of 2–3 seeds, 36 threads (i9-7980XE). Time and peak RAM are end to end, kNN graph
included. Every method gets a RINND graph: k = 90 for t-SNE, 15 for the others.

**`jinaclip-512-1m`, 512k points** (512-d image-text embeddings)

| method | local | global | time | peak RAM |
|---|---|---|---|---|
| openTSNE FFT | **0.970** | 0.168 | 242 s | 4.2 GB |
| manifolds-rs t-SNE FFT | 0.968 | **0.188** | 73 s | 5.6 GB |
| **NCVis+ `n_negative=20`** | 0.962 | 0.175 | 32 s | 2.1 GB |
| **NCVis+** | 0.955 | 0.174 | 20 s | 2.1 GB |
| NCVis 1.5.13 | 0.928 | 0.067 | 23 s | 1.8 GB |

**`dbpedia-512-100k`** (512-d OpenAI text embeddings)

| method | local | global | time | peak RAM |
|---|---|---|---|---|
| openTSNE FFT | **0.984** | 0.278 | 79 s | 1.2 GB |
| manifolds-rs t-SNE FFT | 0.983 | 0.284 | 21 s | 1.4 GB |
| **NCVis+ `n_negative=20`** | 0.983 | 0.284 | 6.1 s | 0.8 GB |
| **NCVis+** | 0.979 | **0.287** | 3.9 s | 0.8 GB |
| NCVis 1.5.13 | 0.963 | 0.234 | 3.9 s | 0.4 GB |

**`jinaclip-512-1m`, 100k points**

| method | local | global | time | peak RAM |
|---|---|---|---|---|
| openTSNE FFT | **0.958** | 0.202 | 83 s | 1.6 GB |
| manifolds-rs t-SNE FFT | 0.957 | **0.214** | 15 s | 1.6 GB |
| **NCVis+ `n_negative=20`** | 0.952 | 0.176 | 7.1 s | 0.8 GB |
| **NCVis+** | 0.945 | 0.176 | 4.6 s | 0.8 GB |
| NCVis 1.5.13 | 0.922 | 0.109 | 4.4 s | 0.4 GB |

What the tables show:

- **NCVis+ matches t-SNE's global structure on dbpedia.** At 512k it sits between openTSNE and
  manifolds-rs, within t-SNE's seed spread (sd 0.018). Stock NCVis keeps 40% of it at 512k.
- **The default is 0.004–0.014 behind t-SNE locally.** `n_negative=20` halves that gap.
- **The exception is jinaclip at 100k,** where global stays at 0.176 against t-SNE's 0.21.

Caveats:

- **Provenance of the scores:** they come from a separate benchmark
  ([drbench](https://github.com/SeanPedersen/projection-benchmark)), and only two datasets have been tested so far.
- **Stock NCVis was run with `min_dist` 0.1 and on the same RINND graph,** not with its own HNSW search.
- **NCVis+ times were measured at a different time from the t-SNE times,** on the same host.
- **How local is scored:** at 512k, and for NCVis+ at 100k, local is estimated on 10k fixed anchor
  points (ranks exact over all N). On stored t-SNE embeddings at 100k this matches exact scoring
  within 0.0005.

## What changed and why

The tuning was done on 100k subsets and checked at 512k. Measured, per change:

| change | effect at 100k (jinaclip) |
|---|---|
| PCA init, first axis scaled to std 10 | global 0.10 → 0.175, seed spread 0.03 → 0.003 |
| `min_dist` 0.1 → 0.001 | local +0.012, free |
| mean negatives per edge 5 → 10, learning rate 1 → 2 | local +0.012, ~1.5× layout time |
| prefetching the negatives' rows | layout 1.5× faster at 36 threads, identical updates |

Tried and dropped:

- **More neighbours:** k = 90, unweighted, was worse on both axes.
- **Perplexity-weighted edges:** local −0.007 to −0.05.
- **Steeper negative-sample ramps:** within noise of the cubic ramp.
- **Moving only the edge's source point:** faster, but global drops to 0.12.
- **Reordering points for cache locality:** no gain at 100k or 512k.
- **NCVis's spectral init:** no better than random, and it costs time. It is removed.

## Parameters

| parameter | default | |
|---|---|---|
| `n_components` | 2 | |
| `n_neighbors` | 15 | kNN graph size |
| `n_epochs` | 50 | |
| `n_negative` | 10 | mean negative samples per edge (cubic ramp over epochs); the time/quality knob |
| `min_dist`, `spread` | 0.001, 1.0 | kernel shape via (a, b), as in UMAP; or pass `a`, `b` directly |
| `learning_rate` | 2.0 | |
| `init` | `"pca"` | `"pca"`, `"random"` or an array [n, n_components] |
| `init_scale` | 10.0 | std of the first init axis |
| `knn` | `"rinnd"` | `"rinnd"` or `"hnsw"` (bundled, M=16, ef_construction=200, ef=64); `knn_params` passes options |
| `metric` | `"euclidean"` | or `"cosine"` |
| `n_jobs` | -1 | all available cores |
| `random_state` | None | results are reproducible only with `n_jobs=1` (Hogwild SGD) |

`fit_transform(X, knn_indices=...)` takes a precomputed graph [n, k] without self-neighbours.

## Install and develop

```sh
uv sync                 # Python 3.12; builds the C++ core (needs a C++17 compiler with OpenMP)
uv run pytest
uv run ruff check src tests
uv sync --reinstall-package ncvis-plus   # after editing csrc/ or _core.pyx
```

## License

MIT, like NCVis. The bundled hnswlib (Apache-2.0) and pcg-cpp (Apache-2.0 / MIT) keep their own licenses.
