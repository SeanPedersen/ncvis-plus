# NCVis+

2D projections with local structure above t-SNE FFT and t-SNE-level global structure on most
data, 4–6× faster than t-SNE FFT and using a third of its memory at 512k points.

NCVis+ is a fork of [NCVis](https://github.com/stat-ml/ncvis) (Artemenkov & Panov, 2020). It keeps NCVis's
noise-contrastive objective and changes five things around it:

- **PCA initialization:** with random or spectral starts NCVis loses global structure as N grows.
- **Retuned defaults:** `min_dist` 0.001, learning rate 2, twice the negative samples.
- **A faster layout kernel:** the random negative rows are prefetched, an edge's samples are applied
  in blocks of 8 instead of one by one, plus a cheaper kernel and RNG.
- **A local-noise final phase:** in the last 10 epochs most negatives come from the points around
  each point instead of from the whole data set.
- **RINND as the default kNN graph.**

```python
from ncvis_plus import NCVisPlus

Y = NCVisPlus(random_state=0).fit_transform(X)                 # balanced default
Y = NCVisPlus(n_negative=20, random_state=0).fit_transform(X)  # slightly higher local, ~1.5x the time
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
| **NCVis+ `n_negative=20`** | **0.973** | 0.173 | 27.4 s | 1.8 GB |
| **NCVis+** | 0.971 | 0.173 | 17.5 s | 1.8 GB |
| openTSNE FFT | 0.970 | 0.168 | 242 s | 4.2 GB |
| manifolds-rs t-SNE FFT | 0.968 | **0.188** | 73 s | 5.6 GB |
| NCVis 1.5.13 | 0.928 | 0.067 | 23 s | 1.8 GB |

**`dbpedia-512-100k`** (512-d OpenAI text embeddings)

| method | local | global | time | peak RAM |
|---|---|---|---|---|
| **NCVis+ `n_negative=20`** | **0.988** | **0.284** | 5.3 s | 0.7 GB |
| **NCVis+** | 0.987 | **0.284** | 3.6 s | 0.7 GB |
| openTSNE FFT | 0.984 | 0.278 | 79 s | 1.2 GB |
| manifolds-rs t-SNE FFT | 0.983 | **0.284** | 21 s | 1.4 GB |
| NCVis 1.5.13 | 0.963 | 0.234 | 3.9 s | 0.4 GB |

**`jinaclip-512-1m`, 100k points**

| method | local | global | time | peak RAM |
|---|---|---|---|---|
| **NCVis+ `n_negative=20`** | **0.962** | 0.178 | 6.1 s | 0.7 GB |
| **NCVis+** | 0.961 | 0.178 | 4.0 s | 0.7 GB |
| openTSNE FFT | 0.958 | 0.202 | 83 s | 1.6 GB |
| manifolds-rs t-SNE FFT | 0.957 | **0.214** | 15 s | 1.6 GB |
| NCVis 1.5.13 | 0.922 | 0.109 | 4.4 s | 0.4 GB |

What the tables show:

- **Locally the default is above both t-SNE implementations on all three data sets:** by
  0.0025–0.003 at 100k and by 0.0007 at 512k (0.9712 ± 0.0002 vs openTSNE's 0.9705 ± 0.0002).
  `n_negative=20` adds 0.001–0.002 for about 1.5× the time.
- **NCVis+ matches t-SNE's global structure on dbpedia.** At 512k it sits between openTSNE and
  manifolds-rs, within t-SNE's seed spread (sd 0.018). Stock NCVis keeps 40% of it at 512k.
- **At 512k the default is faster than stock NCVis** (17.5 s vs 23 s), with all of the gains above.
- **The exception is jinaclip at 100k,** where global stays at 0.178 against t-SNE's 0.20–0.21
  ([#2](https://github.com/SeanPedersen/ncvis-plus/issues/2)).

Caveats:

- **Provenance of the scores:** they come from a separate benchmark
  ([drbench](https://github.com/SeanPedersen/projection-benchmark)), and only two datasets have been tested so far.
- **Stock NCVis was run with `min_dist` 0.1 and on the same RINND graph,** not with its own HNSW search.
- **NCVis+ times were measured at a different time from the t-SNE times,** on the same host.
- **NCVis+ rows:** drbench `ncvis-plus-rinnd-default` and `-n20-rinnd-default` at ncvis-plus
  2f77016, mean of 3 seeds, on the cached RINND graph (its build time is included in the time
  column). jinaclip rows are the drbench scaling rerun. End-to-end times at 512k vary ~±10%
  between runs; the layout kernel alone varies <3%.
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
| samples of an edge in blocks of 8 sharing one snapshot of the point and of Q | layout 1.4× faster; local −0.001 at equal settings |
| local-noise final phase (below) | local 0.944 → 0.961 (512k: 0.955 → 0.971), global unchanged, layout +11–16% |

The blocks remove a serial chain: with per-sample updates each sample waits for the previous
one's `powf`, `expf` and divisions. Layout time and quality at 36 threads, 3 seeds (jinaclip):

| kernel, mean negatives | 100k time | 100k local / global | 512k time | 512k local / global |
|---|---|---|---|---|
| per-sample, 10 | 2.91 s | 0.9450 / 0.1740 | 14.07 s | 0.9560 / 0.1745 |
| blocks of 8, 10 | 2.13 s | 0.9443 / 0.1747 | 10.33 s | 0.9548 / 0.1742 |
| blocks of 8, 13 | 2.72 s | 0.9472 / 0.1753 | 13.14 s | 0.9576 / 0.1745 |
| blocks of 8, 14 | 2.96 s | 0.9482 / 0.1754 | 14.05 s | 0.9586 / 0.1748 |

At the old kernel's time (`n_negative` 13–14) every score is equal or better. Peak RAM is
unchanged (192 MB for the layout at 512k).

**The local-noise final phase.** Uniform negatives almost never land in a small cluster. A
500-point cluster in 512k points gets about 0.1% of the draws, so the arrangement inside clusters
gets almost no repulsion. In the last 10 epochs, 99% of the negatives are drawn instead from the
3×3 grid cells around the source point (cells of max(64, N / 2000) points, rebuilt every epoch
with a random offset). The mixture's density enters the NCE ratio, so the objective learns to
tell a point's graph neighbours from points that are only close in 2D. Measured with the layout
harness (layout time, 36 threads, 3 seeds):

| data | local | recall@15 | global | layout time |
|---|---|---|---|---|
| jinaclip 100k | 0.9445 → 0.9609 | 0.141 → 0.190 | 0.1746 → 0.1754 | 2.18 → 2.42 s |
| dbpedia 100k | 0.9781 → 0.9870 | 0.170 → 0.232 | 0.2869 → 0.2831 | 1.72 → 2.12 s |
| jinaclip 512k | 0.9550 → 0.9715 | 0.105 → 0.139 | 0.1743 → 0.1749 | 10.18 → 11.85 s |

Details matter (experiment log: [#1](https://github.com/SeanPedersen/ncvis-plus/issues/1)):

- **The neighbourhood must be centred on each point.** Fixed partitions, such as k-means on the 2D
  layout or plain grid cells, leave points at the borders without repulsion from the other side.
  Neighbouring cells then grow into each other, and local drops to 0.90–0.93.
- **The phase must be short and come last.** Starting it 15–50 epochs before the end costs local
  quality, because clusters still need uniform repulsion to separate.
- **The objective change is what helps, not less sampling noise.** Drawing the same local negatives
  but reweighting them to keep the uniform objective gains only +0.002.
- **The cell size has to grow with N:** 64-point cells are best at 100k but too small at 512k.

Tried and dropped:

- **More neighbours:** k = 90, unweighted, was worse on both axes, also before the local phase only.
- **Perplexity-weighted edges:** local −0.007 to −0.05.
- **Steeper negative-sample ramps:** within noise of the cubic ramp.
- **Moving only the edge's source point:** faster, but global drops to 0.12.
- **Reordering points for cache locality:** no gain at 100k or 512k.
- **NCVis's spectral init:** no better than random, and it costs time. It is removed.
- **For global structure (#2):** early exaggeration (global −0.014 to −0.046), a λ ramp, and local
  noise early in the run did not help. Larger init scales and lighter kernel tails trade local for
  global.
- **Other kernel tweaks:** write-intent prefetch, drawing negatives one or more edges ahead, and
  `-march=native` (AVX-512) were neutral or slower. Blocks of 16 or whole edges are at most 3%
  faster than 8 and lose more local quality.

## Parameters

| parameter | default | |
|---|---|---|
| `n_components` | 2 | |
| `n_neighbors` | 15 | kNN graph size |
| `n_epochs` | 50 | |
| `n_negative` | 10 | mean negative samples per edge (cubic ramp over epochs); the time/quality knob. 7 is faster than the pre-local-phase default and still at t-SNE's local level |
| `local_epochs` | 10 | final epochs with local noise (2D and 3D only); 0 turns the phase off |
| `local_share` | 0.99 | share of a local epoch's negatives drawn from the grid cells around each point (< 1) |
| `local_cell_points` | None | mean points per grid cell; None = max(64, N / 2000) |
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
uv sync                 # Python 3.11+; builds the C++ core (needs a C++17 compiler with OpenMP)
uv run pytest
uv run ruff check src tests
uv sync --reinstall-package ncvis-plus   # after editing csrc/ or _core.pyx
```

## License

MIT, like NCVis. The bundled hnswlib (Apache-2.0) and pcg-cpp (Apache-2.0 / MIT) keep their own licenses.
