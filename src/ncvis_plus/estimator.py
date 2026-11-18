"""NCVisPlus: the scikit-learn style estimator. kNN graph -> initial layout -> NCE layout.

Defaults (tuned on 512-d text and image embeddings, 100k-512k points, against t-SNE FFT):
- kNN: RINND with k = 15. More neighbours, unweighted, made the layout worse; perplexity-style
  edge weights on k = 15 did too.
- init: PCA scaled to std 10 on the first axis. Smaller scales lose global structure, larger ones
  (30+) keep the layout too close to PCA and lose local structure.
- min_dist 0.001 (stock NCVis via drbench: 0.1) and learning rate 2: both raise local quality at
  no cost.
- 50 epochs, on average 10 negatives per edge (stock: 5). Negatives are the knob that trades
  time for local quality; `n_negative=20` is the higher-quality setting.
- Local phase (2D/3D): in the last 10 epochs 99% of the negatives come from the grid cells around
  the source point (cells of max(64, N / 2000) points), the mixture density in the NCE ratio.
  Local +0.009-0.017 (above FFT t-SNE at 100k-512k), global unchanged on jinaclip, -0.004 on
  dbpedia, +11-22% layout time. Starting it earlier, or a fixed partition (k-means / grid cells),
  breaks the separation between clusters.
"""

import os

import numpy as np

from ncvis_plus import _core
from ncvis_plus.init import pca_init, random_init
from ncvis_plus.kernel import find_ab
from ncvis_plus.knn import BACKENDS, knn_graph, prepare
from ncvis_plus.schedule import noise_plan

INITS = ("pca", "random")
LEARNING_RATE_Q = 1.0
MIN_POINTS = 3
SEED_BITS = 32
LOCAL_DIMS = (2, 3)
MIN_CELL_POINTS = 64
AUTO_CELLS = 2000  # auto cell size: N / AUTO_CELLS points, at least MIN_CELL_POINTS (best at 100k-512k)


class NCVisPlus:
    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 15,
        n_epochs: int = 50,
        n_negative: float = 10,
        local_epochs: int = 10,
        local_share: float = 0.99,
        local_cell_points: float | None = None,
        min_dist: float = 0.001,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        learning_rate: float = 2.0,
        init: str | np.ndarray = "pca",
        init_scale: float = 10.0,
        knn: str = "rinnd",
        knn_params: dict | None = None,
        metric: str = "euclidean",
        n_jobs: int = -1,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.n_negative = n_negative
        self.local_epochs = local_epochs
        self.local_share = local_share
        self.local_cell_points = local_cell_points
        self.min_dist = min_dist
        self.spread = spread
        self.a = a
        self.b = b
        self.learning_rate = learning_rate
        self.init = init
        self.init_scale = init_scale
        self.knn = knn
        self.knn_params = knn_params
        self.metric = metric
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: np.ndarray, knn_indices: np.ndarray | None = None) -> "NCVisPlus":
        self.embedding_ = self.fit_transform(X, knn_indices)
        return self

    def fit_transform(self, X: np.ndarray, knn_indices: np.ndarray | None = None) -> np.ndarray:
        """Embed X [n_points, n_features]. `knn_indices` [n_points, k] (without the point itself)
        skips the kNN search; X is then only used for the PCA init."""
        X = prepare(X, self.metric)
        self._check(X)
        n_threads = _threads(self.n_jobs)
        seed = _seed(self.random_state)
        k = min(self.n_neighbors, len(X) - 1)
        knn = (_checked_knn(knn_indices, len(X)) if knn_indices is not None
               else knn_graph(X, k, self.knn, n_threads, seed, self.knn_params))
        Y = self._initial_layout(X, seed)
        a, b = (self.a, self.b) if self.a is not None else find_ab(self.spread, self.min_dist)
        _core.embed(knn, Y, a, b, self.learning_rate, LEARNING_RATE_Q,
                    noise_plan(self.n_epochs, self.n_negative), seed, n_threads,
                    *self._local_phase(len(X)))
        self.embedding_ = Y
        return Y

    def _local_phase(self, n_points: int) -> tuple[int, float, float]:
        """(epochs, share, cell points) of the local phase; no epochs outside 2D/3D."""
        epochs = min(self.local_epochs, self.n_epochs) if self.n_components in LOCAL_DIMS else 0
        cell_points = (self.local_cell_points if self.local_cell_points is not None
                       else max(MIN_CELL_POINTS, n_points / AUTO_CELLS))
        return epochs, self.local_share, cell_points

    def _initial_layout(self, X: np.ndarray, seed: int) -> np.ndarray:
        if isinstance(self.init, np.ndarray):
            if self.init.shape != (len(X), self.n_components):
                raise ValueError(f"init must have shape {(len(X), self.n_components)}, got {self.init.shape}")
            return np.array(self.init, dtype=np.float32, order="C")
        if self.init == "pca":
            return pca_init(X, self.n_components, self.init_scale, seed)
        return random_init(len(X), self.n_components, self.init_scale, seed)

    def _check(self, X: np.ndarray) -> None:
        if X.ndim != 2 or len(X) < MIN_POINTS:
            raise ValueError(f"X must be 2-d with at least {MIN_POINTS} rows, got shape {X.shape}")
        if self.n_components < 1 or self.n_neighbors < 1:
            raise ValueError("n_components and n_neighbors must be at least 1")
        if (self.a is None) != (self.b is None):
            raise ValueError("give both a and b, or neither")
        if not isinstance(self.init, np.ndarray) and self.init not in INITS:
            raise ValueError(f"init must be one of {INITS} or an array, got {self.init!r}")
        if self.local_epochs < 0 or not 0 <= self.local_share < 1:
            raise ValueError("need local_epochs >= 0 and 0 <= local_share < 1")
        if self.local_cell_points is not None and self.local_cell_points < 1:
            raise ValueError(f"local_cell_points must be at least 1, got {self.local_cell_points}")
        if self.knn not in BACKENDS:
            raise ValueError(f"knn must be one of {BACKENDS}, got {self.knn!r}")


def _checked_knn(knn_indices: np.ndarray, n_points: int) -> np.ndarray:
    knn = np.ascontiguousarray(knn_indices, dtype=np.int64)
    if knn.ndim != 2 or len(knn) != n_points:
        raise ValueError(f"knn_indices must have shape ({n_points}, k), got {knn.shape}")
    if knn.size and (knn.min() < 0 or knn.max() >= n_points):
        raise ValueError("knn_indices contains indices outside [0, n_points)")
    return knn


def _threads(n_jobs: int) -> int:
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return available if n_jobs < 1 else n_jobs


def _seed(random_state: int | None) -> int:
    if random_state is not None:
        return int(random_state) % (1 << SEED_BITS)
    return int(np.random.SeedSequence().generate_state(1)[0])
