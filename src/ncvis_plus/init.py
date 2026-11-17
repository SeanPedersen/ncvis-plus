"""Initial layouts: PCA (default) or Gaussian noise, scaled so the first axis has std `scale`.

PCA init is what gives NCVis+ its global structure: from random or NCVis's spectral init the
layout keeps about half of t-SNE's random-pair Spearman correlation, from PCA it matches t-SNE at
512k points. The PCA is fitted on a row sample and all rows are projected without a centred copy
of X, so it adds little time and memory.
"""

import numpy as np

PCA_FIT_ROWS = 20_000


def pca_init(X: np.ndarray, n_components: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(X), min(PCA_FIT_ROWS, len(X)), replace=False)
    sample = X[np.sort(rows)].astype(np.float64)
    mean = sample.mean(axis=0)
    _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
    components = vt[:n_components].T.astype(np.float32)
    Y = X @ components - (mean.astype(np.float32) @ components)
    if Y.shape[1] < n_components:  # fewer input features than embedding dimensions
        Y = np.hstack([Y, rng.standard_normal((len(X), n_components - Y.shape[1]), dtype=np.float32)])
    return _scaled(Y, scale)


def random_init(n_points: int, n_components: int, scale: float, seed: int) -> np.ndarray:
    Y = np.random.default_rng(seed).standard_normal((n_points, n_components), dtype=np.float32)
    return _scaled(Y, scale)


def _scaled(Y: np.ndarray, scale: float) -> np.ndarray:
    first_std = float(Y[:, 0].std())
    factor = scale / first_std if first_std > 0 else scale
    return np.ascontiguousarray(Y * factor, dtype=np.float32)
