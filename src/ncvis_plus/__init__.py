"""NCVis+: noise-contrastive data visualization with t-SNE-level structure at a fraction of the cost.

A fork of NCVis (Artemenkov & Panov, 2020) with a faster layout kernel, PCA initialization,
RINND kNN graphs and retuned defaults. Usage:

    from ncvis_plus import NCVisPlus
    Y = NCVisPlus(random_state=0).fit_transform(X)
"""

from ncvis_plus.estimator import NCVisPlus

__all__ = ["NCVisPlus"]
__version__ = "0.1.0"
