"""Negative samples per edge for each epoch.

NCVis ramps the count up with the cube of training progress: about one negative per edge in the
first half, most of them in the last epochs. The ramp is rescaled so its mean is `n_negative`,
and every epoch keeps at least one. Steeper ramps were no better at the same cost.
"""

import numpy as np

RAMP_POWER = 3
MIN_NEGATIVES = 1


def noise_plan(n_epochs: int, n_negative: float) -> np.ndarray:
    if n_epochs < 1 or n_negative < MIN_NEGATIVES:
        raise ValueError(f"need n_epochs >= 1 and n_negative >= 1, got {n_epochs}, {n_negative}")
    ramp = np.linspace(0, 1, n_epochs) ** RAMP_POWER if n_epochs > 1 else np.ones(1)
    plan = np.round(ramp / ramp.sum() * n_epochs * n_negative)
    return np.maximum(plan, MIN_NEGATIVES).astype(np.uint32)
