"""Kernel shape: fit (a, b) of q(d) = 1 / (1 + a d^(2b)) to UMAP's min_dist / spread curve.

The target is 1 below `min_dist` and exp(-(d - min_dist) / spread) above, sampled on
[0, 3 * spread], as in UMAP and NCVis. Solved with a few Levenberg-Marquardt steps in numpy, so
the package needs no scipy.
"""

import numpy as np

N_SAMPLES = 300
MAX_STEPS = 200
INITIAL_AB = (1.5, 0.9)
INITIAL_DAMPING = 1e-3
TOLERANCE = 1e-10


def _curve(d: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """q(d) and its Jacobian with respect to (a, b)."""
    safe_d = np.where(d > 0, d, 1.0)
    d_pow = np.where(d > 0, safe_d ** (2 * b), 0.0)
    log_d = np.where(d > 0, np.log(safe_d), 0.0)
    q = 1 / (1 + a * d_pow)
    dq_da = -d_pow * q**2
    dq_db = -a * d_pow * 2 * log_d * q**2
    return q, np.stack([dq_da, dq_db], axis=1)


def find_ab(spread: float = 1.0, min_dist: float = 0.001) -> tuple[float, float]:
    if spread <= 0 or not 0 <= min_dist < 3 * spread:
        raise ValueError(f"need spread > 0 and 0 <= min_dist < 3 * spread, got {spread}, {min_dist}")
    d = np.linspace(0, 3 * spread, N_SAMPLES)
    target = np.where(d < min_dist, 1.0, np.exp(-(d - min_dist) / spread))
    params = np.array(INITIAL_AB)
    damping = INITIAL_DAMPING
    q, jacobian = _curve(d, *params)
    cost = np.sum((q - target) ** 2)
    for _ in range(MAX_STEPS):
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ (q - target)
        delta = np.linalg.solve(normal + damping * np.diag(np.diag(normal)), -gradient)
        candidate = params + delta
        if candidate[0] <= 0 or candidate[1] <= 0:
            damping *= 10
            continue
        q_new, jacobian_new = _curve(d, *candidate)
        cost_new = np.sum((q_new - target) ** 2)
        if cost_new >= cost:
            damping *= 10
            continue
        converged = cost - cost_new < TOLERANCE
        params, q, jacobian, cost, damping = candidate, q_new, jacobian_new, cost_new, damping / 10
        if converged:
            break
    return float(params[0]), float(params[1])
