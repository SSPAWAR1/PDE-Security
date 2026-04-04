"""
Generic statistical helper functions.
"""

from __future__ import annotations

import numpy as np


def holm_correction(pvals: np.ndarray) -> np.ndarray:
    """
    Holm-Bonferroni family-wise error rate correction.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg false discovery rate correction.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ordered = p[order]

    adjusted_ordered = np.empty(m, dtype=float)
    running_min = 1.0

    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ordered[i] * m / rank
        running_min = min(running_min, val)
        adjusted_ordered[i] = min(running_min, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Standard Cohen's d for two independent samples.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")

    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))

    if np.isclose(pooled, 0.0):
        return 0.0

    return float((x.mean() - y.mean()) / pooled)


def paired_cohens_dz(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d_z for paired samples.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    delta = y - x
    if len(delta) < 2:
        return float("nan")

    sd = delta.std(ddof=1)
    if np.isclose(sd, 0.0):
        return 0.0

    return float(delta.mean() / sd)


def paired_signflip_permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> float:
    """
    Paired random sign-flip permutation test on within-pair deltas.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    delta = y - x
    obs = abs(delta.mean())

    hits = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(delta))
        stat = abs(np.mean(signs * delta))
        if stat >= obs:
            hits += 1

    return float((hits + 1.0) / (n_perm + 1.0))
