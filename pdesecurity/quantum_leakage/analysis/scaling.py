"""
Scaling-law analysis helpers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def bootstrap_scaling_exponent(
    N_vals: np.ndarray,
    feat_vals: np.ndarray,
    n_boot: int = 300,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Fit log(feature) = k * log(N) + c using bootstrap resampling.
    Returns mean slope and 95% CI.
    """
    rng = np.random.default_rng(seed)

    logN = np.log(np.asarray(N_vals, dtype=float))
    logF = np.log(np.asarray(feat_vals, dtype=float) + 1e-8)

    ks = []
    n = len(logN)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x = logN[idx]
        y = logF[idx]

        k, _ = np.polyfit(x, y, 1)
        ks.append(k)

    ks = np.asarray(ks)
    return (
        float(ks.mean()),
        float(np.percentile(ks, 2.5)),
        float(np.percentile(ks, 97.5)),
    )
