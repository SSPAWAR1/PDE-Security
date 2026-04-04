"""
Mutual-information plotting helpers.
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_mi_bars(
    feature_names: List[str],
    observed_mean: np.ndarray,
    observed_lo: np.ndarray,
    observed_hi: np.ndarray,
    null_dist: np.ndarray,
    title: str,
    outpath: str,
) -> None:
    """
    Plot observed mutual information with confidence intervals against
    a permutation-null baseline.
    """
    x = np.arange(len(feature_names))
    yerr = np.vstack([observed_mean - observed_lo, observed_hi - observed_mean])

    null_mean = null_dist.mean(axis=0)
    null_hi = np.percentile(null_dist, 97.5, axis=0)

    plt.figure(figsize=(10, 5))
    plt.bar(x, observed_mean, yerr=yerr, capsize=4, alpha=0.85, label="Observed MI")
    plt.plot(
        x,
        null_mean,
        marker="o",
        linestyle="--",
        label="Permutation-null mean",
    )
    plt.fill_between(
        x,
        0,
        null_hi,
        alpha=0.15,
        label="Permutation-null 95% upper band",
    )

    plt.xticks(x, feature_names, rotation=25, ha="right")
    plt.ylabel("Mutual information (bits)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
