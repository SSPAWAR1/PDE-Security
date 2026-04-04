"""
Mutual-information utilities.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def compute_observed_mi_bits(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute observed mutual information in bits on the full dataset.
    """
    mi_nats = mutual_info_classif(
        X,
        y,
        discrete_features=False,
        random_state=seed,
    )
    return mi_nats / np.log(2.0)


def bootstrap_mi_bits(
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap confidence intervals for mutual information in bits.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    samples = []

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        yb = y[idx]

        mi_nats = mutual_info_classif(
            Xb,
            yb,
            discrete_features=False,
            random_state=seed + i,
        )
        samples.append(mi_nats / np.log(2.0))

    samples = np.asarray(samples)
    mean = samples.mean(axis=0)
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)

    return mean, lo, hi


def bootstrap_mi_bits_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Group-aware bootstrap for MI. Resamples groups, not rows.
    """
    rng = np.random.default_rng(seed)

    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}

    samples = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])

        Xb = X[idx]
        yb = y[idx]

        mi_nats = mutual_info_classif(
            Xb,
            yb,
            discrete_features=False,
            random_state=seed,
        )
        samples.append(mi_nats / np.log(2.0))

    samples = np.asarray(samples)
    mean = samples.mean(axis=0)
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)

    return mean, lo, hi, samples


def permutation_null_mi_bits(
    X: np.ndarray,
    y: np.ndarray,
    n_perm: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Standard permutation-null MI distribution.
    """
    rng = np.random.default_rng(seed)
    null_dist = []

    for i in range(n_perm):
        yp = rng.permutation(y)
        mi_nats = mutual_info_classif(
            X,
            yp,
            discrete_features=False,
            random_state=seed + i,
        )
        null_dist.append(mi_nats / np.log(2.0))

    return np.asarray(null_dist)


def paired_label_swap_null_mi_bits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_perm: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Pair-preserving label-swap null for matched-pair experiments.
    Assumes each group contains exactly 2 rows.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)

    pair_indices = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) != 2:
            raise ValueError("Each group must contain exactly 2 rows.")
        pair_indices.append(idx)

    null_dist = []
    for i in range(n_perm):
        yp = np.asarray(y).copy()
        for idx in pair_indices:
            if rng.random() < 0.5:
                yp[idx] = yp[idx[::-1]]

        mi_nats = mutual_info_classif(
            X,
            yp,
            discrete_features=False,
            random_state=seed + i,
        )
        null_dist.append(mi_nats / np.log(2.0))

    return np.asarray(null_dist)


def permutation_pvals(
    observed: np.ndarray,
    null_dist: np.ndarray,
) -> np.ndarray:
    """
    Compute permutation p-values feature-wise.
    """
    pvals = []
    for j in range(len(observed)):
        p = (1.0 + np.sum(null_dist[:, j] >= observed[j])) / (1.0 + len(null_dist))
        pvals.append(p)
    return np.asarray(pvals)
