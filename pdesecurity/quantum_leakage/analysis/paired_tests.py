"""
Paired-feature statistical testing utilities.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .stats import (
    benjamini_hochberg,
    holm_correction,
    paired_cohens_dz,
    paired_signflip_permutation_pvalue,
)


def paired_feature_tests(
    df: pd.DataFrame,
    feature_columns: List[str],
    pair_id_col: str = "sample_id",
    boundary_col: str = "boundary",
    a_label: str = "dirichlet",
    b_label: str = "periodic",
    n_perm: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run paired univariate tests comparing two matched conditions.
    """
    rows = []

    for feat in feature_columns:
        wide = (
            df[[pair_id_col, boundary_col, feat]]
            .pivot(index=pair_id_col, columns=boundary_col, values=feat)
            .dropna()
        )

        if a_label not in wide.columns or b_label not in wide.columns:
            raise ValueError(f"Expected labels '{a_label}' and '{b_label}' in pivoted data.")

        a_vals = wide[a_label].to_numpy(dtype=float)
        b_vals = wide[b_label].to_numpy(dtype=float)
        delta = b_vals - a_vals

        dz = paired_cohens_dz(a_vals, b_vals)
        perm_p = paired_signflip_permutation_pvalue(
            a_vals, b_vals, n_perm=n_perm, seed=seed
        )

        try:
            _, wilcox_p = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            wilcox_p = 1.0

        rows.append({
            "feature": feat,
            "mean_delta_b_minus_a": float(delta.mean()),
            "median_delta_b_minus_a": float(np.median(delta)),
            "cohen_dz": float(dz),
            "wilcoxon_p": float(wilcox_p),
            "paired_perm_p": float(perm_p),
        })

    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = holm_correction(out["wilcoxon_p"].to_numpy())
    out["paired_perm_p_holm"] = holm_correction(out["paired_perm_p"].to_numpy())
    out["wilcoxon_p_bh"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy())
    out["paired_perm_p_bh"] = benjamini_hochberg(out["paired_perm_p"].to_numpy())

    return out
